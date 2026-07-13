"""End-to-end tests for the co-located ``dataframe_query`` tool.

Same harness as test_chdb_tools: a real FastMCP server driven through an
in-memory client, over a real chDB session locked to readonly=2 (mirroring the
server init flow). Skipped when chdb or pandas is not installed.
"""

import concurrent.futures
import json

import pytest

chdb_session = pytest.importorskip("chdb.session")
pd = pytest.importorskip("pandas")
from fastmcp import Client, FastMCP  # noqa: E402

from mcp_clickhouse.chdb_tools import (  # noqa: E402
    register_chdb_only_tools,
    register_dataframe,
    registered_dataframes,
    unregister_dataframe,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def chdb_client():
    session = chdb_session.Session()
    session.query("SET readonly=2", "TabSeparated")
    yield session
    session.close()


@pytest.fixture(scope="module")
def mcp_with_tools(chdb_client):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    mcp = FastMCP(name="chdb-df-test")
    register_chdb_only_tools(
        mcp,
        max_result_bytes=1024 * 1024,
        create_client=lambda: chdb_client,
        query_executor=executor,
        query_timeout=lambda: 30,
    )
    yield mcp
    executor.shutdown(wait=True)


@pytest.fixture(autouse=True)
def orders_frame():
    register_dataframe("orders", pd.DataFrame({"category": ["a", "a", "b"], "price": [10, 20, 30]}))
    yield
    unregister_dataframe("orders")


async def _call(mcp, tool, **args):
    async with Client(mcp) as client:
        result = await client.call_tool(tool, args)
        return json.loads(result.content[0].text)


def _register_on(mcp, chdb_client):
    register_chdb_only_tools(
        mcp,
        max_result_bytes=1024,
        create_client=lambda: chdb_client,
        query_executor=concurrent.futures.ThreadPoolExecutor(max_workers=1),
        query_timeout=lambda: 30,
    )


async def test_tool_appears_when_first_frame_is_published(chdb_client, monkeypatch):
    # Publishing the first frame is the co-located signal: before it the tool
    # is not advertised, after it the tool is live on the already-set-up server.
    from mcp_clickhouse import chdb_tools

    monkeypatch.setattr(chdb_tools, "_DATAFRAMES", {})
    monkeypatch.setattr(chdb_tools, "_DATAFRAME_TOOL_ADDERS", [])
    mcp = FastMCP(name="chdb-df-dynamic")
    _register_on(mcp, chdb_client)
    assert "dataframe_query" not in await mcp.get_tools()
    register_dataframe("tmp_dyn", pd.DataFrame({"x": [1]}))
    assert "dataframe_query" in await mcp.get_tools()


async def test_tool_present_when_frame_published_before_setup(chdb_client, monkeypatch):
    from mcp_clickhouse import chdb_tools

    monkeypatch.setattr(chdb_tools, "_DATAFRAMES", {})
    monkeypatch.setattr(chdb_tools, "_DATAFRAME_TOOL_ADDERS", [])
    register_dataframe("tmp_pre", pd.DataFrame({"x": [1]}))
    mcp = FastMCP(name="chdb-df-preset")
    _register_on(mcp, chdb_client)
    assert "dataframe_query" in await mcp.get_tools()


async def test_dataframe_query_aggregates(mcp_with_tools):
    rows = await _call(
        mcp_with_tools,
        "dataframe_query",
        query="SELECT category, avg(price) AS p FROM {df} GROUP BY category ORDER BY category",
        df_ref="orders",
    )
    assert rows == [{"category": "a", "p": 15}, {"category": "b", "p": 30}]


async def test_unknown_ref_lists_registered(mcp_with_tools):
    with pytest.raises(Exception, match="orders"):
        await _call(mcp_with_tools, "dataframe_query", query="SELECT * FROM {df}", df_ref="nope")


async def test_query_must_use_df_placeholder(mcp_with_tools):
    with pytest.raises(Exception, match=r"\{df\}"):
        await _call(mcp_with_tools, "dataframe_query", query="SELECT 1", df_ref="orders")


async def test_ref_must_be_identifier(mcp_with_tools):
    with pytest.raises(Exception, match="identifier"):
        await _call(mcp_with_tools, "dataframe_query", query="SELECT * FROM {df}", df_ref="x; DROP")


async def test_register_validates_name():
    with pytest.raises(ValueError, match="identifier"):
        register_dataframe("not an identifier", pd.DataFrame())


async def test_unregister_and_listing():
    register_dataframe("tmp_frame", pd.DataFrame({"x": [1]}))
    assert "tmp_frame" in registered_dataframes()
    unregister_dataframe("tmp_frame")
    assert "tmp_frame" not in registered_dataframes()
    unregister_dataframe("tmp_frame")  # unknown names are ignored
