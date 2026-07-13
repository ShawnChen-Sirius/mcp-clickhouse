"""Tests for the CHDB_SOURCES registry: parsing, DDL building, materialization.

Parsing tests are pure (no chDB needed). DDL-building and materialization
tests drive a real in-process chDB session and are skipped when the optional
``chdb`` extra is not installed. Sources are chosen so no network access is
required: file/url views carry an explicit structure (created lazily, without
schema inference), and the PostgreSQL database engine connects lazily.
"""

import json

import pytest

from mcp_clickhouse.chdb_sources import SourceSpec, parse_sources


# --- parsing (pure) ----------------------------------------------------------


def test_parse_multi_type_sources():
    raw = json.dumps(
        [
            {"name": "lake", "type": "s3", "url": "s3://b/x.parquet", "nosign": True},
            {"name": "events", "type": "file", "path": "/data/e.csv", "format": "CSVWithNames"},
            {
                "name": "appdb",
                "type": "postgresql",  # alias for postgres
                "host": "rds",
                "database": "app",
                "user": "ro",
                "password": "pw",
            },
            {
                "name": "warehouse",
                "type": "clickhouse",
                "host": "cloud",
                "database": "analytics",
                "table": "products",
                "user": "default",
            },
        ]
    )
    specs = parse_sources(raw)
    assert [s.name for s in specs] == ["lake", "events", "appdb", "warehouse"]
    assert specs[2].type == "postgres"  # alias normalized


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("not json", "not valid JSON"),
        ('{"name": "x"}', "must be a JSON array"),
        ('[{"name": "bad-name", "type": "file", "path": "/x"}]', "plain SQL identifier"),
        ('[{"name": "x", "type": "ftp", "path": "/x"}]', "unknown type"),
        ('[{"name": "x", "type": "file"}]', "requires"),
        ('[{"name": "x", "type": "file", "path": "/x", "fromat": "CSV"}]', "unknown parameters"),
        (
            '[{"name": "x", "type": "file", "path": "/x"}, {"name": "x", "type": "url", "url": "u"}]',
            "duplicate",
        ),
        ('[{"name": "x", "type": "s3", "url": "u", "access_key_id": "k"}]', "together"),
        (
            '[{"name": "x", "type": "s3", "url": "u", "nosign": true, '
            '"access_key_id": "k", "secret_access_key": "s"}]',
            "mutually exclusive",
        ),
        (
            '[{"name": "x", "type": "postgres", "host": "h", "database": "d", "user": "u", "port": "5432"}]',
            "integer",
        ),
        ('[{"name": "x", "type": "file", "path": 42}]', "must be a string"),
    ],
)
def test_parse_rejects_invalid_config(raw, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_sources(raw)


# --- DDL building (needs chdb.agents quoting) --------------------------------

chdb = pytest.importorskip("chdb")

from mcp_clickhouse.chdb_sources import _build_ddl, materialize  # noqa: E402


def test_ddl_file_with_structure_fills_format_slot():
    ddl = _build_ddl(SourceSpec("ev", "file", {"path": "/data/e.csv", "structure": "id Int64"}))
    assert (
        ddl
        == "CREATE OR REPLACE VIEW `ev` AS SELECT * FROM file('/data/e.csv', 'auto', 'id Int64')"
    )


def test_ddl_s3_nosign_is_unquoted_keyword():
    ddl = _build_ddl(SourceSpec("lake", "s3", {"url": "s3://b/x.parquet", "nosign": True}))
    assert "NOSIGN" in ddl and "'NOSIGN'" not in ddl


def test_ddl_quotes_values():
    ddl = _build_ddl(SourceSpec("ev", "file", {"path": "/da'ta/e.csv"}))
    # the quote in the path must be escaped, not break out of the literal
    assert "'/da'" not in ddl.replace("\\'", "")


def test_ddl_postgres_database_with_defaults():
    ddl = _build_ddl(
        SourceSpec("appdb", "postgres", {"host": "rds", "database": "app", "user": "ro"})
    )
    assert ddl == "CREATE DATABASE `appdb` ENGINE = PostgreSQL('rds:5432', 'app', 'ro', '')"


def test_ddl_clickhouse_secure_default():
    ddl = _build_ddl(
        SourceSpec(
            "wh",
            "clickhouse",
            {"host": "play", "database": "d", "table": "t", "user": "u", "password": "p"},
        )
    )
    assert ddl.startswith("CREATE OR REPLACE VIEW `wh` AS SELECT * FROM remoteSecure('play:9440'")


# --- materialization against a real session ----------------------------------

import chdb.session as chdb_session  # noqa: E402


@pytest.fixture
def session():
    s = chdb_session.Session()
    yield s
    s.close()


def _table_names(s):
    out = str(s.query("SELECT name FROM system.tables WHERE database = 'default'", "CSV"))
    return {line.strip().strip('"') for line in out.splitlines() if line.strip()}


def test_materialize_views_and_database(session):
    specs = parse_sources(
        json.dumps(
            [
                # structure given -> created lazily, no file/network access needed
                {
                    "name": "ev",
                    "type": "file",
                    "path": "/nonexistent/e.csv",
                    "structure": "id Int64",
                },
                {
                    "name": "web",
                    "type": "url",
                    "url": "https://example.invalid/d.csv",
                    "structure": "id Int64",
                },
                # PostgreSQL database engine connects lazily
                {
                    "name": "appdb",
                    "type": "postgres",
                    "host": "127.0.0.1",
                    "port": 15999,
                    "database": "app",
                    "user": "ro",
                    "password": "FAKE_PG_PW",
                },
            ]
        )
    )
    failures = materialize(session, specs)
    assert failures == []
    assert {"ev", "web"} <= _table_names(session)
    dbs = str(session.query("SELECT name FROM system.databases", "CSV"))
    assert "appdb" in dbs


def test_materialize_masks_credentials(session):
    specs = parse_sources(
        json.dumps(
            [
                {
                    "name": "lake",
                    "type": "s3",
                    "url": "https://bucketname.s3.amazonaws.com/x.parquet",
                    "access_key_id": "AKIAFAKE",
                    "secret_access_key": "SUPERSECRETVALUE",
                    "format": "Parquet",
                    "structure": "id Int64",
                },
                {
                    "name": "appdb2",
                    "type": "postgres",
                    "host": "127.0.0.1",
                    "port": 15999,
                    "database": "app",
                    "user": "ro",
                    "password": "PGSECRETVALUE",
                },
            ]
        )
    )
    assert materialize(session, specs) == []
    reflected = str(
        session.query(
            "SELECT create_table_query, as_select FROM system.tables WHERE name = 'lake'",
            "TabSeparated",
        )
    ) + str(
        session.query(
            "SELECT engine_full FROM system.databases WHERE name = 'appdb2'", "TabSeparated"
        )
    )
    assert "SUPERSECRETVALUE" not in reflected
    assert "PGSECRETVALUE" not in reflected
    assert "[HIDDEN]" in reflected


def test_materialize_refuses_file_outside_allowlist(session):
    specs = parse_sources(
        json.dumps([{"name": "ev", "type": "file", "path": "/etc/passwd", "structure": "l String"}])
    )
    failures = materialize(session, specs, file_allowlist=("/data",))
    assert len(failures) == 1 and "CHDB_FILE_ALLOWLIST" in failures[0][1]
    assert "ev" not in _table_names(session)


def test_materialize_refuses_to_replace_foreign_database(session):
    session.query("CREATE DATABASE IF NOT EXISTS localdb", "CSV")
    specs = parse_sources(
        json.dumps(
            [{"name": "localdb", "type": "postgres", "host": "h", "database": "d", "user": "u"}]
        )
    )
    failures = materialize(session, specs)
    assert len(failures) == 1 and "refusing" in failures[0][1]
    # the local database engine is untouched
    engine = str(session.query("SELECT engine FROM system.databases WHERE name='localdb'", "CSV"))
    assert "PostgreSQL" not in engine


def test_materialize_is_idempotent_for_database_proxies(session):
    specs = parse_sources(
        json.dumps(
            [{"name": "pgdb", "type": "postgres", "host": "h1", "database": "d", "user": "u"}]
        )
    )
    assert materialize(session, specs) == []
    # re-materializing (e.g. server restart with a persistent path) replaces the proxy
    specs2 = parse_sources(
        json.dumps(
            [{"name": "pgdb", "type": "postgres", "host": "h2", "database": "d", "user": "u"}]
        )
    )
    assert materialize(session, specs2) == []
    engine_full = str(
        session.query("SELECT engine_full FROM system.databases WHERE name='pgdb'", "TabSeparated")
    )
    assert "h2:5432" in engine_full


def test_materialize_collects_engine_failures(session):
    # MySQL database engine is not compiled into current chdb builds; the entry
    # must be skipped with an error, not raise out of materialize().
    specs = parse_sources(
        json.dumps(
            [
                {"name": "mdb", "type": "mysql", "host": "h", "database": "d", "user": "u"},
                {"name": "ok_view", "type": "file", "path": "/x.csv", "structure": "id Int64"},
            ]
        )
    )
    failures = materialize(session, specs)
    assert [name for name, _ in failures] in ([], ["mdb"])  # tolerate future builds adding MySQL
    assert "ok_view" in _table_names(session)
