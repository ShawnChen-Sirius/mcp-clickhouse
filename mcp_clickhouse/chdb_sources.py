"""Config-declared external sources for the chDB engine (CHDB_SOURCES).

The operator declares sources (files, URLs, S3 buckets, Postgres/MySQL
databases, remote ClickHouse tables) as JSON in the ``CHDB_SOURCES``
environment variable. At server init — while the session is still writable,
before it is locked to ``readonly=2`` — each source is materialized as a named
view (single relations) or a named database (Postgres/MySQL, lazily
connected). Agents then query them like ordinary tables, with no credentials,
placeholders, or connection syntax in their SQL, and discover them through the
regular ``list_databases`` / ``list_tables`` / ``describe_table`` tools.

Credential handling: connection secrets appear only in this server-side config
and in the one CREATE statement run at init. The engine masks them as
``[HIDDEN]`` on every reflective surface an agent can reach (``SHOW CREATE``,
``system.tables.create_table_query`` / ``as_select``,
``system.databases.engine_full``), and the
``format_display_secrets_in_show_and_select`` escape hatch stays disabled
because the required server-side config switch is never set in embedded chDB.

Module layout mirrors chdb_safety/chdb_tools: importable without the optional
``chdb`` package (``parse_sources`` is pure); anything touching the engine or
``chdb.agents`` imports lazily.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Exposure names become plain SQL identifiers (view or database names); keeping
# them to this shape means they never need quoting tricks and cannot smuggle
# SQL. Names an agent must be able to type comfortably anyway.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# type -> (required params, optional params). "postgresql" and "http" are
# accepted as aliases for "postgres" and "url" during parsing.
_SOURCE_PARAMS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "file": (frozenset({"path"}), frozenset({"format", "structure"})),
    "url": (frozenset({"url"}), frozenset({"format", "structure"})),
    "s3": (
        frozenset({"url"}),
        frozenset({"access_key_id", "secret_access_key", "nosign", "format", "structure"}),
    ),
    "postgres": (
        frozenset({"host", "database", "user"}),
        frozenset({"port", "password", "schema"}),
    ),
    "mysql": (
        frozenset({"host", "database", "user"}),
        frozenset({"port", "password"}),
    ),
    "clickhouse": (
        frozenset({"host", "database", "table", "user"}),
        frozenset({"port", "password", "secure"}),
    ),
}

_TYPE_ALIASES = {"postgresql": "postgres", "http": "url"}

_DEFAULT_PORTS = {"postgres": 5432, "mysql": 3306}
# ClickHouse native protocol: TLS and plaintext listen on different ports.
_CLICKHOUSE_PORTS = {True: 9440, False: 9000}

# Source types materialized as CREATE DATABASE (lazily-connecting database
# proxies); everything else becomes a CREATE [OR REPLACE] VIEW.
_DATABASE_TYPES = frozenset({"postgres", "mysql"})


@dataclass(frozen=True)
class SourceSpec:
    """One validated CHDB_SOURCES entry: expose `params` of a `type` as `name`."""

    name: str
    type: str
    params: dict = field(hash=False)


def parse_sources(raw: str) -> list[SourceSpec]:
    """Parse and validate the CHDB_SOURCES JSON. Raises ValueError on any problem.

    Config errors are loud by design: a silently dropped source would surface
    to the agent as a missing table with no explanation, so a bad entry fails
    the whole registry (the caller then disables chDB with the error message
    rather than starting half-configured).
    """
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"CHDB_SOURCES is not valid JSON: {e}") from e
    if not isinstance(entries, list):
        raise ValueError("CHDB_SOURCES must be a JSON array of source objects")

    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"CHDB_SOURCES[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: each source must be a JSON object")
        name = entry.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(
                f"{where}: 'name' must be a plain SQL identifier ([A-Za-z_][A-Za-z0-9_]*), got {name!r}"
            )
        if name in seen:
            raise ValueError(f"{where}: duplicate source name {name!r}")
        seen.add(name)

        stype = entry.get("type")
        stype = _TYPE_ALIASES.get(stype, stype)
        if stype not in _SOURCE_PARAMS:
            raise ValueError(
                f"{where}: unknown type {entry.get('type')!r}; expected one of "
                f"{sorted(_SOURCE_PARAMS) + sorted(_TYPE_ALIASES)}"
            )

        params = {k: v for k, v in entry.items() if k not in ("name", "type")}
        required, optional = _SOURCE_PARAMS[stype]
        missing = required - params.keys()
        if missing:
            raise ValueError(f"{where}: type {stype!r} requires {sorted(missing)}")
        unknown = params.keys() - required - optional
        if unknown:
            # An unrecognized key is almost always a typo of a real one; taking
            # it silently would drop e.g. a misspelled 'structure' on the floor.
            raise ValueError(f"{where}: unknown parameters {sorted(unknown)} for type {stype!r}")
        for key, value in params.items():
            if key == "nosign" and stype == "s3":
                if not isinstance(value, bool):
                    raise ValueError(f"{where}: 'nosign' must be a boolean")
            elif key == "secure" and stype == "clickhouse":
                if not isinstance(value, bool):
                    raise ValueError(f"{where}: 'secure' must be a boolean")
            elif key == "port":
                if not isinstance(value, int) or isinstance(value, bool) or not (0 < value < 65536):
                    raise ValueError(f"{where}: 'port' must be an integer in [1, 65535]")
            elif not isinstance(value, str):
                raise ValueError(f"{where}: {key!r} must be a string")
        if stype == "s3" and params.get("nosign") and "access_key_id" in params:
            raise ValueError(f"{where}: 'nosign' and 'access_key_id' are mutually exclusive")
        if stype == "s3" and ("access_key_id" in params) != ("secret_access_key" in params):
            raise ValueError(
                f"{where}: 'access_key_id' and 'secret_access_key' must be provided together"
            )

        specs.append(SourceSpec(name=name, type=stype, params=params))
    return specs


def _build_ddl(spec: SourceSpec) -> str:
    """Return the CREATE statement that materializes `spec`.

    Values are embedded as quoted SQL string literals (via chdb.agents'
    canonical quoting); this one statement runs once, server-side, at init.

    The credential mechanism is deliberately contained in this function: once
    embedded chDB has stable ClickHouse named-collection support, the
    materialization can switch to collection references here without changing
    the CHDB_SOURCES config surface.
    """
    from chdb.agents import quote_string

    q = quote_string
    p = spec.params
    name = f"`{spec.name}`"

    if spec.type == "file":
        args = [q(p["path"])]
        # file(path[, format[, structure]]) — a structure needs the format slot
        # filled, and 'auto' keeps the engine's format detection.
        if "structure" in p:
            args += [q(p.get("format", "auto")), q(p["structure"])]
        elif "format" in p:
            args.append(q(p["format"]))
        return f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM file({', '.join(args)})"

    if spec.type == "url":
        args = [q(p["url"])]
        if "structure" in p:
            args += [q(p.get("format", "auto")), q(p["structure"])]
        elif "format" in p:
            args.append(q(p["format"]))
        return f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM url({', '.join(args)})"

    if spec.type == "s3":
        args = [q(p["url"])]
        if p.get("nosign"):
            args.append("NOSIGN")  # engine keyword, deliberately unquoted
        elif "access_key_id" in p:
            args += [q(p["access_key_id"]), q(p["secret_access_key"])]
        if "structure" in p:
            args += [q(p.get("format", "auto")), q(p["structure"])]
        elif "format" in p:
            args.append(q(p["format"]))
        return f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM s3({', '.join(args)})"

    if spec.type in _DATABASE_TYPES:
        engine = "PostgreSQL" if spec.type == "postgres" else "MySQL"
        addr = f"{p['host']}:{p.get('port', _DEFAULT_PORTS[spec.type])}"
        args = [q(addr), q(p["database"]), q(p["user"]), q(p.get("password", ""))]
        if spec.type == "postgres" and "schema" in p:
            args.append(q(p["schema"]))
        return f"CREATE DATABASE {name} ENGINE = {engine}({', '.join(args)})"

    if spec.type == "clickhouse":
        secure = p.get("secure", True)
        fn = "remoteSecure" if secure else "remote"
        addr = f"{p['host']}:{p.get('port', _CLICKHOUSE_PORTS[secure])}"
        args = [q(addr), q(p["database"]), q(p["table"]), q(p["user"]), q(p.get("password", ""))]
        return f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {fn}({', '.join(args)})"

    raise ValueError(f"unhandled source type {spec.type!r}")  # unreachable after parse


def _database_conflict(client, spec: SourceSpec) -> str | None:
    """Refuse to (re)claim a database name that holds something else.

    Re-materializing a database proxy means DROP + CREATE, and DROP DATABASE on
    a name the registry does not own could destroy local user tables. Dropping
    a PostgreSQL/MySQL proxy only discards the proxy, never remote data.
    Returns an error string on conflict, None when the name is free or ours.
    """
    from chdb.agents import quote_string

    res = client.query(
        f"SELECT engine FROM system.databases WHERE name = {quote_string(spec.name)}",
        "TabSeparated",
    )
    engine = str(res).strip()
    if not engine:
        return None
    expected = "PostgreSQL" if spec.type == "postgres" else "MySQL"
    if engine != expected:
        return (
            f"database {spec.name!r} already exists with engine {engine} "
            f"(not a {expected} proxy); refusing to replace it"
        )
    client.query(f"DROP DATABASE IF EXISTS `{spec.name}`", "TabSeparated")
    return None


def materialize(client, specs: list[SourceSpec], file_allowlist=()) -> list[tuple[str, str]]:
    """Materialize each source on the (still-writable) session; return failures.

    Per-source failures are collected, not raised: one unreachable or
    unsupported source (e.g. a MySQL proxy on a build without the MySQL
    database engine) should not take down the sources that do work. The caller
    logs the returned ``(name, error)`` pairs.

    ``file_allowlist``: when the operator also set CHDB_FILE_ALLOWLIST, a
    file-type source outside it is refused here for consistency — the registry
    must not become a loophole around the raw-SQL path sandbox.
    """
    failures: list[tuple[str, str]] = []
    for spec in specs:
        try:
            if spec.type == "file" and file_allowlist:
                from chdb.agents.safety import path_allowed

                if not path_allowed(spec.params["path"], list(file_allowlist)):
                    failures.append(
                        (spec.name, f"path {spec.params['path']!r} is outside CHDB_FILE_ALLOWLIST")
                    )
                    continue
            if spec.type in _DATABASE_TYPES:
                conflict = _database_conflict(client, spec)
                if conflict:
                    failures.append((spec.name, conflict))
                    continue
            res = client.query(_build_ddl(spec), "TabSeparated")
            if hasattr(res, "has_error") and res.has_error():
                failures.append((spec.name, str(res.error_message())))
        except Exception as e:  # noqa: BLE001 — engine errors become skip reasons
            failures.append((spec.name, str(e)))
    return failures
