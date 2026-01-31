# orchestrator/session_store.py
import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

logger = logging.getLogger("genai.session_store")


@dataclass(frozen=True)
class Checkpoint:
    covers_seq_end: int
    summary: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class MessageRow:
    seq: int
    role: str
    content: str
    # Links a message to its execution trace/envelopes.
    # May be NULL/absent for legacy rows.
    request_id: Optional[str] = None


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    user_id: Optional[str]
    created_at: str
    last_seen_at: str
    message_count: int
    meta: Dict[str, Any]


class SessionStore:
    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def ensure_schema(self) -> None:
        raise NotImplementedError

    async def touch_session(self, session_id: str, user_id: Optional[str], meta: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError

    async def append_message(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        request_id: str,
        role: str,
        content: str,
    ) -> int:
        """Append a message and return the assigned seq number (monotonic per session)."""
        raise NotImplementedError

    async def get_latest_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        raise NotImplementedError

    async def upsert_checkpoint(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        covers_seq_end: int,
        summary: str,
        meta: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    async def get_messages_range(
        self,
        *,
        session_id: str,
        seq_start: int,
        seq_end: int,
    ) -> List[MessageRow]:
        raise NotImplementedError

    async def get_recent_messages(
        self,
        *,
        session_id: str,
        before_seq: int,
        limit: int,
    ) -> List[MessageRow]:
        raise NotImplementedError

    async def persist_envelope(
        self,
        *,
        stage: str,
        envelope: Dict[str, Any],
        max_bytes: int,
    ) -> None:
        raise NotImplementedError

    async def list_envelopes_for_request(self, *, request_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Return persisted envelopes (all stages) for a given request_id."""
        raise NotImplementedError


class PostgresSessionStore(SessionStore):
    def __init__(
        self,
        *,
        dsn: str,
        pool_min: int = 1,
        pool_max: int = 5,
        connect_timeout_s: float = 5.0,
        statement_timeout_ms: int = 20000,
    ):
        self.dsn = dsn
        self.pool_min = int(pool_min)
        self.pool_max = int(pool_max)
        self.connect_timeout_s = float(connect_timeout_s)
        self.statement_timeout_ms = int(statement_timeout_ms)
        self._pool: Optional[asyncpg.Pool] = None

    async def start(self) -> None:
        if self._pool is not None:
            return

        async def _init_conn(conn: asyncpg.Connection) -> None:
            # Ensure statement timeout is applied to every pool connection.
            await conn.execute(f"SET statement_timeout = {self.statement_timeout_ms};")

            # Make json/jsonb round-trip as Python objects (dict/list) instead of raw strings.
            await conn.set_type_codec("json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
            await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")

        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.pool_min,
            max_size=self.pool_max,
            timeout=self.connect_timeout_s,
            command_timeout=self.connect_timeout_s,
            init=_init_conn,
        )

    async def stop(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresSessionStore not started")
        return self._pool

    async def ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS sessions (
          session_id TEXT PRIMARY KEY,
          user_id TEXT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          meta JSONB NOT NULL DEFAULT '{}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS messages (
          id BIGSERIAL PRIMARY KEY,
          session_id TEXT NOT NULL,
          user_id TEXT NULL,
          request_id TEXT NOT NULL,
          seq INTEGER NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(session_id, seq)
        );

        CREATE INDEX IF NOT EXISTS messages_session_seq_idx
          ON messages (session_id, seq);

        CREATE TABLE IF NOT EXISTS context_artifacts (
          id BIGSERIAL PRIMARY KEY,
          session_id TEXT NOT NULL,
          user_id TEXT NULL,
          type TEXT NOT NULL,
          covers_seq_end INTEGER NOT NULL,
          content TEXT NOT NULL,
          meta JSONB NOT NULL DEFAULT '{}'::jsonb,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE(session_id, type, covers_seq_end)
        );

        CREATE INDEX IF NOT EXISTS context_artifacts_session_type_idx
          ON context_artifacts (session_id, type, covers_seq_end DESC);

        CREATE TABLE IF NOT EXISTS envelopes (
          id BIGSERIAL PRIMARY KEY,
          session_id TEXT NOT NULL,
          request_id TEXT NOT NULL,
          stage TEXT NOT NULL,
          message_type TEXT NOT NULL,
          source TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          envelope JSONB NOT NULL,
          truncated BOOLEAN NOT NULL DEFAULT FALSE
        );

        CREATE INDEX IF NOT EXISTS envelopes_session_stage_idx
          ON envelopes (session_id, stage, created_at DESC);

        CREATE INDEX IF NOT EXISTS envelopes_request_id_idx
          ON envelopes (request_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS envelopes_request_stage_idx
          ON envelopes (request_id, stage);
        """
        async with self.pool.acquire() as conn:
            await conn.execute(ddl)

    async def touch_session(self, session_id: str, user_id: Optional[str], meta: Optional[Dict[str, Any]] = None) -> None:
        # asyncpg expects str/bytes for $3::jsonb unless you register JSON codecs.
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        q = """
        INSERT INTO sessions(session_id, user_id, meta)
        VALUES($1, $2, $3::jsonb)
        ON CONFLICT (session_id) DO UPDATE
          SET user_id = COALESCE(EXCLUDED.user_id, sessions.user_id),
              last_seen_at = now(),
              meta = sessions.meta || EXCLUDED.meta;
        """
        async with self.pool.acquire() as conn:
            await conn.execute(q, session_id, user_id, meta_json)

    async def _next_seq(self, conn: asyncpg.Connection, session_id: str) -> int:
        row = await conn.fetchrow("SELECT COALESCE(MAX(seq), 0) AS m FROM messages WHERE session_id=$1;", session_id)
        return int(row["m"]) + 1

    async def append_message(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        request_id: str,
        role: str,
        content: str,
    ) -> int:
        q = """
        INSERT INTO messages(session_id, user_id, request_id, seq, role, content)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING seq;
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                seq = await self._next_seq(conn, session_id)
                row = await conn.fetchrow(q, session_id, user_id, request_id, int(seq), str(role), str(content))
                return int(row["seq"])

    async def get_latest_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        q = """
        SELECT covers_seq_end, content, meta
        FROM context_artifacts
        WHERE session_id=$1 AND type='summary_checkpoint'
        ORDER BY covers_seq_end DESC
        LIMIT 1;
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(q, session_id)
            if not row:
                return None
            return Checkpoint(
                covers_seq_end=int(row["covers_seq_end"]),
                summary=str(row["content"]),
                meta=dict(row["meta"] or {}),
            )

    async def upsert_checkpoint(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        covers_seq_end: int,
        summary: str,
        meta: Dict[str, Any],
    ) -> None:
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        q = """
        INSERT INTO context_artifacts(session_id, user_id, type, covers_seq_end, content, meta)
        VALUES($1,$2,'summary_checkpoint',$3,$4,$5::jsonb)
        ON CONFLICT (session_id, type, covers_seq_end)
        DO UPDATE SET content=EXCLUDED.content, meta=EXCLUDED.meta, created_at=now();
        """
        async with self.pool.acquire() as conn:
            await conn.execute(q, session_id, user_id, int(covers_seq_end), summary, meta_json)

    async def get_messages_range(self, *, session_id: str, seq_start: int, seq_end: int) -> List[MessageRow]:
        q = """
        SELECT seq, role, content, request_id
        FROM messages
        WHERE session_id=$1 AND seq >= $2 AND seq <= $3
        ORDER BY seq ASC;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, session_id, int(seq_start), int(seq_end))
            return [
                MessageRow(
                    seq=int(r["seq"]),
                    role=str(r["role"]),
                    content=str(r["content"]),
                    request_id=str(r["request_id"]) if r["request_id"] is not None else None,
                )
                for r in rows
            ]

    async def get_recent_messages(self, *, session_id: str, before_seq: int, limit: int) -> List[MessageRow]:
        q = """
        SELECT seq, role, content, request_id
        FROM messages
        WHERE session_id=$1 AND seq < $2
        ORDER BY seq DESC
        LIMIT $3;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, session_id, int(before_seq), int(limit))
            rows2 = list(reversed(rows))
            return [
                MessageRow(
                    seq=int(r["seq"]),
                    role=str(r["role"]),
                    content=str(r["content"]),
                    request_id=str(r["request_id"]) if r["request_id"] is not None else None,
                )
                for r in rows2
            ]

    # ---------------------------
    # UI helpers (NEW)
    # ---------------------------

    @staticmethod
    def _coerce_json_dict(val: Any) -> Dict[str, Any]:
        # Best-effort conversion for json/jsonb columns (handles legacy string storage).

        if val is None:
            return {}
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                obj = json.loads(val)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

    async def list_sessions(self, *, limit: int = 50, offset: int = 0) -> List[SessionRow]:
        q = """
        SELECT
          s.session_id,
          s.user_id,
          s.created_at,
          s.last_seen_at,
          s.meta,
          COALESCE(m.cnt, 0) AS message_count
        FROM sessions s
        LEFT JOIN (
          SELECT session_id, COUNT(*)::int AS cnt
          FROM messages
          GROUP BY session_id
        ) m ON m.session_id = s.session_id
        ORDER BY s.last_seen_at DESC
        LIMIT $1 OFFSET $2;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, int(limit), int(offset))
            out: List[SessionRow] = []
            for r in rows:
                out.append(
                    SessionRow(
                        session_id=str(r["session_id"]),
                        user_id=str(r["user_id"]) if r["user_id"] is not None else None,
                        created_at=str(r["created_at"]),
                        last_seen_at=str(r["last_seen_at"]),
                        message_count=int(r["message_count"] or 0),
                        meta=self._coerce_json_dict(r["meta"]),
                    )
                )
            return out

    async def get_messages_for_session(self, *, session_id: str, limit: int = 500) -> List[MessageRow]:
        q = """
        SELECT seq, role, content, request_id
        FROM messages
        WHERE session_id=$1
        ORDER BY seq ASC
        LIMIT $2;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, str(session_id), int(limit))
            return [
                MessageRow(
                    seq=int(r["seq"]),
                    role=str(r["role"]),
                    content=str(r["content"]),
                    request_id=str(r["request_id"]) if r["request_id"] is not None else None,
                )
                for r in rows
            ]

    async def delete_session(self, *, session_id: str) -> None:
        # cascade manual (messages, artifacts, envelopes, sessions)
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages WHERE session_id=$1;", str(session_id))
                await conn.execute("DELETE FROM context_artifacts WHERE session_id=$1;", str(session_id))
                await conn.execute("DELETE FROM envelopes WHERE session_id=$1;", str(session_id))
                await conn.execute("DELETE FROM sessions WHERE session_id=$1;", str(session_id))

    async def persist_envelope(
        self,
        *,
        stage: str,
        envelope: Dict[str, Any],
        max_bytes: int,
    ) -> None:
        raw = json.dumps(envelope, ensure_ascii=False)
        truncated = False
        if len(raw.encode("utf-8")) > int(max_bytes):
            truncated = True
            env2 = dict(envelope)
            payload = env2.get("payload")
            if isinstance(payload, dict):
                for k in ("message", "content", "text"):
                    if k in payload and isinstance(payload[k], str):
                        payload[k] = payload[k][:2000] + "…(truncated)"
            raw = json.dumps(env2, ensure_ascii=False)

        meta = envelope.get("metadata") or {}
        session_id = meta.get("session_id") or ""
        request_id = meta.get("request_id") or ""
        message_type = meta.get("message_type") or ""
        source = meta.get("source") or ""

        q = """
        INSERT INTO envelopes(session_id, request_id, stage, message_type, source, envelope, truncated)
        VALUES($1, $2, $3, $4, $5, $6::jsonb, $7);
        """
        async with self.pool.acquire() as conn:
            await conn.execute(q, str(session_id), str(request_id), str(stage), str(message_type), str(source), raw, bool(truncated))

    async def list_envelopes_for_request(self, *, request_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        q = """
        SELECT stage, message_type, source, created_at, truncated, envelope
        FROM envelopes
        WHERE request_id=$1
        ORDER BY created_at ASC
        LIMIT $2;
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, str(request_id), int(limit))
            out: List[Dict[str, Any]] = []
            for r in rows:
                env = r["envelope"]
                # Com codecs JSONB configurados, isto deve ser dict. Mantemos fallback.
                if isinstance(env, str):
                    try:
                        env = json.loads(env)
                    except Exception:
                        env = {"raw": env}
                out.append(
                    {
                        "stage": str(r["stage"]),
                        "message_type": str(r["message_type"]),
                        "source": str(r["source"]),
                        "created_at": str(r["created_at"]),
                        "truncated": bool(r["truncated"]),
                        "envelope": env,
                    }
                )
            return out
