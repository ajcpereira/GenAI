#!/usr/bin/env python3
"""
trace_viewer.py

Developer trace viewer for GenAIv3.

Reads persisted envelopes from Postgres and prints a linear,
human-readable trace per request:

REQUEST
→ PLANNER
→ VALIDATOR
→ EXECUTOR
→ FINAL LLM INPUT
→ RESPONSE

Usage:
    python trace_viewer.py <session_id>

Requires:
    DATABASE_URL env var
"""

import os
import sys
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from collections import defaultdict


def die(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def short(obj, max_len=160):
    s = str(obj)
    return s if len(s) <= max_len else s[:max_len] + "…"


def normalize_envelope(raw):
    """
    Envelope may be:
      - dict (json/jsonb)
      - str  (JSON object serialized)
      - str  (double-encoded JSON, i.e. a JSON string that itself contains a JSON object)
    Normalize to dict (best-effort) or raise.
    """
    if raw is None:
        return {}

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")

    if isinstance(raw, str):
        s = raw.strip()
        # First decode
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            die(f"Invalid JSON envelope: {e}")

        # Some rows store JSON as a *string* that contains an object (double-encoded)
        if isinstance(obj, str):
            try:
                obj2 = json.loads(obj)
                obj = obj2
            except json.JSONDecodeError:
                # If it isn't valid JSON, keep as-is (will fail below)
                pass

        if isinstance(obj, dict):
            return obj

        die(f"Envelope decoded to non-object type: {type(obj).__name__}")

    die(f"Unsupported envelope type: {type(raw).__name__}")


def load_envelopes(session_id: str):
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        die("DATABASE_URL env var not set")

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    session_id,
                    request_id,
                    stage,
                    message_type,
                    source,
                    created_at,
                    envelope
                FROM envelopes
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            return cur.fetchall()
    finally:
        conn.close()


def print_request(request_id, rows):
    print("\n" + "=" * 100)
    print(f"REQUEST {request_id}")
    print("=" * 100)

    for row in rows:
        stage = row["stage"] or row["message_type"]
        created_at = row["created_at"]

        envelope = normalize_envelope(row["envelope"])
        metadata = envelope.get("metadata", {})
        payload = envelope.get("payload", {})

        print(f"\n[{stage.upper()}] {created_at}")

        # USER REQUEST
        if stage in ("request", "user_request"):
            print(f"message        : {payload.get('message')}")
            print(f"enabled_tools : {payload.get('enabled_tools')}")

        # PLANNER OUTPUT
        elif stage == "planner_output":
            ui = payload.get("user_intent", {})
            cp = ui.get("context_policy")
            plan = payload.get("plan")

            print(f"intent.summary : {ui.get('summary')}")
            print(f"intent.conf   : {ui.get('confidence')}")
            if cp:
                print(f"context_policy: {cp}")
            print(f"plan          : {short(plan)}")

        # CONTEXT POLICY OUTPUT (read-only, schema-driven)
        elif stage == "context_policy_output":
            print(f"current_user_message: {payload.get('current_user_message')}")
            decision = payload.get("decision") or {}
            if isinstance(decision, dict):
                print(
                    "decision      : "
                    + short(
                        {
                            "mode": decision.get("mode"),
                            "recent_turns": decision.get("recent_turns"),
                            "confidence": decision.get("confidence"),
                        },
                        max_len=300,
                    )
                )
            if payload.get("error"):
                print(f"error         : {payload.get('error')}")

        # VALIDATOR
        elif stage == "validator_output":
            validation = payload.get("validation", {})
            print(f"is_valid      : {validation.get('is_valid')}")
            if validation.get("errors"):
                print(f"errors        : {validation.get('errors')}")
            if validation.get("warnings"):
                print(f"warnings      : {validation.get('warnings')}")

        # EXECUTOR
        elif stage == "executor_result":
            steps = payload.get("steps_executed", [])
            for step in steps:
                print(f"step {step.get('id')} -> {step.get('status')}")
                out = step.get("output")
                if isinstance(out, dict) and out.get("tool_call"):
                    tc = out.get("tool_call") or {}
                    print(f"  tool        : {tc.get('capability')}")
                    if tc.get("inputs") is not None:
                        print(f"  inputs      : {short(tc.get('inputs'), max_len=260)}")

                    http = out.get("http")
                    if isinstance(http, dict):
                        resp = http.get("response") or {}
                        sc = resp.get("status_code")
                        em = http.get("elapsed_ms")
                        print(f"  http        : status={sc} elapsed_ms={em}")
                        req = http.get("request") or {}
                        if req.get("json") is not None:
                            print(f"  http.request: {short(req.get('json'), max_len=260)}")
                        if resp.get("json") is not None:
                            print(f"  http.response.json: {short(resp.get('json'), max_len=260)}")
                        elif resp.get("text") is not None:
                            print(f"  http.response.text: {short(resp.get('text'), max_len=260)}")

                    if out.get("data") is not None:
                        print(f"  data        : {short(out.get('data'), max_len=260)}")
                elif out is not None:
                    print(f"  output      : {short(out)}")
                if step.get("error"):
                    print(f"  error       : {step.get('error')}")

        # FINAL LLM INPUT
        elif stage == "final_llm_input":
            print(f"final_context : {short(payload.get('final_context'))}")
            print(f"output_spec  : {short(payload.get('final_output_spec'))}")

        # RESPONSE
        elif stage == "response":
            print(f"answer        : {payload.get('answer')}")

        # OTHER
        else:
            print(f"payload       : {short(payload)}")


def main():
    if len(sys.argv) != 2:
        die("usage: python trace_viewer.py <session_id>")

    session_id = sys.argv[1]
    rows = load_envelopes(session_id)

    if not rows:
        print(f"No envelopes found for session {session_id}")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["request_id"]].append(row)

    for request_id, req_rows in grouped.items():
        print_request(request_id, req_rows)


if __name__ == "__main__":
    main()
