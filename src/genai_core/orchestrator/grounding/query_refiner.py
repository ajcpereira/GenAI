from __future__ import annotations

from typing import List, Optional


def build_query_ladder(question_type: str, raw: str, entity_hint: str) -> List[str]:
    """
    Build a bounded set of queries: broad -> more constrained.
    Use search operators carefully; keep queries short and safe.
    """
    q0 = raw.strip()
    e = (entity_hint or "").strip()

    ladder: List[str] = [q0]

    if question_type == "date":
        # “internet time” without a dedicated tool: lean on common phrasing
        ladder = [
            "what is today's date",
            "current date today",
            "today date Portugal",
        ]
        return ladder

    if question_type == "latest_version":
        if e:
            ladder = [
                f"{e} latest version",
                f"{e} latest release version",
                f"{e} latest version site:pypi.org OR site:github.com OR site:docs",
            ]
        else:
            ladder = [
                f"{q0}",
                f"{q0} latest version",
                f"{q0} release notes",
            ]
        return ladder

    if question_type == "ownership":
        if e:
            ladder = [
                f"{e} largest shareholder",
                f"{e} top shareholders institutional ownership",
                f"{e} DEF 14A largest shareholders site:sec.gov",
            ]
        else:
            ladder = [q0, f"{q0} largest shareholder", f"{q0} site:sec.gov"]
        return ladder

    if question_type == "acquisition":
        if e:
            ladder = [
                f"{e} latest acquisition",
                f"{e} acquisition announced press release",
                f"{e} acquisition press release site:{e}.com OR site:investor.{e}.com",
            ]
        else:
            ladder = [q0, f"{q0} acquisition announced", f"{q0} press release"]
        return ladder

    if question_type == "news":
        if e:
            ladder = [f"{e} latest news", f"{e} news today", f"{e} official announcement press release"]
        else:
            ladder = [q0, f"{q0} latest news", f"{q0} official announcement"]
        return ladder

    # generic_fact: keep it simple
    if e and e not in q0.lower():
        ladder.append(f"{e} {q0}")

    return ladder


def infer_domain_allowlist(question_type: str, entity_hint: str) -> Optional[List[str]]:
    """
    Optional allowlists (use sparingly).
    This is a safe place to add well-known official domains, but keep it minimal.
    """
    e = (entity_hint or "").lower().strip()
    if not e:
        return None

    if question_type == "latest_version":
        if e == "ubuntu":
            return ["ubuntu.com", "canonical.com", "wiki.ubuntu.com", "releases.ubuntu.com"]
        return None

    if question_type == "acquisition":
        # Company sites vary; don't over-constrain.
        return None

    if question_type == "ownership":
        return ["sec.gov"]

    return None
