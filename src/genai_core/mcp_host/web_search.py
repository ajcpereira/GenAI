# web_search.py
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

log = logging.getLogger("genai_core.mcp_host.web_search")

# ---------------------------------------------------------------------------
# Enterprise-grade goals
# - Provider-driven (no scraping by default)
# - Deterministic timeouts + bounded retries
# - Basic in-memory TTL cache (to reduce cost/latency)
# - Structured logs + safe diagnostics (no secrets)
# - Stable output contract: {"results":[{title,url,snippet}], "error":""}
# ---------------------------------------------------------------------------


# ----------------------------- Configuration -------------------------------

_BRAVE_DEFAULT_BASE_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True)
class WebSearchConfig:
    provider: str
    base_url: str
    api_key: str
    api_key_env: str
    timeout_s: float
    connect_timeout_s: float
    max_top_k: int
    cache_ttl_s: int
    cache_max_items: int
    user_agent: str
    # Scraping fallback is intentionally OFF by default.
    allow_html_scrape_fallback: bool


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def _load_config(args: Dict[str, Any]) -> WebSearchConfig:
    """
    Configuration precedence:
      1) Tool args (explicit per-call override)
      2) Environment variables (recommended for production)
      3) Hard defaults (safe, but you should set env vars)

    Providers:
      - brave: Brave Search API (direct)
      - gateway/internal_gateway/proxy: upstream gateway POST {base_url}/search
      - ddg_html: optional scraping-based provider (not recommended for prod)

    Secrets:
      - API keys must NOT be provided inline via YAML.
      - YAML/tool args should pass api_key_env (name of env var), defaulting to BRAVE_API_KEY.
      - The actual key is resolved from the process environment at runtime.
    """
    # Default provider: brave (direct, stable).
    provider = (args.get("provider") or os.getenv("WEB_SEARCH_PROVIDER") or "brave").strip().lower()

    base_url = (args.get("base_url") or os.getenv("WEB_SEARCH_BASE_URL") or "").strip()

    # Resolve API key only from environment variable name (config-driven)
    api_key_env = (
        args.get("api_key_env")
        or os.getenv("WEB_SEARCH_API_KEY_ENV")
        or "BRAVE_API_KEY"
    )
    api_key = (os.getenv(api_key_env) or "").strip()

    # Provider-specific safe default endpoint (only if caller didn't supply one)
    if not base_url and provider in ("brave", "brave_web", "brave_search"):
        base_url = _BRAVE_DEFAULT_BASE_URL

    timeout_s = float(args.get("timeout_s") or os.getenv("WEB_SEARCH_TIMEOUT_S") or 12.0)
    connect_timeout_s = float(args.get("connect_timeout_s") or os.getenv("WEB_SEARCH_CONNECT_TIMEOUT_S") or 6.0)

    max_top_k = int(args.get("max_top_k") or os.getenv("WEB_SEARCH_MAX_TOP_K") or 10)
    cache_ttl_s = int(args.get("cache_ttl_s") or os.getenv("WEB_SEARCH_CACHE_TTL_S") or 60)
    cache_max_items = int(args.get("cache_max_items") or os.getenv("WEB_SEARCH_CACHE_MAX_ITEMS") or 512)

    user_agent = (
        args.get("user_agent")
        or os.getenv("WEB_SEARCH_USER_AGENT")
        or "genai-core-web-search/1.0 (+https://example.invalid)"
    )

    allow_html_scrape_fallback = bool(
        args.get("allow_html_scrape_fallback")
        if "allow_html_scrape_fallback" in args
        else _env_bool("WEB_SEARCH_ALLOW_HTML_SCRAPE_FALLBACK", False)
    )

    # NOTE: WebSearchConfig must include api_key_env field for this return signature.
    return WebSearchConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        timeout_s=timeout_s,
        connect_timeout_s=connect_timeout_s,
        max_top_k=max(1, min(max_top_k, 50)),
        cache_ttl_s=max(0, cache_ttl_s),
        cache_max_items=max(0, cache_max_items),
        user_agent=user_agent,
        allow_html_scrape_fallback=allow_html_scrape_fallback,
    )


def _redact(s: str) -> str:
    if not s:
        return s
    # Basic redaction: avoid leaking api keys or bearer tokens in logs/errors
    s = re.sub(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*([^\s,;]+)", r"\1=<redacted>", s)
    return s


def _normalize_top_k(args: Dict[str, Any], cfg: WebSearchConfig) -> int:
    top_k = int(args.get("top_k") or args.get("k") or 5)
    if top_k <= 0:
        top_k = 5
    return min(top_k, cfg.max_top_k)


def _require_non_empty_query(query: str) -> Optional[str]:
    if not query or not query.strip():
        return "empty_query"
    # Hard limit to avoid abuse / accidental huge payloads
    if len(query) > 512:
        return "query_too_long"
    return None


# ----------------------------- TTL Cache ----------------------------------

class _TTLCache:
    def __init__(self, ttl_s: int, max_items: int):
        self.ttl_s = ttl_s
        self.max_items = max_items
        self._data: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        if self.ttl_s <= 0 or self.max_items <= 0:
            return None
        item = self._data.get(key)
        if not item:
            return None
        ts, val = item
        if (time.time() - ts) > self.ttl_s:
            self._data.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        if self.ttl_s <= 0 or self.max_items <= 0:
            return
        if len(self._data) >= self.max_items:
            # simple eviction: drop oldest
            oldest_key = min(self._data.items(), key=lambda kv: kv[1][0])[0]
            self._data.pop(oldest_key, None)
        self._data[key] = (time.time(), val)


_CACHE = _TTLCache(ttl_s=60, max_items=512)


def _cache_key(cfg: WebSearchConfig, query: str, top_k: int) -> str:
    # Do NOT include api_key
    return f"p={cfg.provider}|u={cfg.base_url}|q={query.strip().lower()}|k={top_k}"


# -------------------------- Provider Implementations -----------------------

def _safe_results(results: Any) -> List[Dict[str, str]]:
    """
    Ensure stable schema and sanitize types.
    """
    out: List[Dict[str, str]] = []
    if not isinstance(results, list):
        return out
    for r in results:
        if not isinstance(r, dict):
            continue
        title = str(r.get("title") or "").strip()
        url = str(r.get("url") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        if title and url:
            out.append({"title": title, "url": url, "snippet": snippet})
    return out


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)),
    wait=wait_exponential_jitter(initial=0.5, max=6.0),
    stop=stop_after_attempt(3),
)
async def _get(
    client: httpx.AsyncClient,
    url: str,
    params: Dict[str, Any],
    headers: Dict[str, str],
) -> httpx.Response:
    return await client.get(url, params=params, headers=headers)


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)),
    wait=wait_exponential_jitter(initial=0.5, max=6.0),
    stop=stop_after_attempt(3),
)
async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    json_body: Dict[str, Any],
    headers: Dict[str, str],
) -> httpx.Response:
    return await client.post(url, json=json_body, headers=headers)


async def _provider_gateway(cfg: WebSearchConfig, query: str, top_k: int, debug: bool) -> Dict[str, Any]:
    """
    Expected gateway contract (recommended):
      POST {base_url}/search
      body: {"query": "...", "top_k": 5}
      headers: Authorization: Bearer <api_key> (optional)
      response: {"results": [{"title","url","snippet"}], "error": ""}

    This keeps the MCP tool stable and moves vendor-specific logic upstream.
    """
    if not cfg.base_url:
        return {"results": [], "error": "misconfigured:missing_base_url"}

    url = cfg.base_url.rstrip("/") + "/search"
    headers: Dict[str, str] = {"User-Agent": cfg.user_agent}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    timeout = httpx.Timeout(cfg.timeout_s, connect=cfg.connect_timeout_s)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await _post_json(client, url, {"query": query, "top_k": top_k}, headers=headers)

    ctype = (r.headers.get("content-type") or "").lower()
    status = r.status_code

    if status >= 400:
        msg = f"upstream_http_{status}"
        if debug:
            msg += f" content_type={ctype}"
        return {"results": [], "error": msg}

    try:
        data = r.json()
    except Exception:
        return {"results": [], "error": "upstream_invalid_json"}

    results = _safe_results(data.get("results"))
    err = str(data.get("error") or "").strip()

    if results:
        return {"results": results, "error": ""}

    if err:
        return {"results": [], "error": _redact(err)}

    return {"results": [], "error": "no_results"}


async def _provider_brave(cfg: WebSearchConfig, query: str, top_k: int, debug: bool) -> Dict[str, Any]:
    """
    Brave Search API (Web Search):
      GET https://api.search.brave.com/res/v1/web/search?q=...&count=5
      Header: X-Subscription-Token: <API_KEY>

    Docs:
      - Auth header X-Subscription-Token required. :contentReference[oaicite:2]{index=2}
      - Endpoint examples. :contentReference[oaicite:3]{index=3}
    """
    if not cfg.api_key:
        return {"results": [], "error": f"misconfigured:missing_api_key env={cfg.api_key_env}"}
    if not cfg.base_url:
        return {"results": [], "error": "misconfigured:missing_base_url"}

    headers: Dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": cfg.user_agent,
        "X-Subscription-Token": cfg.api_key,
    }

    # Minimal request params: q + count
    params: Dict[str, Any] = {
        "q": query,
        "count": top_k,
    }

    timeout = httpx.Timeout(cfg.timeout_s, connect=cfg.connect_timeout_s)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await _get(client, cfg.base_url, params=params, headers=headers)

    status = r.status_code
    ctype = (r.headers.get("content-type") or "").lower()

    if status >= 400:
        msg = f"upstream_http_{status}"
        if debug:
            msg += f" content_type={ctype}"
            # Best-effort to extract error message (without dumping large bodies)
            try:
                j = r.json()
                err_msg = str(j.get("message") or j.get("error") or "").strip()
                if err_msg:
                    msg += f" msg={_redact(err_msg)}"
            except Exception:
                pass
        return {"results": [], "error": msg}

    try:
        data = r.json()
    except Exception:
        return {"results": [], "error": "upstream_invalid_json"}

    # Brave commonly returns results under data["web"]["results"]
    web = data.get("web") if isinstance(data, dict) else None
    items = None
    if isinstance(web, dict):
        items = web.get("results")

    if not isinstance(items, list):
        # Defensive fallback if schema changes
        items = data.get("results") if isinstance(data, dict) else None

    out: List[Dict[str, str]] = []
    if isinstance(items, list):
        for it in items[:top_k]:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            url = str(it.get("url") or it.get("link") or "").strip()
            # Brave fields vary; "description" is common for snippet-like text
            snippet = str(it.get("description") or it.get("snippet") or "").strip()
            if title and url:
                out.append({"title": title, "url": url, "snippet": snippet})

    if out:
        return {"results": out, "error": ""}

    return {"results": [], "error": "no_results"}


# ------------------------ Optional HTML scraping fallback -------------------

# DuckDuckGo HTML endpoint: fragile; use only if explicitly enabled.
_DDG_HTML = "https://html.duckduckgo.com/html/"
_UA_FALLBACK = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0 Safari/537.36"
)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<script.*?</script>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<style.*?</style>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<.*?>", "", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_ddg_results(html: str, top_k: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    link_re_a = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippet_re_a = re.compile(
        r'(?:class="result__snippet"[^>]*>|class="result__snippet[^"]*"[^>]*>)(.*?)(?:</a>|</div>)',
        flags=re.IGNORECASE | re.DOTALL,
    )

    link_re_b = re.compile(
        r'<a[^>]+class="result-link"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippet_re_b = re.compile(
        r'<(?:div|span)[^>]+class="result-snippet"[^>]*>(.*?)</(?:div|span)>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    links = list(link_re_a.finditer(html)) or list(link_re_b.finditer(html))
    snippets = list(snippet_re_a.finditer(html)) or list(snippet_re_b.finditer(html))

    n = min(len(links), max(0, top_k))
    for i in range(n):
        href = links[i].group(1).strip()
        title_html = links[i].group(2)
        title = _strip_tags(title_html)

        if "uddg=" in href:
            try:
                href = unquote(href.split("uddg=", 1)[1])
            except Exception:
                pass

        snippet = ""
        if i < len(snippets):
            snippet = _strip_tags(snippets[i].group(1))

        if title and href:
            out.append({"title": title, "url": href, "snippet": snippet})

    return out


async def _fallback_ddg_html(query: str, top_k: int, debug: bool) -> Dict[str, Any]:
    timeout = httpx.Timeout(12.0, connect=6.0)
    headers = {
        "User-Agent": _UA_FALLBACK,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.7",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            r = await client.post(_DDG_HTML, data={"q": query})
    except Exception as e:
        return {"results": [], "error": f"fallback_failed:{type(e).__name__}"}

    status = r.status_code
    ctype = (r.headers.get("content-type") or "").lower()
    html = r.text or ""

    results = _extract_ddg_results(html, top_k)
    if results:
        return {"results": results, "error": ""}

    # Diagnostics (sanitized)
    if debug:
        preview = re.sub(r"\s+", " ", html[:1200]).strip()
        signals = []
        low = html.lower()
        for pat in ("captcha", "challenge", "unusual", "blocked", "anomaly"):
            if pat in low:
                signals.append(pat)
        sig = ",".join(signals) if signals else "(none)"
        return {
            "results": [],
            "error": f"no_results status={status} content_type={ctype} signals={sig} html_preview={preview}",
        }

    return {"results": [], "error": f"no_results status={status} content_type={ctype}"}


# ------------------------------- Public API --------------------------------

async def web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool entrypoint.
    Input args:
      - query (str) [required]
      - k or top_k (int) [optional]
      - debug (bool) [optional]
      - provider/base_url/api_key/... [optional overrides; prefer env vars]

    Output:
      {"results":[{"title","url","snippet"}], "error":""}
    """
    t0 = time.time()
    debug = bool(args.get("debug", False))

    query = str(args.get("query") or "").strip()
    err = _require_non_empty_query(query)
    if err:
        return {"results": [], "error": err}

    cfg = _load_config(args)
    top_k = _normalize_top_k(args, cfg)

    # Refresh cache settings if env differs (simple approach)
    global _CACHE
    if _CACHE.ttl_s != cfg.cache_ttl_s or _CACHE.max_items != cfg.cache_max_items:
        _CACHE = _TTLCache(ttl_s=cfg.cache_ttl_s, max_items=cfg.cache_max_items)

    key = _cache_key(cfg, query, top_k)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # Provider dispatch
    provider = cfg.provider
    results: Dict[str, Any]

    try:
        if provider in ("brave", "brave_web", "brave_search"):
            results = await _provider_brave(cfg, query, top_k, debug)
        elif provider in ("gateway", "internal_gateway", "proxy"):
            results = await _provider_gateway(cfg, query, top_k, debug)
        elif provider in ("ddg_html", "duckduckgo_html", "ddg"):
            # Not recommended; often blocked. Keep only for lab / explicit enablement.
            results = await _fallback_ddg_html(query, top_k, debug)
        else:
            results = {"results": [], "error": f"unsupported_provider:{provider}"}

        # Optional last-resort fallback for POC environments (kept)
        if (
            (not results.get("results"))
            and cfg.allow_html_scrape_fallback
            and provider in ("brave", "brave_web", "brave_search", "gateway", "internal_gateway", "proxy")
        ):
            fb = await _fallback_ddg_html(query, top_k, debug)
            if fb.get("results"):
                results = fb
            else:
                if debug:
                    results = {
                        "results": [],
                        "error": f"{results.get('error','')} | fallback={fb.get('error','')}".strip(" |"),
                    }

    except Exception as e:
        results = {"results": [], "error": f"web_search_failed:{type(e).__name__}"}

    # Cache only successful or "no_results" responses (avoid caching transient upstream failures)
    e = str(results.get("error") or "")
    if results.get("results") or e in ("", "no_results"):
        _CACHE.set(key, results)

    # Structured log
    dur_ms = int((time.time() - t0) * 1000)
    safe_err = _redact(e)
    log.info(
        json.dumps(
            {
                "event": "web_search",
                "provider": provider,
                "top_k": top_k,
                "duration_ms": dur_ms,
                "results_count": len(results.get("results") or []),
                "error": safe_err,
            },
            ensure_ascii=False,
        )
    )

    # Ensure stable output contract
    return {
        "results": _safe_results(results.get("results")),
        "error": safe_err,
    }
