"""
Built-in read-only research tools (blueprint capability "Research: search, compare,
analyze"), available even with no MCP servers configured.

  web.search(query, max_results)  -> list of {title, url, snippet}
  web.fetch(url, max_chars)       -> page title + readable text

SECURITY: these run on the backend server, so web.fetch is an SSRF risk. Every hop
(including redirects) must resolve ONLY to public IPs — loopback, private, link-local
(e.g. the cloud metadata address 169.254.169.254), multicast and reserved ranges are
refused — and only http(s) on ports 80/443 is allowed. Responses are size-capped.
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
import time
import urllib.parse as up

import httpx

from .base import ALLOW, ToolError, ToolResult, ToolSpec

MAX_BYTES = 1_000_000
UA = "Mozilla/5.0 (compatible; OnePieceAgent/0.3; +research tool)"


async def _assert_public(url: str, allow_private: bool = False) -> None:
    u = up.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolError("only http(s) URLs are allowed")
    port = u.port or (443 if u.scheme == "https" else 80)
    if port not in (80, 443) and not allow_private:
        raise ToolError("only ports 80/443 are allowed")
    if allow_private:
        return
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(u.hostname, port,
                                                             type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ToolError(f"cannot resolve host {u.hostname}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ToolError(f"refusing to fetch a non-public address ({u.hostname} -> {ip})")


def _html_to_text(doc: str) -> tuple[str, str]:
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", doc, re.S | re.I)
    if m:
        title = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()
    doc = re.sub(r"(?is)<(script|style|noscript|svg|template)[^>]*>.*?</\1>", " ", doc)
    doc = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|article)>", "\n", doc)
    text = html.unescape(re.sub(r"<[^>]+>", " ", doc))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return title, text


async def fetch_text(url: str, max_chars: int = 6000, allow_private: bool = False) -> tuple[str, str, str]:
    """GET with manual, re-validated redirects. Returns (final_url, title, text)."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=False,
                                 headers={"User-Agent": UA}) as client:
        for _ in range(4):
            await _assert_public(url, allow_private)
            async with client.stream("GET", url) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = up.urljoin(url, r.headers["location"])
                    continue
                if r.status_code >= 400:
                    raise ToolError(f"HTTP {r.status_code} from {url}")
                chunks, size = [], 0
                async for chunk in r.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        break
                    chunks.append(chunk)
                body = b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
                ctype = r.headers.get("content-type", "")
            if "html" in ctype or body.lstrip().startswith("<"):
                title, text = _html_to_text(body)
            else:
                title, text = "", body
            return url, title, text[:max_chars]
        raise ToolError("too many redirects")


async def web_fetch(args: dict, *, allow_private: bool = False) -> ToolResult:
    t0 = time.monotonic()
    url, title, text = await fetch_text(args["url"], int(args.get("max_chars") or 6000),
                                        allow_private)
    return ToolResult(True, f"{title}\n{url}\n\n{text}".strip(),
                      {"url": url, "title": title}, elapsed_s=time.monotonic() - t0)


_DDG_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'(?:<a[^>]+class="result__snippet"[^>]*>(.*?)</a>)?', re.S)


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _unwrap_ddg(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    q = up.parse_qs(up.urlparse(href).query)
    return q.get("uddg", [href])[0]


async def web_search(args: dict, *, search_url: str = "https://html.duckduckgo.com/html/") -> ToolResult:
    t0 = time.monotonic()
    query = str(args["query"]).strip()
    n = max(1, min(int(args.get("max_results") or 5), 10))
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}) as client:
        r = await client.post(search_url, data={"q": query})
    if r.status_code >= 400:
        raise ToolError(f"search failed: HTTP {r.status_code}")
    results = []
    for href, title, snippet in _DDG_RESULT.findall(r.text):
        results.append({"title": _clean(title), "url": _unwrap_ddg(href), "snippet": _clean(snippet)})
        if len(results) >= n:
            break
    if not results:
        return ToolResult(False, error="no results (the search engine may have rate-limited us)",
                          elapsed_s=time.monotonic() - t0)
    text = "\n".join(f"{i+1}. {x['title']} — {x['url']}\n   {x['snippet']}" for i, x in enumerate(results))
    return ToolResult(True, text, results, elapsed_s=time.monotonic() - t0)


def builtin_specs(allow_private_fetch: bool = False) -> list[ToolSpec]:
    async def _fetch(args: dict) -> ToolResult:
        return await web_fetch(args, allow_private=allow_private_fetch)

    return [
        ToolSpec(
            name="web.search", source="builtin", read_only=True, idempotent=True,
            policy=ALLOW, risk=1, handler=web_search,
            description="Search the web. Returns the top results with title, URL and snippet.",
            input_schema={"type": "object", "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 300},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10}},
                "required": ["query"], "additionalProperties": False}),
        ToolSpec(
            name="web.fetch", source="builtin", read_only=True, idempotent=True,
            policy=ALLOW, risk=1, handler=_fetch,
            description="Fetch a public web page and return its title and readable text.",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string", "pattern": "^https?://"},
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000}},
                "required": ["url"], "additionalProperties": False}),
    ]
