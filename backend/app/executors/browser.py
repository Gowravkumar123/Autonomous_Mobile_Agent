"""
Browser executor (blueprint: "Browser adapter — Playwright, isolated context, domain
restrictions, screenshot/DOM evidence, download controls").

`BrowserSession` speaks EXACTLY the same contract as the phone's `ws.DeviceSession`:
    request("observe") -> {"screen": ScreenObservation-dict}
    request("act", action=AgentAction-dict) -> {"ok": bool}
    request("handoff", ...) -> {"ok": True, "stop": True, "message": ...}
    push(type, **data)   (forwarded to the task owner's phone, if any)
so the ReAct graph, policy engine, transaction-bound approvals and verifier drive a
website with no changes. The DOM is serialized into the same UiElement shape the phone
uses (ref / parent / depth / role / text / clickable / editable / bounds).

Safety controls:
- One fresh, isolated browser CONTEXT per task (no shared cookies/storage); closed after.
- Downloads disabled. JS dialogs (alert/confirm) are DISMISSED — a "Confirm booking?"
  dialog must never be accepted implicitly.
- Optional domain allowlist enforced on every top-level navigation.
- Payment/auth (Level 4) can't be completed by a headless browser on the user's behalf,
  so a handoff STOPS the task and tells the user where to finish.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import urllib.parse as up
from typing import Any, Awaitable, Callable

log = logging.getLogger("agent.browser")

# Walk the visible DOM; tag elements with data-agent-ref so actions can target them.
_SERIALIZE_JS = r"""
(maxEls) => {
  const INTERACTIVE = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],'
    + '[role=tab],[role=menuitem],[role=checkbox],[role=radio],[role=option],[onclick],'
    + '[contenteditable=""],[contenteditable=true],label[for]';
  const NONCLICK_INPUT = ['text','search','email','tel','url','password','number','date','time'];
  document.querySelectorAll('[data-agent-ref]').forEach(e => e.removeAttribute('data-agent-ref'));
  const vis = (el) => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none'
      && parseFloat(s.opacity || '1') > 0.05 && r.bottom > -innerHeight && r.top < innerHeight * 3;
  };
  const out = []; let next = 0;
  const walk = (node, depth, parentRef, inInteractive) => {
    for (const el of node.children) {
      if (out.length >= maxEls) return;
      if (['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','svg'].includes(el.tagName)) continue;
      if (!vis(el)) continue;
      const tag = el.tagName.toLowerCase();
      const interactive = el.matches(INTERACTIVE);
      const own = [...el.childNodes].filter(n => n.nodeType === 3)
                    .map(n => n.textContent).join(' ').replace(/\s+/g, ' ').trim();
      let myRef = parentRef;
      if (interactive || (!inInteractive && own && own.length < 200)) {
        myRef = next++;
        el.setAttribute('data-agent-ref', String(myRef));
        const r = el.getBoundingClientRect();
        const type = (el.getAttribute('type') || '').toLowerCase();
        const editable = (tag === 'input' && (NONCLICK_INPUT.includes(type) || type === ''))
                         || tag === 'textarea' || el.isContentEditable;
        let text = interactive ? (el.innerText || el.value || '') : own;
        out.push({
          ref: myRef, depth, parent: parentRef,
          role: el.getAttribute('role') || (type ? `${tag}:${type}` : tag),
          text: String(text).replace(/\s+/g, ' ').trim().slice(0, 120),
          desc: el.getAttribute('aria-label') || el.getAttribute('placeholder')
                || el.getAttribute('title') || el.getAttribute('alt') || '',
          resource_id: el.id || el.getAttribute('name') || '',
          clickable: interactive && !editable,
          editable,
          scrollable: false,
          checked: (type === 'checkbox' || type === 'radio') ? !!el.checked : null,
          enabled: !el.disabled,
          bounds: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)],
        });
      }
      walk(el, depth + 1, myRef, inInteractive || interactive);
    }
  };
  if (document.body) walk(document.body, 0, -1, false);
  return { title: document.title, url: location.href, elements: out };
}
"""


def _host(url: str) -> str:
    try:
        return (up.urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def domain_allowed(url: str, allowed: tuple[str, ...]) -> bool:
    """Allow everything if no allowlist; else the host or any parent domain must match."""
    if not allowed:
        return True
    if url.startswith(("about:", "data:")):
        return True
    host = _host(url)
    return any(host == d or host.endswith("." + d) for d in allowed)


class BrowserSession:
    kind = "browser"

    def __init__(self, device_id: str, *, allowed_domains: tuple[str, ...] = (),
                 search_url: str = "https://duckduckgo.com/html/?q={q}",
                 headless: bool = True, viewport: tuple[int, int] = (1280, 900),
                 notify: Callable[[str, dict], Awaitable[None]] | None = None,
                 max_elements: int = 250):
        self.device_id = device_id
        self.allowed = tuple(d.strip().lower().lstrip(".") for d in allowed_domains if d.strip())
        self.search_url = search_url
        self.headless = headless
        self.viewport = viewport
        self.notify = notify
        self.max_elements = max_elements
        self._pw = self._browser = self._context = self.page = None
        self.blocked: list[str] = []         # navigations refused by the allowlist

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self, url: str = "about:blank") -> "BrowserSession":
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport[0], "height": self.viewport[1]},
            accept_downloads=False, locale="en-IN")
        await self._context.route("**/*", self._guard)
        self._context.on("page", self._on_new_page)
        self.page = await self._context.new_page()
        self._wire(self.page)
        if url and url != "about:blank":
            await self._goto(url)
        return self

    async def close(self) -> None:
        for obj in (self._context, self._browser):
            try:
                if obj is not None:
                    await obj.close()
            except Exception:
                pass
        try:
            if self._pw is not None:
                await self._pw.stop()
        except Exception:
            pass

    def _wire(self, page) -> None:
        # The context "page" event also fires for new_page(): wire each page only once,
        # or a dialog gets dismissed twice ("already handled").
        if getattr(page, "_agent_wired", False):
            return
        page._agent_wired = True
        # Never accept a JS dialog implicitly ("Confirm booking?" must not auto-confirm).
        page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))

    def _on_new_page(self, page) -> None:
        # Follow target=_blank / window.open into the new tab.
        self.page = page
        self._wire(page)

    async def _guard(self, route, request) -> None:
        if (request.is_navigation_request() and request.frame.parent_frame is None
                and not domain_allowed(request.url, self.allowed)):
            self.blocked.append(request.url)
            log.warning("browser: blocked navigation to %s (not in allowlist)", request.url)
            await route.abort("blockedbyclient")
            return
        await route.continue_()

    # ------------------------------------------------------------------ #
    # DeviceSession-compatible surface
    # ------------------------------------------------------------------ #
    async def request(self, type_: str, timeout: float = 60.0, **data: Any) -> dict:
        if type_ == "observe":
            return {"screen": await asyncio.wait_for(self._observe(), timeout)}
        if type_ == "act":
            return await asyncio.wait_for(self._act(data.get("action") or {}), timeout)
        if type_ == "handoff":
            url = self.page.url if self.page else ""
            return {"ok": True, "stop": True,
                    "message": f"This step needs you (payment/sign-in). Finish it in your "
                               f"browser at: {url}"}
        return {}

    async def push(self, type_: str, **data: Any) -> None:
        if self.notify is not None:
            try:
                await self.notify(type_, data)
            except Exception as e:
                log.warning("browser notify failed: %s", e)

    def on_message(self, msg: dict) -> None:     # registry compatibility (no socket)
        pass

    def fail_all(self, exc: Exception) -> None:
        pass

    # ------------------------------------------------------------------ #
    # observe / act
    # ------------------------------------------------------------------ #
    async def _settle(self) -> None:
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass
        await asyncio.sleep(0.4)     # let SPAs render after the DOM is ready

    async def _observe(self) -> dict:
        await self._settle()
        page = self.page
        try:
            dom = await page.evaluate(_SERIALIZE_JS, self.max_elements)
        except Exception as e:               # navigating mid-evaluate: retry once
            log.debug("serialize retry: %s", e)
            await asyncio.sleep(0.8)
            dom = await page.evaluate(_SERIALIZE_JS, self.max_elements)
        shot = await page.screenshot(type="jpeg", quality=55)
        return {
            "app_package": _host(dom.get("url", "")) or "browser",
            "activity": f"{dom.get('title', '')} | {dom.get('url', '')}"[:240],
            "elements": dom.get("elements", []),
            "screenshot_b64": base64.b64encode(shot).decode(),
            "is_keyboard_open": False,
        }

    def _to_url(self, text: str) -> str:
        t = (text or "").strip()
        if t.startswith(("http://", "https://", "about:")):
            return t
        if "." in t and " " not in t:
            return "https://" + t
        return self.search_url.format(q=up.quote_plus(t))

    async def _undo_blocked(self, n_before: int, prev_url: str) -> dict | None:
        """If a navigation was refused by the allowlist, Chromium shows its own error page
        (which go_back can't leave); return to the page we were on, and report why."""
        if len(self.blocked) <= n_before:
            return None
        target = self.blocked[-1]
        # The error page commits a moment AFTER the aborted request; wait for it.
        for _ in range(20):
            if self.page.url.startswith("chrome-error://"):
                break
            await asyncio.sleep(0.1)
        if self.page.url != prev_url and prev_url and not prev_url.startswith("chrome-error://"):
            try:
                await self.page.goto(prev_url, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                pass
        return {"ok": False, "error": f"navigation blocked by domain allowlist: {target}"}

    async def _goto(self, url: str) -> dict:
        n, prev = len(self.blocked), self.page.url
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            return await self._undo_blocked(n, prev) or {"ok": True}
        except Exception as e:
            return await self._undo_blocked(n, prev) or \
                {"ok": False, "error": f"navigation failed: {e}"}

    async def _act(self, a: dict) -> dict:
        page = self.page
        t = a.get("type")
        ref = a.get("ref")
        loc = page.locator(f'[data-agent-ref="{ref}"]').first if ref is not None else None
        vw, vh = self.viewport
        n_blocked, prev_url = len(self.blocked), page.url
        try:
            if t == "tap" and loc is not None:
                await loc.click(timeout=6000)
            elif t == "tap_xy" and a.get("x") is not None:
                await page.mouse.click(float(a["x"]) * vw, float(a["y"]) * vh)
            elif t == "type":
                if loc is not None:
                    await loc.fill(a.get("text") or "", timeout=6000)
                else:
                    await page.keyboard.type(a.get("text") or "")
            elif t == "enter":
                await page.keyboard.press("Enter")
            elif t in ("scroll", "swipe"):
                dy = vh * 0.8 * (-1 if a.get("direction") == "up" else 1)
                await page.mouse.wheel(0, dy)
            elif t == "back":
                await page.go_back(timeout=15000)
            elif t == "home":
                await page.goto("about:blank")
            elif t == "open_app":
                res = await self._goto(self._to_url(a.get("text") or ""))
                await self._settle()
                return res
            elif t == "wait":
                await asyncio.sleep(min(float(a.get("seconds") or 1.5), 10))
            else:
                return {"ok": False, "error": f"unsupported action for browser: {t}"}
        except Exception as e:
            return await self._undo_blocked(n_blocked, prev_url) or \
                {"ok": False, "error": f"{t} failed: {str(e)[:160]}"}
        await self._settle()
        # e.g. a tap on an off-site link: refused by the allowlist -> stay put.
        return await self._undo_blocked(n_blocked, prev_url) or {"ok": True}

    async def evidence(self) -> bytes | None:
        """Full-page screenshot of the final state (blueprint: screenshot evidence)."""
        try:
            return await self.page.screenshot(type="png", full_page=True)
        except Exception:
            return None
