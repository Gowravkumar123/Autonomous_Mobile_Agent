"""
Windows adapter: UI Automation tree -> the platform's UiElement shape, and actions back.

Design choices:
- Actions go through UIA *patterns* (Invoke / Value / Toggle / SelectionItem / Scroll /
  ExpandCollapse) by default. They work on minimized or background windows and never move
  the user's real mouse or keyboard, so the agent can work while the user does too.
  `input_mode="mouse"` additionally allows real clicks/keystrokes (focuses the window).
- Enter is delivered as posted key messages to the control's own window handle — again no
  focus change.
- Screenshots use PrintWindow on the TARGET window only: other windows on top of it (or
  the rest of the desktop) are never captured.
- All COM calls run on ONE dedicated thread (UIA objects are apartment-bound).
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as wt
import io
import logging
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .safety import Safety

log = logging.getLogger("onepiece.desktop")

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WM_KEYDOWN, WM_KEYUP, WM_CHAR, VK_RETURN = 0x0100, 0x0101, 0x0102, 0x0D

# 64-bit handles: without explicit types ctypes truncates them to C int.
_H = ctypes.c_void_p
for _fn, _res, _args in (
        (user32.GetWindowDC, _H, [_H]), (user32.ReleaseDC, ctypes.c_int, [_H, _H]),
        (user32.PrintWindow, wt.BOOL, [_H, _H, wt.UINT]),
        (user32.PostMessageW, wt.BOOL, [_H, wt.UINT, wt.WPARAM, wt.LPARAM]),
        (gdi32.CreateCompatibleDC, _H, [_H]),
        (gdi32.CreateCompatibleBitmap, _H, [_H, ctypes.c_int, ctypes.c_int]),
        (gdi32.SelectObject, _H, [_H, _H]), (gdi32.DeleteObject, wt.BOOL, [_H]),
        (gdi32.DeleteDC, wt.BOOL, [_H]),
        (gdi32.GetDIBits, ctypes.c_int, [_H, _H, wt.UINT, wt.UINT, ctypes.c_void_p,
                                         ctypes.c_void_p, wt.UINT]),
        (kernel32.OpenProcess, _H, [wt.DWORD, wt.BOOL, wt.DWORD]),
        (kernel32.CloseHandle, wt.BOOL, [_H]),
        (kernel32.QueryFullProcessImageNameW, wt.BOOL, [_H, wt.DWORD, wt.LPWSTR,
                                                        ctypes.POINTER(wt.DWORD)])):
    _fn.restype, _fn.argtypes = _res, _args

CLICKABLE = {"Button", "CheckBox", "RadioButton", "Hyperlink", "MenuItem", "ListItem",
             "TabItem", "ComboBox", "TreeItem", "SplitButton", "DataItem", "MenuBarItem"}
EDITABLE = {"Edit", "Document"}
SKIP_SUBTREE = {"TitleBar", "ScrollBar", "Thumb"}
CONTAINERS = {"Pane", "Group", "Custom", "Window", "ToolBar", "MenuBar", "StatusBar", "List",
              "Tree", "Table", "DataGrid", "Tab", "Header", "Menu"}


def exe_of_pid(pid: int) -> str:
    h = kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wt.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value.replace("/", "\\").rsplit("\\", 1)[-1].lower()
        return ""
    finally:
        kernel32.CloseHandle(h)


def capture_window(hwnd: int, max_width: int = 1024, quality: int = 55) -> str | None:
    """JPEG (base64) of ONE window's own content via PrintWindow; None if minimized/hidden."""
    if not hwnd or user32.IsIconic(hwnd) or not user32.IsWindowVisible(hwnd):
        return None
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    if w <= 0 or h <= 0 or w * h > 16_000_000:
        return None
    from PIL import Image
    hdc = user32.GetWindowDC(hwnd)
    mdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mdc, bmp)
    try:
        if not user32.PrintWindow(hwnd, mdc, 2):         # PW_RENDERFULLCONTENT
            return None

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                        ("biPlanes", wt.WORD), ("biBitCount", wt.WORD),
                        ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                        ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                        ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]
        bi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(w * h * 4)
        if not gdi32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0):
            return None
        img = Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)
    finally:
        gdi32.SelectObject(mdc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(hwnd, hdc)
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=quality)
    return base64.b64encode(out.getvalue()).decode()


class WindowsDesktop:
    """observe()/act() over one target top-level window, behind the local Safety checks."""

    def __init__(self, safety: Safety, *, title_re: str = "", input_mode: str = "patterns",
                 apps: dict[str, str] | None = None, max_elements: int = 150,
                 max_depth: int = 14, screenshots: bool = True):
        self.safety = safety
        self.title_re = re.compile(title_re, re.I) if title_re else None
        self.input_mode = input_mode
        self.apps = {k.lower(): v for k, v in (apps or {}).items()}
        self.max_elements = max_elements
        self.max_depth = max_depth
        self.screenshots = screenshots
        self.target_hwnd = 0
        self._refs: dict[int, Any] = {}
        self._meta: dict[int, dict] = {}
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="uia", initializer=self._init_com)

    @staticmethod
    def _init_com() -> None:
        import uiautomation as auto
        auto.InitializeUIAutomationInCurrentThread()

    # ---- thread hop --------------------------------------------------------
    async def run(self, fn, *args):
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- window targeting --------------------------------------------------
    def _windows(self) -> list:
        import uiautomation as auto
        return [w for w in auto.GetRootControl().GetChildren()
                if w.NativeWindowHandle and user32.IsWindowVisible(w.NativeWindowHandle)]

    def _target(self):
        import uiautomation as auto
        if self.target_hwnd and user32.IsWindow(self.target_hwnd):
            return auto.ControlFromHandle(self.target_hwnd)
        if self.title_re is not None:
            for w in self._windows():
                if self.title_re.search(w.Name or ""):
                    self.target_hwnd = w.NativeWindowHandle
                    return w
            return None
        fg = auto.GetForegroundControl()
        return fg.GetTopLevelControl() if fg else None

    # ---- observe -----------------------------------------------------------
    def observe(self) -> dict:
        win = self._target()
        self._refs, self._meta = {}, {}
        if win is None:
            return {"app_package": "desktop", "activity": "no target window",
                    "elements": [], "screenshot_b64": None}
        exe = exe_of_pid(win.ProcessId)
        title = win.Name or ""
        if not self.safety.process_allowed(exe):
            # Privacy: report only THAT it's blocked — no tree, no screenshot.
            return {"app_package": exe or "unknown", "activity": "BLOCKED",
                    "elements": [{"ref": 0, "role": "Text",
                                  "text": f"This window ({exe}) is not on the desktop "
                                          f"agent's allowlist. Switch to an allowed app "
                                          f"with open_app, or stop."}],
                    "screenshot_b64": None}
        elements: list[dict] = []
        self._walk(win, 0, -1, elements)
        shot = capture_window(win.NativeWindowHandle) if self.screenshots else None
        return {"app_package": exe, "activity": title[:240], "elements": elements,
                "screenshot_b64": shot, "is_keyboard_open": False}

    def _walk(self, ctl, depth: int, parent: int, out: list[dict]) -> None:
        if depth > self.max_depth:
            return
        child = ctl.GetFirstChildControl()
        while child is not None and len(out) < self.max_elements:
            try:
                role = child.ControlTypeName.removesuffix("Control")
                if role in SKIP_SUBTREE:
                    child = child.GetNextSiblingControl()
                    continue
                my = parent
                el = self._element(child, role, depth, parent, len(out))
                if el is not None:
                    my = el["ref"]
                    out.append(el)
                self._walk(child, depth + 1, my, out)
            except Exception as e:                # control vanished mid-walk
                log.debug("walk skip: %s", e)
            child = child.GetNextSiblingControl()

    def _element(self, c, role: str, depth: int, parent: int, ref: int) -> dict | None:
        name = (c.Name or "").strip()
        aid = c.AutomationId or ""
        editable = False
        value = ""
        password = False
        if role in EDITABLE:
            try:
                password = bool(c.IsPassword)
                vp = _pattern(c, "ValuePattern")
                editable = not vp.IsReadOnly
                value = "" if password else (vp.Value or "")
            except Exception:
                editable = False
        clickable = role in CLICKABLE
        checked = None
        if role in ("CheckBox", "RadioButton") or (role == "Button" and _has(c, "TogglePattern")):
            try:
                checked = _pattern(c, "TogglePattern").ToggleState == 1
            except Exception:
                try:
                    checked = bool(_pattern(c, "SelectionItemPattern").IsSelected)
                except Exception:
                    checked = None
        scrollable = _has(c, "ScrollPattern")
        keep = clickable or editable or scrollable or (name and role not in CONTAINERS)
        if not keep:
            return None
        r = c.BoundingRectangle
        self._refs[ref] = c
        self._meta[ref] = {"role": role, "password": password,
                           "hwnd": c.NativeWindowHandle or 0, "label": name or aid}
        return {
            "ref": ref, "depth": depth, "parent": parent, "role": role,
            "text": (value if role in EDITABLE else name)[:120],
            "desc": ("password field" if password else name if role in EDITABLE else "")[:120],
            "resource_id": aid, "clickable": clickable and not editable,
            "editable": editable and not password, "scrollable": scrollable,
            "checked": checked, "enabled": bool(c.IsEnabled),
            "bounds": [r.left, r.top, r.right, r.bottom],
        }

    # ---- act ---------------------------------------------------------------
    def act(self, a: dict) -> dict:
        t = a.get("type")
        ref = a.get("ref")
        if t == "wait":
            time.sleep(min(float(a.get("seconds") or 1.0), 10))
            return {"ok": True}
        if t == "open_app":
            return self._open_app(a.get("text") or "")
        win = self._target()
        if win is None:
            return {"ok": False, "error": "no target window"}
        exe = exe_of_pid(win.ProcessId)
        if not self.safety.process_allowed(exe):
            return {"ok": False, "error": f"refused: {exe} is not on the allowlist"}
        c = self._refs.get(ref) if ref is not None else None
        meta = self._meta.get(ref, {}) if ref is not None else {}
        if ref is not None and c is None:
            return {"ok": False, "error": f"unknown ref {ref} — observe again"}
        try:
            if t == "tap" and c is not None:
                return self._tap(c, meta)
            if t == "type":
                text = a.get("text") or ""
                if meta.get("password"):
                    return {"ok": False, "error": "refused: the user must type secrets themselves"}
                if not self.safety.path_allowed(text):
                    return {"ok": False, "error": "refused: path is outside the workspace "
                                                  f"({', '.join(map(str, self.safety.workspaces))})"}
                target = c if c is not None else self._first_editable()
                if target is None:
                    return {"ok": False, "error": "no editable field to type into"}
                _pattern(target, "ValuePattern").SetValue(text)
                self._last_edit = target
                return self._settled()
            if t == "enter":
                target = c if c is not None else getattr(self, "_last_edit", None)
                hwnd = target.NativeWindowHandle if target is not None else 0
                if hwnd:
                    user32.PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, 0x001C0001)
                    user32.PostMessageW(hwnd, WM_CHAR, VK_RETURN, 0x001C0001)
                    user32.PostMessageW(hwnd, WM_KEYUP, VK_RETURN, 0xC01C0001)
                    return self._settled()
                if self.input_mode == "mouse" and target is not None:
                    target.SetFocus()
                    target.SendKeys("{Enter}")
                    return self._settled()
                return {"ok": False, "error": "enter needs a native field; tap the submit "
                                              "button instead"}
            if t in ("scroll", "swipe"):
                return self._scroll(c, a.get("direction") or "down")
            if t == "tap_xy" and a.get("x") is not None:
                if self.input_mode != "mouse":
                    return {"ok": False, "error": "tap_xy needs input_mode=mouse; tap by ref"}
                r = win.BoundingRectangle
                win.SetActive()
                import uiautomation as auto
                auto.Click(int(r.left + float(a["x"]) * r.width()),
                           int(r.top + float(a["y"]) * r.height()))
                return self._settled()
            return {"ok": False, "error": f"unsupported on desktop: {t} (tap controls by ref)"}
        except Exception as e:
            return {"ok": False, "error": f"{t} failed: {str(e)[:160]}"}

    def _settled(self) -> dict:
        time.sleep(0.3)
        return {"ok": True}

    def _tap(self, c, meta: dict) -> dict:
        for pattern, call in (("InvokePattern", lambda p: p.Invoke()),
                              ("TogglePattern", lambda p: p.Toggle()),
                              ("SelectionItemPattern", lambda p: p.Select()),
                              ("ExpandCollapsePattern", lambda p: p.Expand())):
            p = _pattern(c, pattern)
            if p is not None:
                call(p)
                return self._settled()
        legacy = _pattern(c, "LegacyIAccessiblePattern")
        if legacy is not None and legacy.DefaultAction:
            legacy.DoDefaultAction()
            return self._settled()
        if self.input_mode == "mouse":
            c.Click(simulateMove=False)
            return self._settled()
        return {"ok": False, "error": f"'{meta.get('label', '?')}' has no automation pattern "
                                      "(set input_mode=mouse to allow real clicks)"}

    def _scroll(self, c, direction: str) -> dict:
        cands = [c] if c is not None else [x for x in self._refs.values() if _has(x, "ScrollPattern")]
        for x in cands:
            p = _pattern(x, "ScrollPattern")
            if p is None:
                continue
            import uiautomation as auto
            big = auto.ScrollAmount.LargeIncrement
            small = auto.ScrollAmount.NoAmount
            if direction in ("up", "down"):
                p.Scroll(small, big if direction == "down" else auto.ScrollAmount.LargeDecrement)
            else:
                p.Scroll(big if direction == "right" else auto.ScrollAmount.LargeDecrement, small)
            return self._settled()
        return {"ok": False, "error": "nothing scrollable here"}

    def _first_editable(self):
        for ref, m in self._meta.items():
            if m["role"] in EDITABLE and not m["password"]:
                return self._refs[ref]
        return None

    def _open_app(self, name: str) -> dict:
        """Retarget to an open allowlisted window whose title matches, else launch a
        configured app. Never changes focus."""
        q = name.strip().lower()
        if not q:
            return {"ok": False, "error": "open_app needs an app or window name"}
        for w in self._windows():
            if q in (w.Name or "").lower() and self.safety.process_allowed(exe_of_pid(w.ProcessId)):
                self.target_hwnd = w.NativeWindowHandle
                return {"ok": True}
        cmd = self.apps.get(q)
        if cmd is None:
            return {"ok": False, "error": f"'{name}' is not open and not a configured app "
                                          f"(configured: {', '.join(sorted(self.apps)) or 'none'})"}
        before = {w.NativeWindowHandle for w in self._windows()}
        subprocess.Popen(cmd, shell=False)
        for _ in range(40):
            time.sleep(0.25)
            for w in self._windows():
                if w.NativeWindowHandle not in before and \
                        self.safety.process_allowed(exe_of_pid(w.ProcessId)):
                    self.target_hwnd = w.NativeWindowHandle
                    return {"ok": True}
        return {"ok": False, "error": f"launched {name} but no allowlisted window appeared"}


def _pattern(c, name: str):
    """Generic pattern lookup (the Get*Pattern helpers exist only on typed subclasses)."""
    import uiautomation as auto
    try:
        return c.GetPattern(getattr(auto.PatternId, name))
    except Exception:
        return None


def _has(c, name: str) -> bool:
    return _pattern(c, name) is not None
