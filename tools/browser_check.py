# -*- coding: utf-8 -*-
"""Browser acceptance of play/index.html with the locally installed Edge / Chrome.

    python -m tools.browser_check                     # desktop self-test (1920x1080, 1366x768) + exit button
    python -m tools.browser_check --mobile            # ... plus phone / tablet / desktop checks via Playwright + Edge
    python -m tools.browser_check --mobile --shots docs/img   # also write the README screenshots

Desktop part: headless Edge/Chrome runs the page's built-in self-test (play/js/app.js `selftest()`) at two
window sizes and asserts no JS errors and no horizontal overflow (read back through --dump-dom).

Mobile part (--mobile; needs `pip install playwright`, uses channel="msedge", no browser download):
tools/mobile_check.py emulates 390x844, 414x896, 360x780 (DPR 3), 768x1024 (DPR 2) with touch, plus
desktop 1920x1080 and 1366x768, and this module adds the scenario assertions of the v2 spec:
  - no horizontal overflow (generic) and no JS errors (generic);
  - the brain panel is below the board, compact and inside the viewport width (phones);
  - tapping the full-layer square knocks out a channel and 「全部恢复」 restores it; ☰ menu opens / closes;
  - every brain can be switched to (board 12x12 / 10x10 follows) and keeps advancing;
  - brain B's guard switch and the policy buttons work; the help overlay shows the touch list on touch devices.
Only one Edge runs on the machine at a time (file lock in tools/mobile_check.py) and it waits for
>= 1.5 GB free memory before starting. Only the processes started here are terminated.
"""

import argparse
import ctypes
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INDEX = ROOT / "play" / "index.html"
SIZES = [(1920, 1080), (1366, 768)]
CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser():
    env = os.environ.get("BROWSER_EXE")
    if env and Path(env).exists():
        return env
    for c in CANDIDATES:
        if Path(c).exists():
            return c
    for name in ("msedge", "microsoft-edge", "google-chrome", "chromium", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    return None


def free_mb():
    if sys.platform != "win32":
        return None

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys // (1024 * 1024)


def wait_for_memory(min_mb, max_wait_s):
    t0 = time.time()
    while True:
        f = free_mb()
        if f is None or f >= min_mb:
            return f
        if time.time() - t0 > max_wait_s:
            return f
        print(f"  free memory {f} MB < {min_mb} MB, waiting...", flush=True)
        time.sleep(15)


def edge_lock():
    """The machine-wide 'one Edge at a time' lock from tools/mobile_check.py (a no-op context if absent)."""
    try:
        from tools.mobile_check import EdgeLock
        return EdgeLock()
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def run_browser(exe, url, size, extra, timeout=120):
    prof = tempfile.mkdtemp(prefix="snake_edge_")
    args = [exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--disable-extensions", "--disable-sync", "--mute-audio", "--hide-scrollbars",
            f"--user-data-dir={prof}", f"--window-size={size[0]},{size[1]}"] + extra + [url]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            p.kill()
        out, _ = p.communicate()
        raise RuntimeError(f"browser timed out after {timeout}s")
    finally:
        time.sleep(0.5)
        shutil.rmtree(prof, ignore_errors=True)
    return out.decode("utf-8", "replace")


def selftest(exe, size):
    url = INDEX.as_uri() + "?selftest=1"
    dom = run_browser(exe, url, size, ["--virtual-time-budget=30000", "--dump-dom"])
    m = re.search(r'<pre id="selftest"[^>]*>(.*?)</pre>', dom, re.S)
    if not m or not m.group(1).strip():
        return {"ok": False, "errors": ["self-test result not found in DOM"], "title": re.findall(r"<title>(.*?)</title>", dom)}
    return json.loads(html.unescape(m.group(1)))


def exit_check(exe):
    """Click 「退出」: the page must stop and show the end message (window.close() is refused for
    a page the user opened, so the fallback page is what users see)."""
    dom = run_browser(exe, INDEX.as_uri() + "?exittest=1", (1366, 768), ["--virtual-time-budget=3000", "--dump-dom"])
    return "已停止运行，所有计算都已结束，可以直接关闭这个标签页" in dom


def screenshot(exe, size, query, path):
    url = INDEX.as_uri() + "?" + query
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    run_browser(exe, url, size, ["--virtual-time-budget=4000", f"--screenshot={path}"])
    return path.exists() and path.stat().st_size > 0


SHOTS = [
    ("watch_1920x1080.png", (1920, 1080), "shot=watch&brain=ppo_linyi&steps=260"),
    ("brain_b_1920x1080.png", (1920, 1080), "shot=watch&brain=ppo_ray&steps=180"),
    ("dqn_1920x1080.png", (1920, 1080), "shot=watch&brain=dqn_own&steps=37"),
    ("knockout_1366x768.png", (1366, 768), "shot=knock&brain=ppo_linyi&steps=120"),
]


# ---------------------------------------------------------------------------- Playwright-driven Edge (preferred)
# The msedge.exe command line hands the URL to an already running Edge (returns at once with no DOM) when the
# user has Edge open, so the desktop checks also go through Playwright (channel "msedge", headless) when it is
# installed; the plain command-line functions above remain the fallback.
def have_playwright():
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


class PwEdge:
    """`with PwEdge() as pw:` -> pw.selftest(size) / pw.exit_check() / pw.screenshot(size, query, path)."""

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._p = sync_playwright().start()
        self.browser = self._p.chromium.launch(channel="msedge", headless=True)
        return self

    def __exit__(self, *a):
        try:
            self.browser.close()
        finally:
            self._p.stop()

    def _page(self, size, query):
        ctx = self.browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1, locale="zh-CN")
        page = ctx.new_page()
        page.goto(INDEX.as_uri() + "?" + query)
        return ctx, page

    def selftest(self, size):
        ctx, page = self._page(size, "selftest=1")
        try:
            page.wait_for_function("document.title.indexOf('SELFTEST') === 0", timeout=60000)
            txt = page.evaluate("document.getElementById('selftest').textContent")
            return json.loads(txt) if txt.strip() else {"ok": False, "errors": ["self-test result not found in DOM"]}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "errors": ["selftest did not finish: %r" % (e,)]}
        finally:
            ctx.close()

    def exit_check(self):
        ctx, page = self._page((1366, 768), "exittest=1")
        try:
            page.wait_for_timeout(1500)
            return "已停止运行，所有计算都已结束，可以直接关闭这个标签页" in page.evaluate("document.body.innerText")
        finally:
            ctx.close()

    def screenshot(self, size, query, path):
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        ctx, page = self._page(size, query)
        try:
            page.wait_for_timeout(1200)
            page.screenshot(path=str(path), full_page=False)
        finally:
            ctx.close()
        return path.exists() and path.stat().st_size > 0

# ---------------------------------------------------------------------------- mobile scenarios (Playwright)
MOBILE_SHOT_SIZE = "m390x844"   # the phone size whose screenshots go into README


def _touch_drag(page, x0, y0, x1, y1, steps=6):
    """A real touch sequence (touchStart / touchMove... / touchEnd) through the DevTools protocol."""
    cdp = page.context.new_cdp_session(page)
    try:
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x0, "y": y0, "id": 1}]})
        for i in range(1, steps + 1):
            t = i / steps
            cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x0 + (x1 - x0) * t, "y": y0 + (y1 - y0) * t, "id": 1}]})
            page.wait_for_timeout(30)
        page.wait_for_timeout(120)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    finally:
        cdp.detach()


def _rect(page, sel):
    return page.evaluate("s => { const e = document.querySelector(s); if (!e) return null; const r = e.getBoundingClientRect();"
                         " return {x: r.x, y: r.y, w: r.width, h: r.height, vis: !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length)}; }", sel)


def _no_scroll(page):
    return page.evaluate("Math.max(document.documentElement.scrollHeight, document.body.scrollHeight) <= window.innerHeight + 1")


def mobile_scenario(shots_dir):
    """Returns the scenario(page, size) callback for tools.mobile_check.run; `shots_dir` gets the README phone shots."""
    def scenario(page, size):
        fails = []
        mobile = size["kind"] == "mobile"
        name = size["name"]

        def check(cond, msg):
            if not cond:
                fails.append(msg)

        tap = page.tap if mobile else page.click
        # 0. first visit: the help overlay is open; on touch it must show the touch list, not the keyboard list
        check(page.evaluate("!document.getElementById('help').hidden"), "首次打开没有弹出操作说明")
        if mobile:
            check(page.evaluate("document.body.classList.contains('touchp')"), "触屏竖屏没有加上 body.touchp")
            check(_rect(page, ".touch-list")["vis"], "触屏说明没有显示触屏操作列表")
            check(not _rect(page, ".keys-list")["vis"], "触屏说明仍显示键盘列表")
            hc = _rect(page, "#help-close")
            check(hc and hc["y"] + hc["h"] <= size["h"] + 1, "说明浮层的「开始」按钮不在视口内")
        else:
            check(not page.evaluate("document.body.classList.contains('touchp')"), "电脑尺寸误判为手机竖屏")
        tap("#help-close")
        page.wait_for_timeout(200)
        check(page.evaluate("document.getElementById('help').hidden"), "操作说明关不掉")

        # 1. default brain plays; the brain panel is visible (phones: below the board, inside the viewport width)
        check(page.evaluate("window.__app && window.__app.brain() === 'ppo_linyi'"), "默认大脑不是 ppo_linyi")
        page.wait_for_timeout(900)
        check(page.evaluate("window.__app.watchSteps() > 0"), "观看模式没有前进")
        right, board, fc = _rect(page, "#right"), _rect(page, "#board"), _rect(page, "#fc-canvas")
        check(right and right["vis"] and fc and fc["vis"], "大脑面板不可见")
        if mobile:
            check(right["y"] >= board["y"] + board["h"] - 1, "手机：大脑面板不在棋盘下方")
            check(right["x"] >= -1 and right["x"] + right["w"] <= size["w"] + 1, "手机：大脑面板超出屏幕宽度")
            check(fc["x"] + fc["w"] <= size["w"] + 1, "手机：全连接层画布超出屏幕宽度")
            check(board["w"] >= min(size["w"] * 0.8, 600), "手机：棋盘没有按宽度缩放（%.0f / %d）" % (board["w"], size["w"]))
            inrow = _rect(page, "#row-input")
            check(inrow and inrow["h"] < 160, "手机：输入层的图换行了（高 %.0f）" % (inrow["h"] if inrow else -1))
            check(page.evaluate("window.__app.kConv()") <= 6, "kConv")
            # ☰ menu opens and closes
            page.tap("#menu-btn"); page.wait_for_timeout(150)
            check(page.evaluate("!document.getElementById('menu').hidden"), "☰ 菜单打不开")
            page.tap("#menu-btn"); page.wait_for_timeout(150)
            check(page.evaluate("document.getElementById('menu').hidden"), "☰ 菜单关不掉")
        else:
            check(right["x"] >= board["x"] + board["w"], "电脑：大脑面板不在右栏")

        # 2. knock out one conv channel by tapping the full-layer square, then restore
        page.evaluate("document.getElementById('sq-conv1').scrollIntoView({block: 'center'})"); page.wait_for_timeout(150)
        tap("#sq-conv1", position={"x": 6, "y": 6}); page.wait_for_timeout(250)
        check(page.evaluate("window.__app.knocked() === 1"), "点方阵没有敲掉神经元")
        if mobile:
            page.tap("#menu-btn"); page.wait_for_timeout(100); page.tap("#menu-restore"); page.wait_for_timeout(250)
        else:
            page.click("#btn-restore"); page.wait_for_timeout(250)
        check(page.evaluate("window.__app.knocked() === 0"), "「全部恢复」没有恢复")
        if mobile and name == MOBILE_SHOT_SIZE and shots_dir:
            page.evaluate("window.scrollTo(0, 0); document.getElementById('toast').classList.remove('show')"); page.wait_for_timeout(400)
            page.screenshot(path=str(Path(shots_dir) / "mobile_watch_390x844.png"))
            page.evaluate("document.getElementById('right').scrollIntoView({block: 'start'})"); page.wait_for_timeout(150)
            page.screenshot(path=str(Path(shots_dir) / "mobile_brain_390x844.png"))

        # 3. switch brains: board size and panel follow; every brain advances
        page.evaluate("window.scrollTo(0, 0)")
        for bid, n in (("ppo_ray", 12), ("ppo_mlp_linyi", 12), ("dqn_own", 10), ("ppo_linyi", 12)):
            tap("#opts .preset[data-kind=brain][data-id=%s]" % bid); page.wait_for_timeout(700)
            check(page.evaluate("window.__app.brain()") == bid, "切不到大脑 %s" % bid)
            check(page.evaluate("window.__app.n()") == n, "大脑 %s 的棋盘不是 %d×%d" % (bid, n, n))
            check(page.evaluate("window.__app.watchSteps() > 0"), "大脑 %s 没有前进" % bid)
            check(_rect(page, "#fc-canvas")["vis"], "大脑 %s 的面板没有画出来" % bid)
            if mobile:
                r2 = _rect(page, "#right")
                check(r2["x"] + r2["w"] <= size["w"] + 1, "手机：大脑 %s 的面板超出屏幕宽度" % bid)
            if mobile and bid == "ppo_ray" and name == MOBILE_SHOT_SIZE and shots_dir:
                page.evaluate("window.scrollTo(0, 0); document.getElementById('toast').classList.remove('show')"); page.wait_for_timeout(400)
                page.screenshot(path=str(Path(shots_dir) / "mobile_brain_b_390x844.png"))
        # 4. the guard switch of brain B and the policy buttons
        tap("#opts .preset[data-kind=brain][data-id=ppo_ray]"); page.wait_for_timeout(300)
        check(page.evaluate("window.__app.guard() === true"), "大脑 B 默认没有开护栏")
        tap("#ck-guard"); page.wait_for_timeout(200)
        check(page.evaluate("window.__app.guard() === false"), "护栏开关没有关掉")
        tap("#opts .preset[data-kind=policy][data-v=sample]"); page.wait_for_timeout(200)
        check(page.evaluate("window.__app.policy() === 'sample'"), "选动作没有切到抽样")
        tap("#opts .preset[data-kind=brain][data-id=ppo_linyi]"); page.wait_for_timeout(300)
        page.evaluate("window.scrollTo(0, 0)")
        return fails
    return scenario


def mobile_check(shots_dir, out_dir):
    from tools import mobile_check as mc
    report = mc.run(str(INDEX), out_dir, scenario=mobile_scenario(shots_dir))
    for s in report["sizes"]:
        print("  %-11s %4dx%-4d %s" % (s["name"], s["w"], s["h"], "PASS" if not s["fails"] else "FAIL: " + " | ".join(s["fails"])), flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", help="directory for screenshots (README images)")
    ap.add_argument("--mobile", action="store_true", help="also run the phone / tablet / desktop checks with Playwright + Edge")
    ap.add_argument("--mobile-out", help="directory for the mobile check report and per-size screenshots (default: temp)")
    ap.add_argument("--min-free-mb", type=int, default=1500)
    ap.add_argument("--max-wait", type=int, default=600, help="seconds to wait for free memory")
    args = ap.parse_args(argv)
    exe = find_browser()
    if not exe:
        print("no Edge / Chrome found: browser check NOT run")
        return 2
    print("browser:", exe)
    results = {}
    ok = True
    shots = None
    if args.shots:
        shots = Path(args.shots)
        if not shots.is_absolute():
            shots = ROOT / shots
        shots.mkdir(parents=True, exist_ok=True)
    pw = have_playwright()
    print("desktop checks via:", "Playwright + Edge (headless)" if pw else "msedge.exe command line (install playwright if Edge is already open)")
    with edge_lock():
        wait_for_memory(args.min_free_mb, args.max_wait)
        drv = PwEdge().__enter__() if pw else None
        try:
            for size in SIZES:
                f = wait_for_memory(args.min_free_mb, args.max_wait)
                print(f"[{size[0]}x{size[1]}] free memory: {f} MB", flush=True)
                r = drv.selftest(size) if drv else selftest(exe, size)
                results[f"{size[0]}x{size[1]}"] = r
                good = bool(r.get("ok")) and not r.get("errors")
                ok = ok and good
                print(f"  {'PASS' if good else 'FAIL'}: {json.dumps(r, ensure_ascii=False)[:600]}", flush=True)
            ex = drv.exit_check() if drv else exit_check(exe)
            print(f"  exit button: {'PASS' if ex else 'FAIL'}", flush=True)
            ok = ok and ex
            if shots:
                for name, size, q in SHOTS:
                    wait_for_memory(args.min_free_mb, args.max_wait)
                    done = drv.screenshot(size, q + "&nohelp=1", shots / name) if drv else screenshot(exe, size, q + "&nohelp=1", shots / name)
                    print(f"  screenshot {name}: {'ok' if done else 'FAILED'}", flush=True)
                    ok = ok and done
        finally:
            if drv:
                drv.__exit__(None, None, None)
    if args.mobile:
        out = args.mobile_out or os.path.join(tempfile.gettempdir(), "snake_mobile_check")
        print("[mobile] Playwright + Edge, DevTools device emulation ->", out, flush=True)
        rep = mobile_check(str(shots) if shots else None, out)
        ok = ok and rep["ok"]
        print(f"  mobile: {'PASS' if rep['ok'] else 'FAIL'} ({sum(1 for s in rep['sizes'] if not s['fails'])}/{len(rep['sizes'])} sizes)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
