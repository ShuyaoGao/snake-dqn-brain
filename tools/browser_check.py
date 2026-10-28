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
  - 观战: the brain panel is below the board, compact and inside the viewport width;
  - 我来玩: the whole page does not scroll (scrollHeight <= innerHeight); a touch drag on the joystick
    changes the snake's direction (up, then right); the big round key pauses / resumes;
  - 「看大脑」 opens the full-screen brain sheet and 「返回游戏」 closes it; ☰ menu opens / closes;
  - the help overlay shows the touch illustration (no keyboard list) on touch devices.
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
    ("watch_1920x1080.png", (1920, 1080), "shot=watch&steps=37"),
    ("play_1920x1080.png", (1920, 1080), "shot=play&steps=14"),
    ("knockout_1366x768.png", (1366, 768), "shot=knock&steps=25"),
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

        # 0. first visit: the help overlay is open; on touch it must show the touch illustration, not the keyboard list
        check(page.evaluate("!document.getElementById('help').hidden"), "首次打开没有弹出操作说明")
        if mobile:
            check(page.evaluate("document.body.classList.contains('touchp')"), "触屏竖屏没有加上 body.touchp")
            check(_rect(page, ".demo-stick")["vis"], "触屏说明没有显示摇杆图解")
            check(not _rect(page, ".keys-list")["vis"], "触屏说明仍显示键盘列表")
            hc = _rect(page, "#help-close")
            check(hc and hc["y"] + hc["h"] <= size["h"] + 1, "说明浮层的「开始」按钮不在视口内")
            page.tap("#help-close")
        else:
            check(not page.evaluate("document.body.classList.contains('touchp')"), "电脑尺寸误判为手机竖屏")
            page.click("#help-close")
        page.wait_for_timeout(200)
        check(page.evaluate("document.getElementById('help').hidden"), "操作说明关不掉")

        # 1. 观战 (default): brain panel visible; on phones it sits below the board and fits the width
        check(page.evaluate("window.__app && window.__app.mode() === 'watch'"), "默认模式不是观战")
        page.wait_for_timeout(700)
        check(page.evaluate("window.__app.watchSteps() > 0"), "观战模式没有前进")
        right, board, fc = _rect(page, "#right"), _rect(page, "#board"), _rect(page, "#fc-canvas")
        check(right and right["vis"] and fc and fc["vis"], "大脑面板不可见")
        if mobile:
            check(right["y"] >= board["y"] + board["h"] - 1, "手机观战：大脑面板不在棋盘下方")
            check(right["x"] >= -1 and right["x"] + right["w"] <= size["w"] + 1, "手机观战：大脑面板超出屏幕宽度")
            check(fc["x"] + fc["w"] <= size["w"] + 1, "手机观战：全连接层画布超出屏幕宽度")
            check(board["w"] >= min(size["w"] * 0.8, 600), "手机观战：棋盘没有按宽度缩放（%.0f / %d）" % (board["w"], size["w"]))
            inrow = _rect(page, "#row-input")
            check(inrow and inrow["h"] < 160, "手机观战：输入层 5 张图换行了（高 %.0f）" % (inrow["h"] if inrow else -1))
            check(page.evaluate("window.__app.kConv()") <= 6, "kConv")
            # ☰ menu opens and closes
            page.tap("#menu-btn"); page.wait_for_timeout(150)
            check(page.evaluate("!document.getElementById('menu').hidden"), "☰ 菜单打不开")
            page.tap("#menu-btn"); page.wait_for_timeout(150)
            check(page.evaluate("document.getElementById('menu').hidden"), "☰ 菜单关不掉")
            # knock out one conv channel by tapping the full-layer square, then restore from the menu
            page.evaluate("document.getElementById('sq-conv1').scrollIntoView({block: 'center'})"); page.wait_for_timeout(150)
            page.tap("#sq-conv1", position={"x": 6, "y": 6}); page.wait_for_timeout(200)
            check(page.evaluate("window.__app.knocked() === 1"), "手机观战：点方阵没有敲掉神经元")
            page.tap("#menu-btn"); page.wait_for_timeout(100); page.tap("#menu-restore"); page.wait_for_timeout(200)
            check(page.evaluate("window.__app.knocked() === 0"), "菜单「全部恢复」没有恢复")
            if name == MOBILE_SHOT_SIZE and shots_dir:
                page.evaluate("window.scrollTo(0, 0)"); page.wait_for_timeout(100)
                page.screenshot(path=str(Path(shots_dir) / "mobile_watch_390x844.png"))
                page.evaluate("document.getElementById('right').scrollIntoView({block: 'start'})"); page.wait_for_timeout(150)
                page.screenshot(path=str(Path(shots_dir) / "mobile_brain_390x844.png"))
                page.evaluate("window.scrollTo(0, 0)")

        # 2. 我来玩
        page.evaluate("window.scrollTo(0, 0)")
        (page.tap if mobile else page.click)("#tabs .tab[data-mode=play]"); page.wait_for_timeout(300)
        check(page.evaluate("window.__app.mode() === 'play'"), "切不到「我来玩」")
        if mobile:
            check(_no_scroll(page), "手机「我来玩」整页还能滚动（scrollHeight > innerHeight）")
            tc, bd = _rect(page, "#touchctl"), _rect(page, "#board")
            check(tc and tc["vis"] and tc["y"] + tc["h"] <= size["h"] + 1, "操作区不在视口内")
            check(bd and bd["y"] >= 0 and bd["y"] + bd["h"] <= (tc["y"] if tc else size["h"]) + 1, "棋盘不在视口内或与操作区重叠")
            check(bd["w"] >= 150, "棋盘太小")
            check(not _rect(page, "#right")["vis"], "手机「我来玩」大脑面板没有默认收起")
            big = _rect(page, "#tk-pause")
            check(big and big["w"] >= 64 and big["h"] >= 64, "大圆键小于 64px")
            # joystick: a touch drag up (initial direction is right = 3), then right; the snake must follow (turn-based: one step each)
            zone = _rect(page, "#stick-zone")
            cx, cy = zone["x"] + zone["w"] / 2, zone["y"] + zone["h"] * 0.6
            steps0 = page.evaluate("window.__app.playSteps()")
            _touch_drag(page, cx, cy, cx, cy - 70); page.wait_for_timeout(250)
            check(page.evaluate("window.__app.playDir() === 0"), "摇杆向上拖后蛇的方向不是上（%r）" % page.evaluate("window.__app.playDir()"))
            _touch_drag(page, cx, cy, cx + 70, cy); page.wait_for_timeout(250)
            check(page.evaluate("window.__app.playDir() === 3"), "摇杆向右拖后蛇的方向不是右（%r）" % page.evaluate("window.__app.playDir()"))
            check(page.evaluate("window.__app.playSteps()") == steps0 + 2, "回合制：两次拖摇杆应各走一步")
            check(page.evaluate("document.getElementById('touchctl').getAttribute('data-dir') === '-1'"), "松手后摇杆没有回中")
            tc2 = _rect(page, "#touchctl")
            check(tc2 and abs(tc2["y"] - tc["y"]) <= 1 and tc2["y"] + tc2["h"] <= size["h"] + 1, "走几步后操作区位置变了 / 被挤出视口（y %.0f → %.0f）" % (tc["y"], tc2["y"]))
            check(_no_scroll(page), "走几步后整页可以滚动了")
            if name == MOBILE_SHOT_SIZE and shots_dir:
                # README shot: joystick held to the right, game in progress
                cdp = page.context.new_cdp_session(page)
                cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": cx, "y": cy, "id": 1}]})
                cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": cx + 40, "y": cy, "id": 1}]})
                page.wait_for_timeout(150)
                page.screenshot(path=str(Path(shots_dir) / "mobile_play_390x844.png"))
                cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []}); cdp.detach()
                page.wait_for_timeout(100)
            # big key: in turn-based mode it is 「直走」 = one straight step (the README shot above may have added a step)
            s0 = page.evaluate("window.__app.playSteps()")
            page.tap("#tk-pause"); page.wait_for_timeout(200)
            check(page.evaluate("window.__app.playSteps()") == s0 + 1, "大圆键「直走」没有走一步")
            # timed level (new game): the big key starts, pauses and resumes
            page.tap("#mob-levels .preset[data-i='2']"); page.wait_for_timeout(150)
            page.tap("#b-mnext"); page.wait_for_timeout(200)
            lv = page.evaluate("(() => { const b = document.querySelector('#mob-levels .preset.on'); return {on: b ? b.dataset.i : null, html: document.getElementById('mob-levels').innerHTML.slice(0, 200), ms: window.__app.playMs()}; })()")
            check(lv["on"] == "2", "顶部一行的难度档位点不中: %r" % (lv,))
            check(page.evaluate("document.getElementById('tk-pause').textContent") == "开始", "限时档位新一局大圆键不是「开始」")
            steps0 = page.evaluate("window.__app.playSteps()")
            page.tap("#tk-pause"); page.wait_for_timeout(700)
            check(page.evaluate("document.getElementById('tk-pause').textContent") == "暂停", "开始后大圆键不是「暂停」")
            s1 = page.evaluate("window.__app.playSteps()")
            check(s1 > steps0, "限时档位开始后蛇没有自动前进")
            page.tap("#tk-pause"); page.wait_for_timeout(600)
            s2 = page.evaluate("window.__app.playSteps()")
            page.wait_for_timeout(600)
            check(page.evaluate("window.__app.playSteps()") == s2, "暂停后蛇还在走")
            check(page.evaluate("document.getElementById('tk-pause').textContent") == "继续", "暂停后大圆键不是「继续」")
            check(_no_scroll(page), "操作后整页可以滚动了")
            # 「看大脑」 sheet opens / closes
            page.tap("#btn-brain"); page.wait_for_timeout(300)
            check(page.evaluate("window.__app.sheet() && document.body.classList.contains('sheet-brain')"), "「看大脑」没有打开")
            r2 = _rect(page, "#right")
            check(r2 and r2["vis"] and r2["w"] >= size["w"] - 2, "「看大脑」面板不是全屏")
            check(_rect(page, "#brain-close")["vis"], "「返回游戏」按钮不可见")
            if name == MOBILE_SHOT_SIZE and shots_dir:
                page.screenshot(path=str(Path(shots_dir) / "mobile_sheet_390x844.png"))
            page.tap("#brain-close"); page.wait_for_timeout(250)
            check(not page.evaluate("window.__app.sheet()") and not _rect(page, "#right")["vis"], "「返回游戏」没有关闭大脑面板")
            check(_no_scroll(page), "关闭大脑面板后整页可以滚动")
        else:
            # desktop: keyboard play works and the right column is visible next to the board
            page.keyboard.press("ArrowUp"); page.wait_for_timeout(150)
            check(page.evaluate("window.__app.playDir() === 0"), "电脑：方向键没有改变方向")
            right, board = _rect(page, "#right"), _rect(page, "#board")
            check(right["x"] >= board["x"] + board["w"], "电脑：大脑面板不在右栏")
            check(not _rect(page, "#touchctl")["vis"], "电脑上显示了触屏操作区")

        # 3. 练习 then back to 观战: no overflow (generic check runs after the scenario)
        (page.tap if mobile else page.click)("#tabs .tab[data-mode=practice]"); page.wait_for_timeout(250)
        check(page.evaluate("window.__app.mode() === 'practice'"), "切不到「练习」")
        (page.tap if mobile else page.click)("#tabs .tab[data-mode=watch]"); page.wait_for_timeout(250)
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
