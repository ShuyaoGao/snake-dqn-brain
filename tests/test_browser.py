# -*- coding: utf-8 -*-
"""Browser acceptance of play/index.html with the locally installed Edge / Chrome.

Opt-in (it starts a real browser): SNAKE_BROWSER_TEST=1 python -m pytest tests/test_browser.py
Same as: python -m tools.browser_check --mobile

Desktop: headless self-test at 1920x1080 and 1366x768 + the exit button.
Mobile: Playwright (channel msedge) emulates 390x844 / 414x896 / 360x780 / 768x1024 with touch and the two
desktop sizes, and runs the scenario assertions in tools/browser_check.py (joystick drag changes the snake's
direction, no page scroll in 我来玩, brain sheet opens / closes, no overflow, no JS errors).
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import browser_check  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("SNAKE_BROWSER_TEST") != "1",
                                reason="set SNAKE_BROWSER_TEST=1 to run the headless browser test")


@pytest.mark.parametrize("size", browser_check.SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_selftest_no_js_errors(size):
    exe = browser_check.find_browser()
    if not exe:
        pytest.skip("no Edge / Chrome installed")
    with browser_check.edge_lock():
        browser_check.wait_for_memory(1500, 600)
        if browser_check.have_playwright():
            with browser_check.PwEdge() as drv:
                r = drv.selftest(size)
        else:
            r = browser_check.selftest(exe, size)
    assert r.get("errors") == [], r
    assert r.get("ok") is True, r
    assert r["viewport"] == [size[0], size[1]] or r["viewport"][0] <= size[0]


def test_exit_button_stops_page():
    exe = browser_check.find_browser()
    if not exe:
        pytest.skip("no Edge / Chrome installed")
    with browser_check.edge_lock():
        browser_check.wait_for_memory(1500, 600)
        if browser_check.have_playwright():
            with browser_check.PwEdge() as drv:
                assert drv.exit_check()
        else:
            assert browser_check.exit_check(exe)


def test_mobile_and_desktop_sizes():
    pytest.importorskip("playwright")
    if not browser_check.find_browser():
        pytest.skip("no Edge / Chrome installed")
    out = os.path.join(tempfile.gettempdir(), "snake_mobile_check_pytest")
    rep = browser_check.mobile_check(None, out)
    bad = {s["name"]: s["fails"] for s in rep["sizes"] if s["fails"]}
    assert rep["ok"] and not bad, bad
    assert len(rep["sizes"]) == 6
