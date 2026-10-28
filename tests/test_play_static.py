# -*- coding: utf-8 -*-
"""play/ must work from file:// with no install and no network, and stay small."""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAY = ROOT / "play"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _texts():
    for p in PLAY.rglob("*"):
        if p.is_file() and p.suffix in (".html", ".js", ".css"):
            t = p.read_text(encoding="utf-8")
            if p.name == "snake_cnn_fp16.js":
                t = re.sub(r'"b64":"[^"]*"', '"b64":""', t)
            yield p, t


def test_ascii_paths():
    for p in PLAY.rglob("*"):
        assert p.relative_to(PLAY).as_posix().isascii(), p


def test_scripts_and_styles_are_local_and_exist():
    html = (PLAY / "index.html").read_text(encoding="utf-8")
    srcs = re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
    hrefs = re.findall(r'<link[^>]*\shref="([^"]+)"', html)
    assert srcs and hrefs
    for s in srcs + hrefs:
        assert not re.match(r"^[a-z]+:", s), s
        assert (PLAY / s).is_file(), s
    assert 'type="module"' not in html
    # model must be loaded before app.js
    assert srcs.index("model/snake_cnn_fp16.js") < srcs.index("js/app.js")


def test_no_network_no_module_no_service_worker():
    for p, t in _texts():
        assert not re.search(r"https?://", t), f"URL in {p}"
        for bad in ("fetch(", "XMLHttpRequest", "serviceWorker", "import(", "@import", "@font-face", 'type="module"'):
            assert bad not in t, f"{bad} in {p}"


def test_required_ui_parts():
    html = (PLAY / "index.html").read_text(encoding="utf-8")
    for needle in ('data-mode="watch"', 'data-mode="play"', 'data-mode="practice"', 'id="btn-exit"',
                   'id="btn-help"', 'id="ck-big"', 'id="foot"', 'id="formula"', 'id="btn-restore"',
                   # 手机竖屏（v2 第八节）：☰ 菜单、操作区（摇杆 + 大圆键）、看大脑 / 返回、横屏提示、触屏图解
                   'id="menu-btn"', 'id="menu-exit"', 'id="touchctl"', 'id="stick-zone"', 'id="stick-knob"',
                   'data-tk="pause"', 'id="btn-brain"', 'id="brain-close"', 'id="rotate-hint"', 'class="demo demo-stick touch-only"',
                   'viewport-fit=cover'):
        assert needle in html, needle
    app = (PLAY / "js" / "app.js").read_text(encoding="utf-8")
    assert "已停止运行，所有计算都已结束，可以直接关闭这个标签页" in app
    assert "window.close()" in app
    css = (PLAY / "css" / "app.css").read_text(encoding="utf-8")
    for needle in ("body.touchp", "100dvh", "safe-area-inset", "touch-action: none", "user-select: none", "#rotate-hint"):
        assert needle in css, needle
    touch = (PLAY / "js" / "touchctl.js").read_text(encoding="utf-8")
    for needle in ("pointerId", "navigator.vibrate", "contextmenu", "touchcancel", '"blur"'):
        assert needle in touch, needle


def test_credits_filled():
    t = (PLAY / "js" / "credits.js").read_text(encoding="utf-8")
    data = json.loads(t[t.index("{"): t.rindex("}") + 1])
    assert data["AUTHOR"] == "Shuyao Gao"
    assert data["AUTHOR_EMAIL"] == "gaoshuyao.mk@gmail.com"      # public contact address (v2 署名)
    assert "aSSIST University" in data["UNIVERSITY_EN"] and data["UNIVERSITY_KO"]
    assert "{{" not in t


def test_size_budget():
    total = sum(p.stat().st_size for p in PLAY.rglob("*") if p.is_file())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in PLAY.rglob("*"):
            if p.is_file():
                z.write(p, "play/" + p.relative_to(PLAY).as_posix())
    assert len(buf.getvalue()) <= 15 * 1024 * 1024
    assert total <= 15 * 1024 * 1024


def test_result_json_consistent_with_checkpoint():
    r = json.loads((ROOT / "results" / "result.json").read_text(encoding="utf-8"))
    ck = r["checkpoint"]
    assert ck["params"] == sum(ck["tensor_sizes"].values())
    e30 = r["eval"]["float32"]["eval30_seed999"]
    # train.py's final evaluation (30 games, seed 999) is what the checkpoint stored
    assert abs(e30["avg_score"] - ck["avg_score_stored_in_checkpoint"]) < 1e-9
    assert r["training"]["recorded"] is False
