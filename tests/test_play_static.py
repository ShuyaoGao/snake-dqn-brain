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
            if p.parent.name == "model":
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
    # every model listed in config.js is loaded before app.js
    cfg = (PLAY / "js" / "config.js").read_text(encoding="utf-8")
    files = re.findall(r'file:\s*"([^"]+)"', cfg)
    assert len(files) == 4
    for f in files:
        assert f in srcs and srcs.index(f) < srcs.index("js/app.js"), f


def test_no_network_no_module_no_service_worker():
    for p, t in _texts():
        if p.name == "config.js":
            # the only URLs are the upstream repositories named for attribution (opened by the user, never fetched)
            t = re.sub(r'url:\s*"https://github\.com/[^"]*"', 'url:""', t)
        if p.parent.name == "model":
            t = re.sub(r'\* Source: .*', '', t)        # attribution comment of the exported upstream weights
            t = re.sub(r'"url":"https://github\.com/[^"]*"', '"url":""', t)
        assert not re.search(r"https?://", t), f"URL in {p}"
        for bad in ("fetch(", "XMLHttpRequest", "serviceWorker", "import(", "@import", "@font-face", 'type="module"'):
            assert bad not in t, f"{bad} in {p}"


def test_required_ui_parts():
    html = (PLAY / "index.html").read_text(encoding="utf-8")
    for needle in ('id="btn-exit"', 'id="btn-help"', 'id="ck-big"', 'id="foot"', 'id="formula"', 'id="btn-restore"',
                   'id="brain-body"', 'id="opts"', 'id="menu-btn"', 'id="menu-exit"', 'id="menu-restore"',
                   'class="touch-list touch-only"', 'viewport-fit=cover'):
        assert needle in html, needle
    for gone in ('data-mode="play"', 'data-mode="practice"', 'id="touchctl"', 'touchctl.js'):
        assert gone not in html, gone
    app = (PLAY / "js" / "app.js").read_text(encoding="utf-8")
    assert "已停止运行，所有计算都已结束，可以直接关闭这个标签页" in app
    assert "window.close()" in app
    for needle in ("setBrain(", "data-kind='brain'", "ck-guard", "data-kind='policy'", "toggleKnock("):
        assert needle in app, needle
    css = (PLAY / "css" / "app.css").read_text(encoding="utf-8")
    for needle in ("body.touchp", "safe-area-inset", "user-select: none", "#menu-btn"):
        assert needle in css, needle
    assert not (PLAY / "js" / "touchctl.js").exists()


def test_credits_filled():
    t = (PLAY / "js" / "credits.js").read_text(encoding="utf-8")
    data = json.loads(t[t.index("{"): t.rindex("}") + 1])
    assert data["AUTHOR"] == "Shuyao Gao"
    assert data["AUTHOR_EMAIL"] == "gaoshuyao.mk@gmail.com"      # public contact address (v2 署名)
    assert "aSSIST University" in data["UNIVERSITY_EN"] and data["UNIVERSITY_KO"]
    assert "{{" not in t


def test_results_js_matches_results_json():
    t = (PLAY / "js" / "results.js").read_text(encoding="utf-8")
    res = json.loads(t[t.index("{"): t.rindex("}") + 1])
    rec = json.loads((ROOT / "results" / "brains.json").read_text(encoding="utf-8"))
    assert set(res) == set(rec["brains"])
    for bid, d in rec["brains"].items():
        assert res[bid]["default"]["avg_score"] == d["eval"]["default"]["avg_score"]
        assert res[bid]["default"]["max_score"] == d["eval"]["default"]["max_score"]


def test_size_budget():
    total = sum(p.stat().st_size for p in PLAY.rglob("*") if p.is_file())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in PLAY.rglob("*"):
            if p.is_file():
                z.write(p, "play/" + p.relative_to(PLAY).as_posix())
    assert len(buf.getvalue()) <= 15 * 1024 * 1024
    assert total <= 16 * 1024 * 1024


def test_result_json_consistent_with_checkpoint():
    r = json.loads((ROOT / "results" / "result.json").read_text(encoding="utf-8"))
    ck = r["checkpoint"]
    assert ck["params"] == sum(ck["tensor_sizes"].values())
    e30 = r["eval"]["float32"]["eval30_seed999"]
    # train.py's final evaluation (30 games, seed 999) is what the checkpoint stored
    assert abs(e30["avg_score"] - ck["avg_score_stored_in_checkpoint"]) < 1e-9
    assert r["training"]["recorded"] is False


def test_model_files_carry_attribution():
    for name, proj in (("ppo_cnn_linyi_fp16.js", "linyiLYi/snake-ai"), ("ppo_cnn_ray_fp16.js", "Part-time-Ray/AI-Snake-Game"),
                       ("ppo_mlp_linyi_fp16.js", "linyiLYi/snake-ai")):
        head = (PLAY / "model" / name).read_text(encoding="utf-8")[:600]
        assert proj in head and "license" in head
