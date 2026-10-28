# -*- coding: utf-8 -*-
"""Fill every {{PLACEHOLDER}} from credits.json (author, series, GitHub user, repo name ...)
and the numbers from results/result.json, then rebuild play/js/credits.js.

    python -m tools.fill_credits            # render tools/templates/*.tmpl -> repo root
    python -m tools.fill_credits --release  # ... and rebuild release/*.zip

Templates live in tools/templates/ (README.md.tmpl -> README.md, ...). Re-running is safe:
outputs are always regenerated from the templates. Finally it lists any placeholder that is
still unfilled anywhere in the repository.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "tools" / "templates"
PH = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
TEXT_EXT = {".md", ".txt", ".cff", ".json", ".js", ".html", ".css", ".py", ".ini", ".tmpl", ""}


def numbers_from_results():
    p = ROOT / "results" / "result.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text(encoding="utf-8"))
    ck, ev = r["checkpoint"], r["eval"]
    t = ck["tensor_sizes"]
    lay = lambda k: t[k + ".weight"] + t[k + ".bias"]  # noqa: E731
    e30, e20, h30 = ev["float32"]["eval30_seed999"], ev["float32"]["eval20_seed999"], ev["float16"]["eval30_seed999"]
    g = lambda v: f"{v:g}" if abs(v - round(v, 2)) < 1e-9 else f"{v:.2f}"  # noqa: E731
    return {
        "PARAMS": f"{ck['params']:,}", "P_CONV1": f"{lay('conv1'):,}", "P_CONV2": f"{lay('conv2'):,}",
        "P_CONV3": f"{lay('conv3'):,}", "P_FC1": f"{lay('fc1'):,}", "P_FC2": f"{lay('fc2'):,}",
        "FC1_SHARE": f"{100 * lay('fc1') / ck['params']:.1f}",
        "EVAL30_AVG_SCORE": g(e30["avg_score"]), "EVAL30_AVG_STEPS": f"{e30['avg_steps']:.1f}",
        "EVAL30_MAX": str(e30["max_score"]), "EVAL20_AVG_SCORE": g(e20["avg_score"]),
        "EVAL20_AVG_STEPS": f"{e20['avg_steps']:.1f}", "EVAL20_MAX": str(e20["max_score"]),
        "EVAL30_FP16_AVG_SCORE": g(h30["avg_score"]),
        "CKPT_AVG_SCORE": g(ck["avg_score_stored_in_checkpoint"]), "CKPT_BYTES": f"{ck['bytes']:,}",
    }


def render(values):
    written = []
    for tpl in sorted(TEMPLATES.glob("*.tmpl")):
        text = tpl.read_text(encoding="utf-8")
        text = PH.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), text)
        out = ROOT / tpl.name[:-len(".tmpl")]
        out.write_text(text, encoding="utf-8", newline="\n")
        written.append(out)
    return written


def leftover_placeholders():
    left = []
    for p in ROOT.rglob("*"):
        if (not p.is_file() or ".git" in p.parts or "templates" in p.parts or p.suffix not in TEXT_EXT
                or p.resolve() == Path(__file__).resolve()):
            continue
        if p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in PH.finditer(text):
            left.append((p.relative_to(ROOT).as_posix(), m.group(0)))
    return left


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", action="store_true")
    args = ap.parse_args(argv)
    if not TEMPLATES.is_dir():
        raise SystemExit("tools/templates/ not found (run this inside the published repository)")
    credits = json.loads((ROOT / "credits.json").read_text(encoding="utf-8"))
    values = {k: v for k, v in credits.items() if not k.startswith("_")}
    values.update(numbers_from_results())
    for p in render(values):
        print("wrote", p.relative_to(ROOT).as_posix())
    from tools import build
    build.main(["--skip-model"] + (["--release"] if args.release else []))
    left = leftover_placeholders()
    if left:
        print("UNFILLED placeholders:")
        for f, ph in left:
            print("  ", f, ph)
    else:
        print("no unfilled placeholders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
