# -*- coding: utf-8 -*-
"""The browser version (play/js/*.js) must match the Python originals, for every brain.

Runs the JS through Node (tests/js/harness.js); skipped if `node` is not installed.
  - PyRandom   == CPython random.Random                 (same food positions / samples per seed)
  - SnakeEnv   == snake_env.SnakeEnv                     (state, rewards, deaths, step by step)
  - observations and action masks of every brain == brains.py (bit-exact)
  - BrainNet forward / gradients / knock-out == brains.py NumPy reference (all 4 brains)
  - greedy / sampled rollouts in JS == NumPy, and == results/brains.json
  - config.js brains list == the model files that exist; DQN export == checkpoint (if PyTorch is installed)
"""
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from brains import MASKS, OBS, brain_files, load_brain  # noqa: E402
from snake_env import SnakeEnv  # noqa: E402
from tools.eval_brains import config_brains, play  # noqa: E402

NODE = shutil.which("node") or (r"C:\Program Files\nodejs\node.exe" if os.name == "nt" else None)
if NODE and not Path(NODE).exists():
    NODE = None
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
HARNESS = ROOT / "tests" / "js" / "harness.js"
BRAIN_IDS = [b["id"] for b in config_brains()]


def run_js(cmd, tmp_path):
    f = tmp_path / "cmd.json"
    f.write_text(json.dumps(cmd), encoding="utf-8")
    p = subprocess.run([NODE, str(HARNESS), str(f)], capture_output=True, timeout=1800)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    return json.loads(p.stdout.decode("utf-8"))


def brain_cfg(bid):
    return [b for b in config_brains() if b["id"] == bid][0]


# ------------------------------------------------------------------ random / env
def test_pyrandom_matches_cpython(tmp_path):
    seeds = [0, 1, 42, 999, 1001, 123456789, 2 ** 32 + 7, 2 ** 40 + 3]
    ns = [1, 2, 3, 7, 10, 64, 97, 100, 1000, 2 ** 31 - 1]
    js = run_js({"cmd": "random", "seeds": seeds, "ns": ns}, tmp_path)
    for s, got in zip(seeds, js):
        r = random.Random(s)
        floats = [r.random() for _ in range(5)]
        below = [r.randrange(n) for n in ns]          # randrange(n) == _randbelow(n)
        assert got["floats"] == floats, s
        assert got["below"] == below, s


def _py_traj(n, seed, actions):
    env = SnakeEnv(n, seed)
    s = env.reset()
    out = [{"state": s.ravel().tolist(), "snake": [list(p) for p in env.snake],
            "food": list(env.food) if env.food else None, "score": 0, "steps": 0,
            "dir": env.direction, "done": False, "reward": None, "reason": ""}]
    for a in actions:
        if env.done:
            break
        s, r, d, info = env.step(a)
        out.append({"state": s.ravel().tolist(), "snake": [list(p) for p in env.snake],
                    "food": list(env.food) if env.food else None, "score": env.score,
                    "steps": env.steps, "dir": env.direction, "done": env.done, "reward": r,
                    "reason": info.get("reason", "")})
    return out


def test_env_matches_python_step_by_step(tmp_path):
    rng = np.random.default_rng(7)
    runs = []
    for k in range(12):
        n = [10, 10, 10, 6, 4, 12][k % 6]
        acts = [int(a) for a in rng.choice(4, size=400)]   # random walk (includes 180-degree turns)
        runs.append({"n": n, "seed": 1000 + k, "actions": acts})
    loop = [0, 2, 1, 3] * 200                               # a safe loop circles until it eats or starves
    runs.append({"n": 10, "seed": 5, "actions": loop})
    js = run_js({"cmd": "env", "runs": runs}, tmp_path)
    reasons = set()
    for run, jt in zip(runs, js):
        pt = _py_traj(run["n"], run["seed"], run["actions"])
        assert len(pt) == len(jt)
        for a, b in zip(pt, jt):
            assert a["snake"] == b["snake"] and a["food"] == b["food"]
            assert (a["score"], a["steps"], a["dir"], a["done"], a["reason"]) == \
                   (b["score"], b["steps"], b["dir"], b["done"], b["reason"])
            if a["reward"] is not None:
                assert abs(a["reward"] - b["reward"]) < 1e-12
            np.testing.assert_array_equal(np.float32(a["state"]), np.float32(b["state"]))
        reasons.add(pt[-1]["reason"])
    assert "撞墙/撞身" in reasons and "饥饿超时" in reasons


def _good_walk(n, seed, steps):
    """A long trajectory that mostly survives (brain A's greedy moves, 10% random) so the snake grows."""
    b = load_brain(brain_files()["ppo_linyi"]) if n == 12 else None
    env = SnakeEnv(n, seed)
    env.reset()
    rng = random.Random(seed)
    acts = []
    while len(acts) < steps and not env.done:
        a = rng.randrange(4) if (b is None or rng.random() < 0.1) else b.act(env)[0]
        acts.append(a)
        env.step(a)
    return acts


@pytest.mark.parametrize("n,seed,steps", [(12, 11, 900), (12, 12, 400), (10, 13, 200)])
def test_observations_and_masks_match_python(tmp_path, n, seed, steps):
    """Every observation builder and action mask is bit-exact (gray levels from np.linspace included)."""
    kinds = ["dqn5", "linyi_cnn", "ray_cnn", "linyi_mlp"]
    acts = _good_walk(n, seed, steps)
    js = run_js({"cmd": "obs", "n": n, "seed": seed, "actions": acts, "kinds": kinds}, tmp_path)
    env = SnakeEnv(n, seed)
    env.reset()
    i = 0
    lengths = set()

    def check():
        o = js[i]
        for k in kinds:
            np.testing.assert_array_equal(np.float32(o[k]), OBS[k](env), err_msg=k)
        assert o["safe"] == MASKS["safe"](env)
        assert o["turn"] == MASKS["turn"](env)
        lengths.add(len(env.snake))
    check()
    for a in acts:
        if env.done:
            break
        env.step(a)
        if not env.done:
            i += 1
            check()
    assert i == len(js) - 1
    assert len(lengths) >= 3 if n == 12 and steps >= 400 else True


# ------------------------------------------------------------------ networks
def _states(brain, k, seed):
    """Mixed states: the brain's own games with 30% random moves."""
    rng = random.Random(seed)
    env = SnakeEnv(brain.n, seed)
    out = []
    while len(out) < k:
        env.reset()
        while not env.done and len(out) < k:
            if rng.random() < 0.5:
                out.append((brain.obs(env), brain.env_mask_to_brain(brain.mask_fn(env))))
            a = rng.randrange(4) if rng.random() < 0.3 else brain.act(env)[0]
            env.step(a)
    return out


@pytest.mark.parametrize("bid", BRAIN_IDS)
def test_forward_and_gradients_match_numpy(tmp_path, bid):
    cfg = brain_cfg(bid)
    b = load_brain(ROOT / "play" / cfg["file"])
    hidden = b.spec["trunk"] + b.spec["pi"][:-1]
    cases = []
    for i, (obs, mask) in enumerate(_states(b, 12, 21)):
        knock = [] if i % 3 else [[hidden[0]["name"], i % 8], [hidden[-1]["name"], (7 * i) % 64]]
        cases.append({"obs": obs.tolist(), "mask": mask, "knock": knock})
    js = run_js({"cmd": "forward", "model": cfg["file"], "cases": cases}, tmp_path)
    for c, got in zip(cases, js):
        b.knock = {}
        for layer, idx in c["knock"]:
            b.knock.setdefault(layer, set()).add(idx)
        ref = b.forward(np.float32(c["obs"]), c["mask"], grad=True)
        assert got["params"] == b.params
        np.testing.assert_allclose(np.float64(got["q"]), ref["logits"], atol=1e-4)
        if ref["value"] is not None:
            assert abs(got["value"] - ref["value"]) < 1e-4
        assert got["action"] == ref["action"]
        for name in [L["name"] for L in hidden]:
            a = ref["acts"][name]["a"].ravel()
            np.testing.assert_allclose(np.float64(got["acts"][name]), a, atol=1e-4 * max(1.0, np.abs(a).max()), err_msg=name)
        for k, v in ref["contrib"].items():
            sc = max(1e-6, np.abs(v).max())
            np.testing.assert_allclose(np.float64(got["contrib"][k]) / sc, v / sc, atol=2e-3, err_msg=k)
        sc = max(1e-6, np.abs(ref["saliency"]).max())
        np.testing.assert_allclose(np.float64(got["saliency"]) / sc, ref["saliency"].ravel() / sc, atol=2e-3)
        sc = max(1e-6, np.abs(ref["dx"]).max())
        np.testing.assert_allclose(np.float64(got["dx"]) / sc, ref["dx"].ravel() / sc, atol=2e-3)


def test_numpy_gradient_matches_finite_differences():
    """brains.py's hand-written backward pass is checked against finite differences (small MLP brain)."""
    b = load_brain(brain_files()["ppo_mlp_linyi"])
    env = SnakeEnv(b.n, 4)
    env.reset()
    for _ in range(7):
        env.step(b.act(env)[0])
    x = b.obs(env).astype(np.float64)
    mask = b.env_mask_to_brain(b.mask_fn(env))
    ref = b.forward(np.float32(x), mask, grad=True)
    a = ref["action"]
    eps = 1e-4                                   # small enough not to cross a ReLU kink (1e-3 does)
    worst = 0.0
    for idx in np.random.default_rng(0).choice(x.size, 25, replace=False):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fd = (b.forward(np.float32(xp), mask)["logits"][a] - b.forward(np.float32(xm), mask)["logits"][a]) / (2 * eps)
        worst = max(worst, abs(fd - ref["dx"].ravel()[idx]))
    assert worst < 5e-3 * max(1.0, np.abs(ref["dx"]).max())


@pytest.mark.parametrize("bid", BRAIN_IDS)
def test_js_rollouts_match_numpy_and_recorded_results(tmp_path, bid):
    """Watch mode replays the evaluation games (seed 999): JS == NumPy == results/brains.json (first 10 games)."""
    cfg = brain_cfg(bid)
    b = load_brain(ROOT / "play" / cfg["file"])
    rec = json.loads((ROOT / "results" / "brains.json").read_text(encoding="utf-8"))
    d = rec["brains"][bid]["eval"]["default"]
    games = min(10, d["games"])
    guard = "safe" if cfg["guard"] else "trained"
    js = run_js({"cmd": "rollout", "model": cfg["file"], "games": games, "seed": rec["seed"], "policy": cfg["policy"], "guard": guard}, tmp_path)
    py = play(b, games, rec["seed"], cfg["policy"], guard)
    assert [(g["score"], g["steps"]) for g in js] == [(g["score"], g["steps"]) for g in py]
    assert [g["score"] for g in js] == d["scores"][:games] and [g["steps"] for g in js] == d["steps"][:games]
    assert d["policy"] == cfg["policy"] and d["guard"] == guard


def test_config_lists_exactly_the_model_files(tmp_path):
    cfg = run_js({"cmd": "config"}, tmp_path)
    ids = [b["id"] for b in cfg["brains"]]
    assert ids == BRAIN_IDS and len(set(ids)) == len(ids)
    for b in cfg["brains"]:
        assert (ROOT / "play" / b["file"]).is_file(), b["file"]
        assert b["policy"] in ("greedy", "sample") and isinstance(b["guard"], bool)
        assert b["source"] and b["license"]
    assert [a["name"] for a in cfg["actions"]] == ["上", "下", "左", "右"]
    for kind, groups in cfg["inputs"].items():
        env = SnakeEnv(12 if kind != "dqn5" else 10, 1)
        env.reset()
        c = OBS[kind](env).size // (SnakeEnv.CHANNELS * 100 if kind == "dqn5" else 1)
        assert len(groups) == {"dqn5": 5, "linyi_cnn": 3, "ray_cnn": 3, "linyi_mlp": 1}[kind]
    assert [g["name"] for g in cfg["inputs"]["dqn5"]] == ["蛇头", "身体", "食物", "方向·上下", "方向·左右"]


def test_dqn_js_is_current_checkpoint():
    torch = pytest.importorskip("torch")
    from tools.common import load_checkpoint
    ck = load_checkpoint()
    b = load_brain(brain_files()["dqn_own"])
    assert b.params == sum(v.numel() for v in ck["model"].values()) == 1696580
    for k, v in ck["model"].items():
        ref = v.numpy().astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(b.w[k], ref)
