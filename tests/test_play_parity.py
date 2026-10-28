# -*- coding: utf-8 -*-
"""The browser version (play/js/*.js) must match the Python / PyTorch originals.

Runs the JS through Node (tests/js/harness.js); skipped if `node` is not installed.
  - PyRandom   == CPython random.Random            (same food positions per seed)
  - SnakeEnv   == snake_env.SnakeEnv                (state, rewards, deaths, step by step)
  - BrainNet   == model.SnakeCNN with float16 weights (Q, activations, gradients, knock-out)
  - JS greedy rollouts == PyTorch greedy rollouts  (same seeds -> same scores)
  - inputGroups in config.js == the channels SnakeEnv.state() really writes
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

from snake_env import SnakeEnv  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
HARNESS = ROOT / "tests" / "js" / "harness.js"
MODEL_JS = ROOT / "play" / "model" / "snake_cnn_fp16.js"


def run_js(cmd, tmp_path):
    f = tmp_path / "cmd.json"
    f.write_text(json.dumps(cmd), encoding="utf-8")
    p = subprocess.run([NODE, str(HARNESS), str(f)], capture_output=True, timeout=600)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    return json.loads(p.stdout.decode("utf-8"))


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
    # a small safe loop (up, left, down, right) circles until it eats or starves
    loop = [0, 2, 1, 3] * 200
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


def test_input_groups_match_env_channels(tmp_path):
    """Feature names on the brain panel must be the real input channels, in order."""
    cfg = run_js({"cmd": "config"}, tmp_path)
    groups = cfg["inputGroups"]
    assert [g["ch"] for g in groups] == list(range(SnakeEnv.CHANNELS))
    env = SnakeEnv(10, 3)
    s = env.reset()                                   # head (5,5), body (5,4),(5,3), moving right
    hr, hc = env.snake[0]
    br, bc = env.snake[1]
    fr, fc = env.food
    probe = {
        "head": s[:, hr, hc].argmax() == 0 and s[0].sum() == 1,
        "body": s[1, br, bc] > 0 and s[1, hr, hc] == 0 and s[0, br, bc] == 0,
        "food": s[2, fr, fc] == 1 and s[2].sum() == 1,
        "dir_dr": np.all(s[3] == 0),                  # moving right: dr = 0
        "dir_dc": np.all(s[4] == 1),                  #               dc = +1
    }
    keys = [g["key"] for g in groups]
    assert keys == ["head", "body", "food", "dir_dr", "dir_dc"]
    for g in groups:
        assert probe[g["key"]], g
    env.step(0)                                       # turn up -> dr plane -1, dc plane 0
    s = env.state()
    assert np.all(s[3] == -1) and np.all(s[4] == 0)
    names = [g["name"] for g in groups]
    assert names == ["蛇头", "身体", "食物", "方向·上下", "方向·左右"]
    assert [a["name"] for a in cfg["actions"]] == ["上", "下", "左", "右"]


# ------------------------------------------------------------------ network
torch = pytest.importorskip("torch")
from tools.common import build_net, load_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def ck():
    return load_checkpoint()


def _states(n, k=40, seed=11):
    """Mixed states: greedy-ish games plus random walks."""
    rng = random.Random(seed)
    env = SnakeEnv(n, seed)
    out = []
    while len(out) < k:
        s = env.reset()
        while not env.done and len(out) < k:
            if rng.random() < 0.5:
                out.append((s.copy(), env.legal_mask()))
            s, *_ = env.step(rng.randrange(4) if rng.random() < 0.3 else env.direction)
    return out


def _torch_ref(net, state, legal, knock):
    """Masked forward + autograd, mirroring BrainNet.forward."""
    import torch.nn.functional as F
    x = torch.from_numpy(state).unsqueeze(0).requires_grad_(True)
    masks = {k: torch.ones(m) for k, m in (("conv1", 32), ("conv2", 64), ("conv3", 64), ("fc1", 256))}
    for layer, idx in knock:
        masks[layer][idx] = 0
    a1 = F.relu(net.conv1(x)) * masks["conv1"].view(1, -1, 1, 1)
    a2 = F.relu(net.conv2(a1)) * masks["conv2"].view(1, -1, 1, 1)
    a3 = F.relu(net.conv3(a2)) * masks["conv3"].view(1, -1, 1, 1)
    h = F.relu(net.fc1(a3.flatten(1))) * masks["fc1"]
    q = net.fc2(h)[0]
    for t in (a1, a2, a3, h):
        t.retain_grad()
    act = int((q.detach() + torch.tensor(legal).log()).argmax())
    q[act].backward()
    c = lambda t: (t * t.grad).flatten(2).sum(2)[0].detach().numpy()  # noqa: E731
    return {"q": q.detach().numpy(), "action": act, "h": h[0].detach().numpy(),
            "dx": x.grad[0].numpy().ravel(), "a1": a1[0].detach().numpy().ravel(),
            "contrib": {"input": (x * x.grad).flatten(2).sum(2)[0].detach().numpy(),
                        "conv1": c(a1), "conv2": c(a2), "conv3": c(a3),
                        "fc1": (h * h.grad)[0].detach().numpy()},
            "saliency": x.grad[0].abs().sum(0).numpy().ravel()}


def test_model_js_is_current_checkpoint(ck):
    from tools.export_model import decode_js
    payload, w = decode_js(MODEL_JS)
    assert payload["meta"]["params"] == sum(v.numel() for v in ck["model"].values()) == 1696580
    assert payload["meta"]["n"] == ck["n"] and payload["meta"]["channels"] == ck["channels"]
    for k, v in ck["model"].items():
        ref = v.numpy().astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(w[k], ref)


def test_forward_and_gradients_match_torch(ck, tmp_path):
    net16 = build_net(ck, fp16=True)
    cases = []
    for i, (s, legal) in enumerate(_states(ck["n"])):
        knock = [] if i % 3 else [["conv2", i % 64], ["fc1", (7 * i) % 256], ["conv1", i % 32]]
        cases.append({"state": s.ravel().tolist(), "legal": legal, "knock": knock})
    js = run_js({"cmd": "forward", "cases": cases}, tmp_path)
    max_q = 0.0
    for c, got in zip(cases, js):
        ref = _torch_ref(net16, np.float32(c["state"]).reshape(5, ck["n"], ck["n"]), c["legal"], c["knock"])
        dq = np.abs(np.float32(got["q"]) - ref["q"]).max()
        max_q = max(max_q, dq)
        assert dq < 1e-4
        assert got["action"] == ref["action"]
        np.testing.assert_allclose(got["h"], ref["h"], atol=1e-4)
        np.testing.assert_allclose(got["a1"], ref["a1"], atol=1e-5)
        sc = max(1e-6, np.abs(ref["dx"]).max())
        np.testing.assert_allclose(np.float32(got["dx"]) / sc, ref["dx"] / sc, atol=1e-3)
        np.testing.assert_allclose(np.float32(got["saliency"]) / sc, ref["saliency"] / sc, atol=1e-3)
        for k in ("input", "conv1", "conv2", "conv3", "fc1"):
            r = ref["contrib"][k]
            s2 = max(1e-6, np.abs(r).max())
            np.testing.assert_allclose(np.float32(got["contrib"][k]) / s2, r / s2, atol=2e-3, err_msg=k)
        assert got["params"] == 1696580


def test_fp16_close_to_fp32(ck):
    """The browser's float16 weights stay close to the original float32 network."""
    n32, n16 = build_net(ck), build_net(ck, fp16=True)
    worst = 0.0
    with torch.no_grad():
        for s, _ in _states(ck["n"], k=200, seed=3):
            x = torch.from_numpy(s).unsqueeze(0)
            worst = max(worst, float((n32(x) - n16(x)).abs().max()))
    assert worst < 5e-3


def test_js_greedy_rollouts_match_torch(ck, tmp_path):
    """Watch mode in the browser replays train.py's evaluation games (seed 999)."""
    from tools.eval_model import rollout_games
    games = 30
    js = run_js({"cmd": "greedy", "n": ck["n"], "seed": 999, "games": games}, tmp_path)
    py = rollout_games(build_net(ck, fp16=True), ck["n"], games, 999)
    assert [(g["score"], g["steps"]) for g in js] == [(g["score"], g["steps"]) for g in py]
    # ... and these are exactly the 30 evaluation games recorded in results/result.json
    res = json.loads((ROOT / "results" / "result.json").read_text(encoding="utf-8"))
    e30 = res["eval"]["float32"]["eval30_seed999"]
    assert [g["score"] for g in js] == e30["scores"] and [g["steps"] for g in js] == e30["steps"]
