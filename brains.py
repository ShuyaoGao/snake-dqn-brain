# -*- coding: utf-8 -*-
"""Reference implementation (NumPy only, no PyTorch) of every "brain" the page can run.

A brain = observation builder + network spec + float16 weights, all read from the exported
play/model/*.js file, so Python and the browser use byte-identical weights.

    from brains import load_brain, rollout
    b = load_brain("play/model/ppo_cnn_ray_fp16.js")
    games = rollout(b, games=30, seed=999)

Observation builders reproduce the original projects' wrappers exactly (see THIRD_PARTY.md):
  dqn5       this repository's snake_env.SnakeEnv.state()           (5, N, N) float
  linyi_cnn  linyiLYi/snake-ai  snake_game_custom_wrapper_cnn.py     (3, 84, 84) = 12x12 cells x 7 px, /255
  ray_cnn    Part-time-Ray/AI-Snake-Game  wrapper.py get_obs()       (3, 60, 60) = 12x12 cells x 5 px, /255
  linyi_mlp  linyiLYi/snake-ai  snake_game_custom_wrapper_mlp.py     (12, 12) float in [-1, 1]
The network spec (conv / fc layers, policy and value heads, action order, action mask) is stored
inside the .js payload by tools/export_sb3.py, so this module has no per-model code paths.
"""

import base64
import json
import re
from pathlib import Path

import numpy as np

from snake_env import DIRS, OPPOSITE, SnakeEnv

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- observations
def gray_u8(start, stop, L):
    """np.linspace(start, stop, L, dtype=np.uint8) as the original wrappers call it."""
    return np.linspace(start, stop, L, dtype=np.uint8)


def obs_dqn5(env):
    return env.state().ravel().astype(np.float32)


def obs_linyi_cnn(env):
    n = env.n
    g = np.zeros((n, n), dtype=np.uint8)
    rows = [p[0] for p in env.snake]
    cols = [p[1] for p in env.snake]
    g[rows, cols] = gray_u8(200, 50, len(env.snake))
    img = np.stack((g, g, g), axis=-1)
    hr, hc = env.snake[0]
    tr, tc = env.snake[-1]
    img[hr, hc] = [0, 255, 0]
    img[tr, tc] = [255, 0, 0]
    if env.food is not None:
        img[env.food[0], env.food[1]] = [0, 0, 255]
    img = np.repeat(np.repeat(img, 7, axis=0), 7, axis=1)          # (84, 84, 3)
    return (img.transpose(2, 0, 1).astype(np.float32) / 255.0).ravel()


def obs_ray_cnn(env):
    """Part-time-Ray's wrapper indexes the image as [x][y] (x = column), so the picture the
    network sees is the transposed board; directions keep their meaning (up = row - 1)."""
    n = env.n
    img = np.zeros((n, n, 3), dtype=np.uint8)
    gray = gray_u8(200, 50, len(env.snake))
    for i, (r, c) in enumerate(env.snake):
        img[c, r, :] = gray[i]
    hr, hc = env.snake[0]
    img[hc, hr] = [0, 255, 0]
    if env.food is not None:
        img[env.food[1], env.food[0]] = [0, 0, 255]
    img = np.repeat(np.repeat(img, 5, axis=0), 5, axis=1)          # (60, 60, 3)
    return (img.transpose(2, 0, 1).astype(np.float32) / 255.0).ravel()


def obs_linyi_mlp(env):
    n = env.n
    g = np.zeros((n, n), dtype=np.float32)
    rows = [p[0] for p in env.snake]
    cols = [p[1] for p in env.snake]
    g[rows, cols] = np.linspace(0.8, 0.2, len(env.snake), dtype=np.float32)
    g[env.snake[0][0], env.snake[0][1]] = 1.0
    if env.food is not None:
        g[env.food[0], env.food[1]] = -1.0
    return g.ravel()


OBS = {"dqn5": obs_dqn5, "linyi_cnn": obs_linyi_cnn, "ray_cnn": obs_ray_cnn, "linyi_mlp": obs_linyi_mlp}


# ---------------------------------------------------------------- action masks (in ENV action order 0=up 1=down 2=left 3=right)
def mask_turn(env):
    return [1.0 if a != OPPOSITE[env.direction] else 0.0 for a in range(4)]


def mask_safe(env):
    """linyiLYi/snake-ai SnakeEnv._check_action_validity: no 180-degree turn, no wall, no body
    (the tail cell counts as free unless this step eats the food)."""
    out = []
    hr, hc = env.snake[0]
    for a in range(4):
        if a == OPPOSITE[env.direction]:
            out.append(0.0)
            continue
        r, c = hr + DIRS[a][0], hc + DIRS[a][1]
        body = env.snake if (env.food is not None and (r, c) == tuple(env.food)) else env.snake[:-1]
        bad = (r, c) in body or r < 0 or r >= env.n or c < 0 or c >= env.n
        out.append(0.0 if bad else 1.0)
    if not any(out):                                   # trapped: fall back to the plain turn mask
        return mask_turn(env)
    return out


def mask_none(env):
    return [1.0, 1.0, 1.0, 1.0]


MASKS = {"turn": mask_turn, "safe": mask_safe, "none": mask_none}


# ---------------------------------------------------------------- model file
def decode_js(path):
    """Parse play/model/*.js (window.SNAKE_BRAINS["id"] = {...}; or window.SNAKE_MODEL = {...};)."""
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text[text.index('{"format"'): text.rindex("}") + 1])
    raw = np.frombuffer(base64.b64decode(payload["b64"]), dtype="<f2")
    w = {}
    for t in payload["tensors"]:
        w[t["name"]] = raw[t["offset"]: t["offset"] + t["length"]].astype(np.float32).reshape(t["shape"])
    return payload, w


def spec_from_v1(meta):
    """The original DQN export (format fp16-b64-v1) described only n / channels; rebuild its spec."""
    n, ch = meta["n"], meta["channels"]
    conv = lambda name, cin, cout: {"name": name, "kind": "conv", "cin": cin, "cout": cout, "k": 3, "s": 1, "p": 1,  # noqa: E731
                                    "hin": n, "win": n, "hout": n, "wout": n}
    return {"obs": "dqn5", "n": n, "input": {"c": ch, "h": n, "w": n},
            "trunk": [conv("conv1", ch, 32), conv("conv2", 32, 64), conv("conv3", 64, 64),
                      {"name": "fc1", "kind": "fc", "in": 64 * n * n, "out": 256}],
            "pi": [{"name": "fc2", "kind": "fc", "in": 256, "out": 4}], "v": [],
            "actionMap": [0, 1, 2, 3], "mask": "turn", "out": "q"}


def load_brain(path):
    payload, w = decode_js(path)
    spec = payload.get("spec") or spec_from_v1(payload["meta"])
    return Brain(spec, w, payload.get("meta", {}))


# ---------------------------------------------------------------- network
def conv2d(x, wt, b, s, p):
    """x (cin, h, w) -> (cout, ho, wo); float64 accumulation. Plain loops over the kernel."""
    cin, h, wd = x.shape
    cout, _, k, _ = wt.shape
    if p:
        x = np.pad(x, ((0, 0), (p, p), (p, p)))
        h, wd = h + 2 * p, wd + 2 * p
    ho, wo = (h - k) // s + 1, (wd - k) // s + 1
    out = np.zeros((cout, ho, wo), dtype=np.float64)
    xw = x.astype(np.float64)
    ww = wt.astype(np.float64)
    for ky in range(k):
        for kx in range(k):
            patch = xw[:, ky: ky + s * ho: s, kx: kx + s * wo: s]       # (cin, ho, wo)
            out += np.tensordot(ww[:, :, ky, kx], patch, axes=([1], [0]))
    out += b.astype(np.float64)[:, None, None]
    return out


def conv2d_back(dz, wt, s, p, hin, win):
    """gradient w.r.t. the (unpadded) conv input given dz (cout, ho, wo)."""
    cout, cin, k, _ = wt.shape
    _, ho, wo = dz.shape
    dx = np.zeros((cin, hin + 2 * p, win + 2 * p), dtype=np.float64)
    ww = wt.astype(np.float64)
    for ky in range(k):
        for kx in range(k):
            dx[:, ky: ky + s * ho: s, kx: kx + s * wo: s] += np.tensordot(ww[:, :, ky, kx], dz, axes=([0], [0]))
    if p:
        dx = dx[:, p: p + hin, p: p + win]
    return dx


class Brain:
    def __init__(self, spec, w, meta=None):
        self.spec, self.w, self.meta = spec, w, meta or {}
        self.n = spec["n"]
        self.obs_fn = OBS[spec["obs"]]
        self.mask_fn = MASKS[spec["mask"]]
        self.action_map = list(spec["actionMap"])
        self.params = int(sum(v.size for v in w.values()))
        self.knock = {}                                  # layer name -> set of knocked units / channels

    # -- helpers
    def obs(self, env):
        return self.obs_fn(env)

    def env_mask_to_brain(self, m):
        return [m[self.action_map[i]] for i in range(4)]

    def _mask_vec(self, name, size):
        v = np.ones(size)
        for i in self.knock.get(name, ()):
            v[i] = 0.0
        return v

    def forward(self, x, brain_mask=None, grad=False):
        """x: flat float32 obs. brain_mask: 4 floats in BRAIN action order. Returns dict with
        logits (brain order), value, action (brain index), acts, and if grad: contributions."""
        sp = self.spec
        c, h, wd = sp["input"]["c"], sp["input"]["h"], sp["input"]["w"]
        cur = np.asarray(x, dtype=np.float32).reshape(c, h, wd).astype(np.float64)
        acts = {}
        trunk = sp["trunk"]
        for L in trunk:
            z, a = self._layer(L, cur)
            acts[L["name"]] = {"z": z, "a": a}
            cur = a
        feat = cur
        pi_layers = sp["pi"]
        cur = feat
        for i, L in enumerate(pi_layers):
            last = i == len(pi_layers) - 1
            z, a = self._layer(L, cur, relu=not last)
            acts[L["name"]] = {"z": z, "a": a}
            cur = a
        logits = cur.reshape(-1)
        value = None
        if sp["v"]:
            cv = feat
            for i, L in enumerate(sp["v"]):
                last = i == len(sp["v"]) - 1
                z, a = self._layer(L, cv, relu=not last)
                cv = a
            value = float(cv.reshape(-1)[0])
        m = np.array(brain_mask if brain_mask is not None else [1, 1, 1, 1], dtype=np.float64)
        masked = np.where(m > 0, logits, -np.inf)
        action = int(np.argmax(masked))
        res = {"logits": logits, "value": value, "action": action, "acts": acts, "mask": m}
        if grad:
            res.update(self._backward(sp, x, acts, action))
        return res

    def _layer(self, L, x, relu=True):
        name = L["name"]
        if L["kind"] == "conv":
            z = conv2d(x, self.w[name + ".weight"], self.w[name + ".bias"], L["s"], L["p"])
            a = np.maximum(z, 0) * self._mask_vec(name, L["cout"])[:, None, None] if relu else z
        else:
            xf = x.reshape(-1)
            z = self.w[name + ".weight"].astype(np.float64) @ xf + self.w[name + ".bias"].astype(np.float64)
            a = np.maximum(z, 0) * self._mask_vec(name, L["out"]) if relu else z
        return z, a

    def _backward(self, sp, x, acts, action):
        """d logits[action] / d(everything): per-unit contributions (activation x gradient) and input saliency."""
        layers = sp["trunk"] + sp["pi"]
        n_last = len(layers) - 1
        g = None                                         # gradient w.r.t. the current layer's OUTPUT activation
        contrib = {}
        for i in range(n_last, -1, -1):
            L = layers[i]
            name = L["name"]
            z = acts[name]["z"]
            if i == n_last:
                dz = np.zeros_like(z)
                dz.reshape(-1)[action] = 1.0
            else:
                mvec = self._mask_vec(name, L["cout"] if L["kind"] == "conv" else L["out"])
                mv = mvec[:, None, None] if L["kind"] == "conv" else mvec
                dz = np.where(z > 0, g * mv, 0.0)
                a = acts[name]["a"]
                contrib[name] = (a * g).reshape(a.shape[0], -1).sum(1) if L["kind"] == "conv" else a * g
            if L["kind"] == "conv":
                g = conv2d_back(dz, self.w[name + ".weight"], L["s"], L["p"], L["hin"], L["win"])
            else:
                g = (self.w[name + ".weight"].astype(np.float64).T @ dz.reshape(-1))
                prev_shape = (acts[layers[i - 1]["name"]]["a"].shape if i > 0
                              else (sp["input"]["c"], sp["input"]["h"], sp["input"]["w"]))
                g = g.reshape(prev_shape)
        xin = np.asarray(x, dtype=np.float64).reshape(sp["input"]["c"], sp["input"]["h"], sp["input"]["w"])
        contrib["input"] = (xin * g).reshape(xin.shape[0], -1).sum(1)
        return {"dx": g, "contrib": contrib, "saliency": np.abs(g).sum(0)}

    def act(self, env):
        """Greedy action in ENV order (after the brain's own action map and mask)."""
        bm = self.env_mask_to_brain(self.mask_fn(env))
        r = self.forward(self.obs(env), bm)
        return self.action_map[r["action"]], r


def rollout(brain, games, seed, max_steps=5000):
    env = SnakeEnv(brain.n, seed)
    out = []
    for _ in range(games):
        env.reset()
        info = {}
        while not env.done and env.steps < max_steps:
            a, _ = brain.act(env)
            _, _, _, info = env.step(a)
        out.append({"score": env.score, "steps": env.steps, "length": len(env.snake),
                    "reason": info.get("reason", "达到步数上限")})
    return out


def summarize(games):
    sc = [g["score"] for g in games]
    st = [g["steps"] for g in games]
    reasons = {}
    for g in games:
        reasons[g["reason"]] = reasons.get(g["reason"], 0) + 1
    return {"games": len(games), "avg_score": float(np.mean(sc)), "avg_steps": float(np.mean(st)),
            "avg_length": float(np.mean([g["length"] for g in games])),
            "max_score": int(max(sc)), "min_score": int(min(sc)), "end_reasons": reasons,
            "scores": sc, "steps": st}


def brain_files():
    """id -> path of every exported brain under play/model/ (from play/js/config.js's `brains`)."""
    cfg = (ROOT / "play" / "js" / "config.js").read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'id:\s*"([a-z0-9_]+)"[^}]*?file:\s*"([^"]+)"', cfg):
        out[m.group(1)] = ROOT / "play" / m.group(2)
    return out
