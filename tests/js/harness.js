// Node harness used by tests/test_play_parity.py: loads the browser scripts from play/
// into a sandbox and runs one JSON command (argv[2] = path to command file).
// Prints a JSON result on stdout.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "..", "..");
const PLAY = path.join(ROOT, "play");

function load(names) {
  const ctx = { console, Math, Float32Array, Float64Array, Uint32Array, Set, Buffer, atob };
  ctx.window = ctx;
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  for (const n of names) vm.runInContext(fs.readFileSync(path.join(PLAY, n), "utf8"), ctx, { filename: n });
  return ctx;
}

const cmd = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const files = ["js/config.js", "js/pyrandom.js", "js/snake_env.js", "js/brain_net.js"];
if (cmd.model) files.push(cmd.model);                 // e.g. "model/ppo_cnn_ray_fp16.js"
const W = load(files);
const arr = (a) => Array.from(a);

/** The brain payload of a loaded model file (v2: window.SNAKE_BRAINS[id]; v1: window.SNAKE_MODEL). */
function payload() {
  if (W.SNAKE_BRAINS) { const ids = Object.keys(W.SNAKE_BRAINS); if (ids.length) return W.SNAKE_BRAINS[ids[0]]; }
  return W.SNAKE_MODEL;
}
function brainMask(env, net, kind) {
  const m = env.maskOf(kind);
  return net.spec.actionMap.map((ea) => m[ea]);
}
let out;

if (cmd.cmd === "random") {
  out = cmd.seeds.map((s) => {
    const r = new W.PyRandom(s);
    const floats = [];
    for (let i = 0; i < 5; i++) floats.push(r.random());
    const below = cmd.ns.map((n) => r.randbelow(n));
    return { seed: s, floats, below };
  });
} else if (cmd.cmd === "config") {
  out = W.APP_CONFIG;
} else if (cmd.cmd === "env") {
  out = cmd.runs.map((run) => {
    const env = new W.SnakeEnv(run.n, run.seed);
    const traj = [];
    let s = env.reset();
    traj.push({ state: arr(s), snake: env.snake.map((p) => [p[0], p[1]]), food: env.food ? [env.food[0], env.food[1]] : null, score: env.score, steps: env.steps,
                dir: env.direction, done: env.done, reward: null, reason: "" });
    for (const a of run.actions) {
      if (env.done) break;
      const r = env.step(a);
      s = env.state();
      traj.push({ state: arr(s), snake: env.snake.map((p) => [p[0], p[1]]), food: env.food ? [env.food[0], env.food[1]] : null, score: env.score, steps: env.steps,
                  dir: env.direction, done: env.done, reward: r.reward, reason: r.info.reason || "" });
    }
    return traj;
  });
} else if (cmd.cmd === "obs") {
  // observations + masks of every kind along a random walk: {n, seed, actions, kinds}
  const env = new W.SnakeEnv(cmd.n, cmd.seed);
  env.reset();
  out = [];
  const rec = () => { const o = {}; for (const k of cmd.kinds) o[k] = arr(env.obs(k)); o.safe = env.safeMask(); o.turn = env.legalMask(); out.push(o); };
  rec();
  for (const a of cmd.actions) { if (env.done) break; env.step(a); if (!env.done) rec(); }
} else if (cmd.cmd === "forward") {
  // {model, cases: [{obs, mask (brain order), knock: [[layer, idx], ...]}]}
  const net = new W.BrainNet(payload());
  out = cmd.cases.map((c) => {
    net.restoreAll();
    for (const [layer, idx] of c.knock || []) net.setKnock(layer, idx, true);
    const r = net.forward(Float32Array.from(c.obs), c.mask, true);
    const contrib = {};
    for (const k in r.contrib) contrib[k] = arr(r.contrib[k]);
    const acts = {};
    for (const k in r.acts) acts[k] = arr(r.acts[k].a);
    return { q: arr(r.q), value: r.value, action: r.action, probs: r.probs, h: arr(r.h), dx: arr(r.dx), saliency: arr(r.saliency),
             contrib, acts, params: net.params, spec: net.spec };
  });
} else if (cmd.cmd === "rollout") {
  // {model, games, seed, policy: "greedy"|"sample", guard: "trained"|"safe"|"turn"|"none", maxSteps}
  const net = new W.BrainNet(payload());
  const env = new W.SnakeEnv(net.n, cmd.seed);
  const rng = new W.PyRandom(cmd.seed);
  const maskKind = cmd.guard === "trained" || !cmd.guard ? net.spec.mask : cmd.guard;
  out = [];
  for (let g = 0; g < cmd.games; g++) {
    env.reset();
    let info = {};
    while (!env.done && env.steps < (cmd.maxSteps || 5000)) {
      const m = brainMask(env, net, maskKind);
      const r = net.forward(env.obs(net.spec.obs), m, false);
      const a = cmd.policy === "sample" ? W.BrainNet.sample(r.probs, rng.random()) : r.action;
      info = env.step(net.spec.actionMap[a]).info;
    }
    out.push({ score: env.score, steps: env.steps, length: env.snake.length, reason: info.reason || "达到步数上限" });
  }
} else {
  throw new Error("unknown cmd " + cmd.cmd);
}
process.stdout.write(JSON.stringify(out));
