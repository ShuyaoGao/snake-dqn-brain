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
  const ctx = { console, Math, Float32Array, Uint32Array, Set, Buffer, atob };
  ctx.window = ctx;
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  for (const n of names) vm.runInContext(fs.readFileSync(path.join(PLAY, n), "utf8"), ctx, { filename: n });
  return ctx;
}

const cmd = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const needModel = ["forward", "greedy"].includes(cmd.cmd);
const files = ["js/config.js", "js/pyrandom.js", "js/snake_env.js", "js/brain_net.js"];
if (needModel) files.push("model/snake_cnn_fp16.js");
const W = load(files);
const arr = (a) => Array.from(a);
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
} else if (cmd.cmd === "forward") {
  const net = new W.BrainNet(W.SNAKE_MODEL);
  out = cmd.cases.map((c) => {
    net.restoreAll();
    for (const [layer, idx] of c.knock || []) net.setKnock(layer, idx, true);
    const r = net.forward(Float32Array.from(c.state), c.legal, true);
    return { q: arr(r.q), action: r.action, h: arr(r.h), dx: arr(r.dx), saliency: arr(r.saliency),
             contrib: { input: arr(r.contrib.input), conv1: arr(r.contrib.conv1), conv2: arr(r.contrib.conv2),
                        conv3: arr(r.contrib.conv3), fc1: arr(r.contrib.fc1) },
             a1: arr(r.a1), params: net.params };
  });
} else if (cmd.cmd === "greedy") {
  const net = new W.BrainNet(W.SNAKE_MODEL);
  const env = new W.SnakeEnv(cmd.n, cmd.seed);
  out = [];
  for (let g = 0; g < cmd.games; g++) {
    let s = env.reset();
    let info = {};
    while (!env.done) {
      const legal = env.legalMask();
      const r = net.forward(s, legal, false);
      info = env.step(r.action).info;
      s = env.state();
    }
    out.push({ score: env.score, steps: env.steps, reason: info.reason || "" });
  }
} else {
  throw new Error("unknown cmd " + cmd.cmd);
}
process.stdout.write(JSON.stringify(out));
