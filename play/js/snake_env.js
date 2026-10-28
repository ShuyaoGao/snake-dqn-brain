/* Line-by-line port of snake_env.py (SnakeEnv) for the browser.
 *
 * Same grid, state encoding, rewards, 180-degree rule, hunger timeout and — via
 * PyRandom — the same food positions for the same seed. Locked by
 * tests/test_play_parity.py (trajectories compared step by step with Python).
 *
 * State (5, N, N), flattened channel-major into a Float32Array:
 *   ch0 head (=1)
 *   ch1 body brightness: next to head brightest, fading to the tail (min 0.15)
 *   ch2 food (=1)
 *   ch3 current direction dr plane (whole board filled with -1/0/1)
 *   ch4 current direction dc plane
 * Actions: 0=up 1=down 2=left 3=right.
 */
(function (g) {
  "use strict";
  var DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  var OPPOSITE = [1, 0, 3, 2];
  var CHANNELS = 5;

  function SnakeEnv(n, seed) {
    this.n = n || 12;
    this.seed = seed;
    this.rng = new g.PyRandom(seed);
  }

  SnakeEnv.DIRS = DIRS;
  SnakeEnv.OPPOSITE = OPPOSITE;
  SnakeEnv.CHANNELS = CHANNELS;

  SnakeEnv.prototype.reset = function () {
    var n = this.n, r = Math.floor(n / 2), c = Math.floor(n / 2);
    this.snake = [[r, c], [r, c - 1], [r, c - 2]];
    this.direction = 3;
    this.score = 0;
    this.steps = 0;
    this.hunger = 0;
    this._placeFood();
    this.done = false;
    return this.state();
  };

  SnakeEnv.prototype._occupied = function () {
    var s = new Set(), n = this.n;
    for (var i = 0; i < this.snake.length; i++) s.add(this.snake[i][0] * n + this.snake[i][1]);
    return s;
  };

  SnakeEnv.prototype._placeFood = function () {
    var occ = this._occupied(), empty = [], n = this.n;
    for (var r = 0; r < n; r++)
      for (var c = 0; c < n; c++)
        if (!occ.has(r * n + c)) empty.push([r, c]);
    this.food = empty.length ? this.rng.choice(empty) : null;
  };

  SnakeEnv.prototype.state = function () {
    var n = this.n, nn = n * n, s = new Float32Array(CHANNELS * nn);
    var L = this.snake.length, h = this.snake[0];
    s[h[0] * n + h[1]] = 1.0;
    for (var i = 1; i < L; i++) {
      var p = this.snake[i];
      s[nn + p[0] * n + p[1]] = Math.max(0.15, 1.0 - i / L);
    }
    if (this.food) s[2 * nn + this.food[0] * n + this.food[1]] = 1.0;
    var d = DIRS[this.direction];
    for (var k = 0; k < nn; k++) { s[3 * nn + k] = d[0]; s[4 * nn + k] = d[1]; }
    return s;
  };

  SnakeEnv.prototype.legalMask = function () {
    var m = [1, 1, 1, 1];
    m[OPPOSITE[this.direction]] = 0;
    return m;
  };

  SnakeEnv.prototype.step = function (action) {
    if (this.done) throw new Error("step() after done");
    if (action === OPPOSITE[this.direction]) action = this.direction;
    this.direction = action;
    var d = DIRS[action], hr = this.snake[0][0], hc = this.snake[0][1];
    var nr = hr + d[0], nc = hc + d[1];
    this.steps += 1;
    this.hunger += 1;

    var reward = -0.01;
    if (this.food) {
      var oldD = Math.abs(hr - this.food[0]) + Math.abs(hc - this.food[1]);
      var newD = Math.abs(nr - this.food[0]) + Math.abs(nc - this.food[1]);
      reward += newD < oldD ? 0.03 : -0.03;
    }
    var hitBody = false;
    for (var i = 0; i < this.snake.length - 1; i++)
      if (this.snake[i][0] === nr && this.snake[i][1] === nc) { hitBody = true; break; }
    if (!(nr >= 0 && nr < this.n && nc >= 0 && nc < this.n) || hitBody) {
      this.done = true;
      return { reward: -1.0, done: true, info: { reason: "撞墙/撞身" } };
    }
    this.snake.unshift([nr, nc]);
    if (this.food && nr === this.food[0] && nc === this.food[1]) {
      this.score += 1;
      this.hunger = 0;
      reward = 1.0;
      this._placeFood();
      if (this.food === null) {
        this.done = true;
        return { reward: 5.0, done: true, info: { reason: "通关！" } };
      }
    } else {
      this.snake.pop();
    }
    if (this.hunger > 2 * this.n * this.n) {
      this.done = true;
      return { reward: -0.5, done: true, info: { reason: "饥饿超时" } };
    }
    return { reward: reward, done: false, info: {} };
  };

  /** Predict the outcome of `action` without changing the env (port of vote_core.peek). */
  SnakeEnv.prototype.peek = function (action) {
    if (action === OPPOSITE[this.direction]) action = this.direction;
    var d = DIRS[action], nr = this.snake[0][0] + d[0], nc = this.snake[0][1] + d[1];
    if (!(nr >= 0 && nr < this.n && nc >= 0 && nc < this.n)) return "撞墙";
    for (var i = 0; i < this.snake.length - 1; i++)
      if (this.snake[i][0] === nr && this.snake[i][1] === nc) return "撞到自己";
    if (this.food && nr === this.food[0] && nc === this.food[1]) return "吃到食物";
    return "安全";
  };

  /** Plain-object snapshot (for replay / history). */
  SnakeEnv.prototype.snap = function () {
    return { n: this.n, snake: this.snake.map(function (p) { return [p[0], p[1]]; }),
             food: this.food ? [this.food[0], this.food[1]] : null, dir: this.direction,
             score: this.score, steps: this.steps, done: this.done };
  };

  /** Port of vote_core.end_reason: friendlier death reason. */
  SnakeEnv.endReason = function (info, peeked, n) {
    var r = (info && info.reason) || "";
    if (r === "撞墙/撞身") return (peeked === "撞墙" || peeked === "撞到自己") ? peeked : "撞墙/撞身";
    if (r === "饥饿超时") return "饿死（" + (2 * n * n) + " 步没吃到）";
    if (r.indexOf("通关") === 0) return "通关";
    return r || "结束";
  };

  g.SnakeEnv = SnakeEnv;
})(typeof window !== "undefined" ? window : globalThis);
