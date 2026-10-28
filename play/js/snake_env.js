/* Line-by-line port of snake_env.py (SnakeEnv) for the browser, plus the observation builders of
 * the other brains (brains.py OBS) and their action masks.
 *
 * Same grid, state encoding, rewards, 180-degree rule, hunger timeout and — via PyRandom — the
 * same food positions for the same seed. Locked by tests/test_play_parity.py.
 *
 * State (5, N, N) for the DQN ("dqn5"), flattened channel-major into a Float32Array:
 *   ch0 head (=1)  ch1 body brightness (next to head brightest, min 0.15)  ch2 food (=1)
 *   ch3 current direction dr plane  ch4 current direction dc plane
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

  /** 180-degree turn forbidden (the DQN's training mask). Env action order. */
  SnakeEnv.prototype.legalMask = function () {
    var m = [1, 1, 1, 1];
    m[OPPOSITE[this.direction]] = 0;
    return m;
  };

  /** linyiLYi/snake-ai _check_action_validity: no 180-degree turn, no wall, no body (the tail cell is
   * free unless this step eats the food). Falls back to legalMask() when every move is fatal. */
  SnakeEnv.prototype.safeMask = function () {
    var out = [0, 0, 0, 0], any = false, hr = this.snake[0][0], hc = this.snake[0][1];
    for (var a = 0; a < 4; a++) {
      if (a === OPPOSITE[this.direction]) continue;
      var r = hr + DIRS[a][0], c = hc + DIRS[a][1];
      var eats = this.food && r === this.food[0] && c === this.food[1];
      var lim = eats ? this.snake.length : this.snake.length - 1, hit = false;
      for (var i = 0; i < lim; i++) if (this.snake[i][0] === r && this.snake[i][1] === c) { hit = true; break; }
      if (hit || r < 0 || r >= this.n || c < 0 || c >= this.n) continue;
      out[a] = 1; any = true;
    }
    return any ? out : this.legalMask();
  };

  SnakeEnv.prototype.maskOf = function (kind) {
    if (kind === "safe") return this.safeMask();
    if (kind === "turn") return this.legalMask();
    return [1, 1, 1, 1];
  };

  // ---------------------------------------------------------------- observations of the other brains
  /** np.linspace(start, stop, L, dtype=np.uint8): float64 arithmetic, last element = stop, truncation. */
  SnakeEnv.grayU8 = function (start, stop, L) {
    var out = new Array(L);
    if (L === 1) { out[0] = Math.trunc(start); return out; }
    var step = (stop - start) / (L - 1);
    for (var i = 0; i < L; i++) out[i] = Math.trunc(i * step + start);
    out[L - 1] = Math.trunc(stop);
    return out;
  };
  /** np.linspace(start, stop, L, dtype=np.float32). */
  SnakeEnv.linspaceF32 = function (start, stop, L) {
    var out = new Array(L);
    if (L === 1) { out[0] = Math.fround(start); return out; }
    var step = (stop - start) / (L - 1);
    for (var i = 0; i < L; i++) out[i] = Math.fround(i * step + start);
    out[L - 1] = Math.fround(stop);
    return out;
  };

  /** 12x12 cells -> 3 x (n*scale) x (n*scale) image, channel-major, /255. `transposed`: image index [col][row]
   * (Part-time-Ray/AI-Snake-Game indexes its picture as [x][y]). */
  SnakeEnv.prototype._cnnObs = function (scale, transposed, tailRed) {
    var n = this.n, W = n * scale, HW = W * W, x = new Float32Array(3 * HW);
    var L = this.snake.length, gray = SnakeEnv.grayU8(200, 50, L), i;
    var cells = [];                                             // [r, c, R, G, B] in drawing order (later wins)
    for (i = 0; i < L; i++) cells.push([this.snake[i][0], this.snake[i][1], gray[i], gray[i], gray[i]]);
    cells.push([this.snake[0][0], this.snake[0][1], 0, 255, 0]);
    if (tailRed) cells.push([this.snake[L - 1][0], this.snake[L - 1][1], 255, 0, 0]);
    if (this.food) cells.push([this.food[0], this.food[1], 0, 0, 255]);
    for (i = 0; i < cells.length; i++) {
      var ce = cells[i], y0 = (transposed ? ce[1] : ce[0]) * scale, x0 = (transposed ? ce[0] : ce[1]) * scale;
      for (var ch = 0; ch < 3; ch++) {
        var v = ce[2 + ch] / 255;
        for (var dy = 0; dy < scale; dy++) {
          var row = ch * HW + (y0 + dy) * W + x0;
          for (var dx = 0; dx < scale; dx++) x[row + dx] = v;
        }
      }
    }
    return x;
  };

  SnakeEnv.prototype.obsLinyiCnn = function () { return this._cnnObs(7, false, true); };
  SnakeEnv.prototype.obsRayCnn = function () { return this._cnnObs(5, true, false); };
  SnakeEnv.prototype.obsLinyiMlp = function () {
    var n = this.n, x = new Float32Array(n * n), L = this.snake.length, v = SnakeEnv.linspaceF32(0.8, 0.2, L);
    for (var i = 0; i < L; i++) x[this.snake[i][0] * n + this.snake[i][1]] = v[i];
    x[this.snake[0][0] * n + this.snake[0][1]] = 1.0;
    if (this.food) x[this.food[0] * n + this.food[1]] = -1.0;
    return x;
  };

  /** Observation by kind (brains.py OBS). */
  SnakeEnv.prototype.obs = function (kind) {
    if (kind === "dqn5") return this.state();
    if (kind === "linyi_cnn") return this.obsLinyiCnn();
    if (kind === "ray_cnn") return this.obsRayCnn();
    if (kind === "linyi_mlp") return this.obsLinyiMlp();
    throw new Error("unknown obs kind " + kind);
  };

  /** Geometry of an observation kind: image size, pixels per cell, transposed? (for saliency -> board cells). */
  SnakeEnv.obsGeometry = function (kind, n) {
    if (kind === "linyi_cnn") return { c: 3, h: 7 * n, w: 7 * n, scale: 7, transposed: false };
    if (kind === "ray_cnn") return { c: 3, h: 5 * n, w: 5 * n, scale: 5, transposed: true };
    if (kind === "linyi_mlp") return { c: 1, h: n, w: n, scale: 1, transposed: false };
    return { c: CHANNELS, h: n, w: n, scale: 1, transposed: false };
  };

  /** Sum a per-pixel map (h*w of the observation) into n*n board cells. */
  SnakeEnv.toBoard = function (map, kind, n) {
    var geo = SnakeEnv.obsGeometry(kind, n), out = new Float32Array(n * n), s = geo.scale;
    for (var y = 0; y < geo.h; y++) for (var x = 0; x < geo.w; x++) {
      var a = Math.floor(y / s), b = Math.floor(x / s);
      var r = geo.transposed ? b : a, c = geo.transposed ? a : b;
      out[r * n + c] += map[y * geo.w + x];
    }
    return out;
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
