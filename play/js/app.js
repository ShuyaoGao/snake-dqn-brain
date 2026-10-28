/* 贪吃蛇 · 看懂 AI 的大脑 — page logic (plain script, works from file://).
 * One mode: the AI plays, you watch its brain. Several brains can be switched (config.js `brains`);
 * the right column is computed live from the real weights (play/js/brain_net.js). */
(function () {
  "use strict";
  var CFG = window.APP_CONFIG, CREDITS = window.CREDITS || {}, RESULTS = window.BRAIN_RESULTS || {};
  var SnakeEnv = window.SnakeEnv, BrainNet = window.BrainNet;
  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);
  var ACT = CFG.actions;

  function store(key, val) {
    try {
      if (val === undefined) return localStorage.getItem(CFG.storagePrefix + key);
      localStorage.setItem(CFG.storagePrefix + key, val);
    } catch (e) { /* private window / blocked storage: just forget */ }
    return null;
  }

  // ------------------------------------------------------------------ brains
  function payloadOf(cfg) {
    if (cfg.global) return window[cfg.global] || null;
    return (window.SNAKE_BRAINS && window.SNAKE_BRAINS[cfg.id]) || null;
  }
  var BRAINS = CFG.brains.filter(payloadOf);
  if (!BRAINS.length) {
    document.body.innerHTML = "<div class='ended'>模型文件（play/model/*.js）没有加载成功。<br>请把整个 play 文件夹一起解压后再打开 index.html。</div>";
    return;
  }

  // ------------------------------------------------------------------ app state
  var app = {
    brain: null, net: null, spec: null, N: 12, NN: 144, geo: null, inputs: [], lastHidden: null,
    policy: "greedy", guard: false,
    alive: true, fog: true, autoNext: true, timers: [], view: null, dirty: false, sizes: {},
    kConv: CFG.topK.conv, touch: false, touchp: false
  };
  var W = { env: null, rng: null, playing: true, speedIdx: CFG.watchDefault, ms: CFG.watchSpeeds[CFG.watchDefault].ms,
            game: 0, results: {}, qCurve: [], over: null, lastRes: null, totalSteps: 0, timer: null, knockedThisGame: false };

  function later(fn, ms) {
    var id = setTimeout(function () {
      app.timers = app.timers.filter(function (t) { return t !== id; });
      if (app.alive) fn();
    }, ms);
    app.timers.push(id);
    return id;
  }
  function cancel(id) { if (id) { clearTimeout(id); app.timers = app.timers.filter(function (t) { return t !== id; }); } }

  function toast(msg) {
    var t = $("toast");
    t.textContent = msg; t.classList.add("show");
    cancel(t._timer); t._timer = later(function () { t.classList.remove("show"); }, 1400);
  }

  function brainById(id) { for (var i = 0; i < BRAINS.length; i++) if (BRAINS[i].id === id) return BRAINS[i]; return null; }
  function curResults() { return W.results[app.brain.id] || (W.results[app.brain.id] = []); }
  function invMap() { var inv = [0, 0, 0, 0]; app.spec.actionMap.forEach(function (ea, i) { inv[ea] = i; }); return inv; }
  function maskKind() { return app.guard ? "safe" : app.spec.mask; }
  function valueOf(res) {
    if (res.value !== null && res.value !== undefined) return res.value;
    var m = -Infinity;
    for (var i = 0; i < res.q.length; i++) if (res.mask[i] && res.q[i] > m) m = res.q[i];
    return m;
  }

  /** Switch brain: new network, new board size, new panel; the watch loop restarts on the same seed. */
  function setBrain(id) {
    var cfg = brainById(id) || BRAINS[0];
    app.brain = cfg;
    app.net = new BrainNet(payloadOf(cfg));
    app.spec = app.net.spec;
    app.N = app.spec.n; app.NN = app.N * app.N;
    app.geo = SnakeEnv.obsGeometry(app.spec.obs, app.N);
    app.inputs = CFG.inputs[app.spec.obs] || [];
    var hid = app.net.hiddenLayers();
    app.lastHidden = hid.length ? hid[hid.length - 1] : null;
    app.policy = cfg.policy || "greedy";
    app.guard = !!cfg.guard;
    store("brain", cfg.id);
    buildPanel();
    renderFormula();
    hideBanner();
    watchRestart();
    renderOpts(); renderControls(); renderExplain(); renderPills();
    layout();
    scheduleWatch();
  }

  /** Run the current brain on `env` (full = also gradients for the panel). */
  function analyze(env, full) {
    var em = env.maskOf(maskKind()), am = app.spec.actionMap, bm = am.map(function (ea) { return em[ea]; });
    var res = app.net.forward(env.obs(app.spec.obs), bm, full !== false);
    res.greedy = res.action;
    if (app.policy === "sample") res.action = BrainNet.sample(res.probs, W.rng.random());
    res.envAction = am[res.action];
    res.envMask = em;
    if (res.saliency) res.boardSal = SnakeEnv.toBoard(res.saliency, app.spec.obs, app.N);
    return res;
  }

  function envFromSnap(snap) {
    var e = new SnakeEnv(snap.n, 0);
    e.snake = snap.snake.map(function (p) { return [p[0], p[1]]; });
    e.food = snap.food ? [snap.food[0], snap.food[1]] : null;
    e.direction = snap.dir; e.score = snap.score; e.steps = snap.steps; e.done = snap.done;
    return e;
  }

  // ================================================================== WATCH loop
  function watchRestart() {
    W.env = new SnakeEnv(app.N, CFG.watchSeed);
    W.rng = new window.PyRandom(CFG.watchSeed);
    W.game = 0;
    W.results[app.brain.id] = [];
    watchNextGame();
  }

  function watchNextGame() {
    W.env.reset();
    W.game += 1;
    W.qCurve = [];
    W.over = null;
    W.knockedThisGame = app.net.knockCount() > 0;
    watchAnalyze(true);
  }

  function watchAnalyze(full) {
    W.lastRes = analyze(W.env, full);
    W.qCurve.push(valueOf(W.lastRes));
    if (W.qCurve.length > 600) W.qCurve.shift();
    setWatchView();
  }

  function setWatchView() {
    var res = W.lastRes, a = res.envAction, extra = "";
    if (app.policy === "sample" && res.action !== res.greedy) extra = "（按概率抽到的；最大概率是 " + ACT[app.spec.actionMap[res.greedy]].arrow + "）";
    if (a === SnakeEnv.OPPOSITE[W.env.direction]) extra = "（掉头无效，按直行）";
    app.view = {
      snap: W.env.snap(), res: res, curve: W.qCurve,
      label: W.over ? "<b>本局结束</b>：显示的是撞上之前那一步的局面" : "<b>此刻局面</b> → 打算往 <b>" + ACT[a].arrow + " " + ACT[a].name + "</b>" + extra,
      arrows: W.over ? [] : [{ from: W.env.snake[0], dir: a, color: CFG.colors.chosen, dashed: false }],
      fog: app.fog && res.boardSal ? res.boardSal : null
    };
    if (W.over) app.view.snap = W.over.snapBefore;
    markDirty();
  }

  function watchStep() {
    if (W.over) return;
    var env = W.env, a = W.lastRes.envAction;
    var before = env.snap();
    var peeked = env.peek(a);
    var r = env.step(a);
    W.totalSteps++;
    if (app.net.knockCount() > 0) W.knockedThisGame = true;
    if (r.done) {
      var reason = SnakeEnv.endReason(r.info, peeked, app.N);
      W.over = { reason: reason, score: env.score, steps: env.steps, snapBefore: before };
      curResults().push({ score: env.score, steps: env.steps, length: env.snake.length, reason: reason, knocked: W.knockedThisGame,
                          guard: app.guard, policy: app.policy });
      if (!W.lastRes.contrib) { var e2 = envFromSnap(before); W.lastRes = analyze(e2, true); }
      setWatchView();
      showWatchBanner();
      renderExplain();
      if (app.autoNext && W.playing) {
        later(function () { if (W.over) { hideBanner(); watchNextGame(); scheduleWatch(); } }, Math.min(1300, Math.max(250, W.ms * 4)));
      }
      return;
    }
    // 极速档：梯度只每 4 步算一次（右边面板隔几步刷新），其余步只做前向
    watchAnalyze(W.ms > 20 || W.totalSteps % 4 === 0);
  }

  function scheduleWatch() {
    cancel(W.timer);
    W.timer = null;
    if (!W.playing || !app.alive) return;
    W.timer = later(function () {
      W.timer = null;
      if (W.over) return;
      watchStep();
      scheduleWatch();
    }, W.ms);
  }

  function setWatchPlaying(on) {
    W.playing = on;
    if (on && W.over) { hideBanner(); watchNextGame(); }
    scheduleWatch();
    renderControls();
  }

  function manualStep() {
    if (W.playing) setWatchPlaying(false);
    if (W.over) { hideBanner(); watchNextGame(); }
    else watchStep();
  }

  function showWatchBanner() {
    var o = W.over, b = $("banner");
    b.innerHTML = "<div class='big'>本局结束：" + o.reason + "</div><div>吃到 " + o.score + " 个食物，走了 " + o.steps + " 步</div>" +
      (app.autoNext && W.playing ? "<div style='color:#aab'>马上开下一局…</div>" : "<div style='color:#aab'>按「播放」或空格开下一局</div>");
    b.hidden = false;
  }
  function hideBanner() { $("banner").hidden = true; }

  // ================================================================== LEFT: pills / options / controls / explain
  function renderPills() {
    var e = W.env, done = curResults(), sum = 0, h = [];
    done.forEach(function (r) { sum += r.score; });
    h.push("<span class='pill'>第 <b>" + W.game + "</b> 局</span>");
    h.push("<span class='pill'>得分 <b>" + e.score + "</b></span>");
    h.push("<span class='pill'>长度 <b>" + e.snake.length + "</b></span>");
    h.push("<span class='pill'>步数 <b>" + e.steps + "</b></span>");
    h.push("<span class='pill'>已看完 <b>" + done.length + "</b> 局，平均 <b>" + (done.length ? (sum / done.length).toFixed(2) : "–") + "</b> 分</span>");
    $("pills").innerHTML = h.join("");
  }

  function presetButtons(list, idx, kind) {
    return list.map(function (p, i) {
      return "<button class='preset" + (i === idx ? " on" : "") + "' data-kind='" + kind + "' data-i='" + i + "' data-tip='" + p.name + "：" + p.tip + "'>" + p.name + "</button>";
    }).join("");
  }

  function renderOpts() {
    var o = $("opts"), b = app.brain, trainedSafe = app.spec.mask === "safe";
    var h = "<div class='opt-row'><span class='lab'>大脑</span>" +
      BRAINS.map(function (c) {
        return "<button class='preset" + (c.id === b.id ? " on" : "") + "' data-kind='brain' data-id='" + c.id + "' data-tip='" + c.title + "（快捷键 " + c.key + "）：" + c.trained + "'>" + c.name + "</button>";
      }).join("") + "</div>" +
      "<div class='src'>来源：" + (b.url ? "<a href='" + b.url + "' target='_blank' rel='noopener'>" + b.source + "</a>" : b.source) + "（" + b.license + "）· " + b.algo + " · " + b.trained +
      " · 棋盘 " + app.N + "×" + app.N + " · 参数 " + app.net.params.toLocaleString("en-US") + " 个</div>" +
      "<div class='opt-row'><span class='lab'>速度</span>" + presetButtons(CFG.watchSpeeds, W.speedIdx, "speed") +
      "<input type='range' id='rng-ms' min='5' max='1000' step='5' value='" + W.ms + "' data-tip='细调每步间隔'>" +
      "<span class='ms' id='ms-lab'>" + W.ms + " 毫秒/步</span></div>" +
      "<div class='opt-row'>" +
      "<label class='ck' data-tip='红雾 = 网络做这次决定时最“在意”的格子：|∂(所选方向的分)/∂输入|，按格子汇总，颜色越深影响越大' data-anim='fog'><input type='checkbox' id='ck-fog'" + (app.fog ? " checked" : "") + "> 注意力红雾</label>" +
      "<label class='ck'><input type='checkbox' id='ck-auto'" + (app.autoNext ? " checked" : "") + "> 死了自动开下一局</label>" +
      (trainedSafe
        ? "<label class='ck' data-tip='这个模型训练时就带“不能撞死、不能掉头”的动作掩码，是它的一部分，不能关'><input type='checkbox' checked disabled> 护栏（训练自带）</label>"
        : "<label class='ck' data-tip='护栏 = 只在不会立刻撞墙、撞身的方向里选。这是我们加在模型外面的规则，不是模型学会的；关掉就是原模型本来的样子'><input type='checkbox' id='ck-guard'" + (app.guard ? " checked" : "") + "> 护栏（不撞死）</label>") +
      "<span class='lab' style='min-width:0'>选动作</span>" +
      "<button class='preset" + (app.policy === "greedy" ? " on" : "") + "' data-kind='policy' data-v='greedy' data-tip='每一步都走分最高的方向（确定性，同一种子每次一样）'>最大分</button>" +
      "<button class='preset" + (app.policy === "sample" ? " on" : "") + "' data-kind='policy' data-v='sample' data-tip='按概率抽签决定方向（PPO 训练时就是这么走的；用固定种子的随机数，可复现）'>按概率抽样</button>" +
      "</div>";
    o.innerHTML = h;
    bindTips(o);
  }

  function renderControls() {
    var c = $("controls");
    c.innerHTML = "<button class='btn " + (W.playing ? "" : "go") + "' id='b-play' data-tip='播放 / 暂停（空格）'>" + (W.playing ? "⏸ 暂停" : "▶ 播放") + "</button>" +
      "<button class='btn' id='b-step' data-tip='暂停并只走一步（→ 或 . 键），方便对照右边的大脑' data-anim='step'>单步</button>" +
      "<button class='btn' id='b-reset' data-tip='从头再来：回到和评估脚本一样的第 1 局（种子 " + CFG.watchSeed + "）'>重开</button>";
    bindTips(c);
  }

  function fmtNum(v, d) { return (v === null || v === undefined || isNaN(v)) ? "–" : (+v).toFixed(d === undefined ? 1 : d); }

  function renderExplain() {
    var b = app.brain, e = $("explain"), h = "";
    h += "<p><b>" + b.title + "：</b>" + b.blurb + "</p>";
    h += "<p><b>右边是它的大脑：</b>从输入到每一层的真实激活，再到 4 个方向的分，全部是真实权重在你的浏览器里实时算出来的；点任何方块或圆点可以把那个神经元敲掉。" +
      (app.spec.out === "q" ? "输出是 Q 值（“往这边走，以后大概还能拿到多少奖励”）。" : "输出是 4 个方向的打分（logit），换算成概率就是它每一步的“倾向”；旁边的“局面价值”来自它自己的价值网络。") + "</p>";
    // comparison table: this session + the recorded evaluation
    var rows = BRAINS.map(function (c) {
      var r = W.results[c.id] || [], sum = 0, mx = 0;
      r.forEach(function (g) { sum += g.score; if (g.score > mx) mx = g.score; });
      var ev = RESULTS[c.id] && RESULTS[c.id]["default"];
      return "<tr" + (c.id === b.id ? " class='cur'" : "") + "><td>" + c.name + "</td><td>" + (r.length ? fmtNum(sum / r.length, 2) + "（" + r.length + " 局，最高 " + mx + "）" : "–") + "</td>" +
        "<td>" + (ev ? fmtNum(ev.avg_score, 2) + "（最高 " + ev.max_score + "）" : "–") + "</td></tr>";
    }).join("");
    h += "<table class='hist'><tr><th>大脑</th><th>本页看过的局：平均吃到</th><th>固定种子评估 30 局：平均吃到</th></tr>" + rows + "</table>";
    h += "<p style='color:var(--muted)'>评估数字来自 results/brains.json（种子 " + CFG.watchSeed + "，每个大脑按它默认的选动作方式和护栏设置跑 30 局）；本页的开局顺序和评估一样，不敲神经元、不改设置时前 30 局就是那 30 局。</p>";
    e.innerHTML = h;
    bindTips(e);
  }

  // ================================================================== BOARD
  function drawBoard() {
    var v = app.view;
    if (!v) return;
    var cv = $("board"), s = app.sizes.board, dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(s * dpr)) { cv.width = Math.round(s * dpr); cv.height = Math.round(s * dpr); cv.style.width = s + "px"; cv.style.height = s + "px"; }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var snap = v.snap, n = snap.n, nn = n * n, cs = s / n;
    ctx.fillStyle = "#0b1016"; ctx.fillRect(0, 0, s, s);
    ctx.strokeStyle = CFG.colors.grid; ctx.lineWidth = 1;
    for (var i = 0; i <= n; i++) {
      ctx.beginPath(); ctx.moveTo(i * cs, 0); ctx.lineTo(i * cs, s); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cs); ctx.lineTo(s, i * cs); ctx.stroke();
    }
    if (v.fog) {
      var mx = 1e-8;
      for (i = 0; i < nn; i++) if (v.fog[i] > mx) mx = v.fog[i];
      for (i = 0; i < nn; i++) {
        var f = v.fog[i] / mx;
        if (f > 0.08) { ctx.fillStyle = "rgba(255,60,40," + (0.55 * f).toFixed(3) + ")"; ctx.fillRect((i % n) * cs, Math.floor(i / n) * cs, cs, cs); }
      }
    }
    var L = snap.snake.length;
    for (var k = L - 1; k >= 0; k--) {
      var p = snap.snake[k], val = k === 0 ? 1 : Math.max(0.15, 1 - k / L);
      if (k === 0) ctx.fillStyle = CFG.colors.head;
      else ctx.fillStyle = "rgb(" + (20 * val | 0) + "," + ((60 + 140 * val) | 0) + "," + ((40 + 60 * val) | 0) + ")";
      roundRect(ctx, p[1] * cs + 2, p[0] * cs + 2, cs - 4, cs - 4, Math.min(8, cs / 5));
      ctx.fill();
      if (k === 0) {
        ctx.fillStyle = "#10151c";
        var d = SnakeEnv.DIRS[snap.dir], cx = p[1] * cs + cs / 2, cy = p[0] * cs + cs / 2;
        var ox = d[1] * cs * 0.15, oy = d[0] * cs * 0.15, px = -d[0] * cs * 0.17, py = d[1] * cs * 0.17;
        dot(ctx, cx + ox + px, cy + oy + py, cs * 0.07); dot(ctx, cx + ox - px, cy + oy - py, cs * 0.07);
      }
    }
    if (snap.food) {
      ctx.fillStyle = CFG.colors.food;
      dot(ctx, snap.food[1] * cs + cs / 2, snap.food[0] * cs + cs / 2, cs * 0.3);
    }
    (v.arrows || []).forEach(function (a) { drawArrow(ctx, a.from, a.dir, cs, a.color, a.dashed); });
  }
  function dot(ctx, x, y, r) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function drawArrow(ctx, from, dir, cs, color, dashed) {
    var d = SnakeEnv.DIRS[dir], x0 = from[1] * cs + cs / 2, y0 = from[0] * cs + cs / 2;
    var x1 = x0 + d[1] * cs * 0.95, y1 = y0 + d[0] * cs * 0.95;
    ctx.save();
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = Math.max(2.5, cs * 0.09);
    if (dashed) ctx.setLineDash([cs * 0.12, cs * 0.1]);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.setLineDash([]);
    var ang = Math.atan2(y1 - y0, x1 - x0), hs = cs * 0.24;
    ctx.beginPath(); ctx.moveTo(x1 + Math.cos(ang) * hs * 0.4, y1 + Math.sin(ang) * hs * 0.4);
    ctx.lineTo(x1 - Math.cos(ang - 0.5) * hs, y1 - Math.sin(ang - 0.5) * hs);
    ctx.lineTo(x1 - Math.cos(ang + 0.5) * hs, y1 - Math.sin(ang + 0.5) * hs); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  // ================================================================== BRAIN PANEL
  function heat(v) {      // 0..1 -> black -> orange -> yellow
    v = Math.max(0, Math.min(1, v));
    return "rgb(" + (Math.min(255, 255 * v * 1.6) | 0) + "," + ((200 * Math.pow(v, 1.6)) | 0) + "," + ((50 * Math.pow(v, 3)) | 0) + ")";
  }
  function heatRGB(v, out, o) {
    v = Math.max(0, Math.min(1, v));
    out[o] = Math.min(255, 255 * v * 1.6) | 0; out[o + 1] = (200 * Math.pow(v, 1.6)) | 0; out[o + 2] = (50 * Math.pow(v, 3)) | 0; out[o + 3] = 255;
  }
  function divergeRGB(v, out, o) {   // -1..1 -> blue / black / amber
    v = Math.max(-1, Math.min(1, v));
    if (v >= 0) { out[o] = 255 * v | 0; out[o + 1] = 183 * v | 0; out[o + 2] = 77 * v | 0; }
    else { out[o] = 79 * -v | 0; out[o + 1] = 195 * -v | 0; out[o + 2] = 247 * -v | 0; }
    out[o + 3] = 255;
  }
  function sizeCanvas(cv, w, h) {
    var dpr = window.devicePixelRatio || 1;
    if (cv._w !== w || cv._h !== h || cv._d !== dpr) {
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.width = w + "px"; cv.style.height = h + "px";
      cv._w = w; cv._h = h; cv._d = dpr;
    }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }
  var OFF = document.createElement("canvas");
  /** Draw a w x h map (arr[off..], row-major) into a `size` px square; scale = value for full brightness. */
  function drawMap(cv, arr, off, w, h, size, scale, signed) {
    var ctx = sizeCanvas(cv, size, size);
    if (OFF.width !== w || OFF.height !== h) { OFF.width = w; OFF.height = h; }
    var octx = OFF.getContext("2d"), img = octx.createImageData(w, h), px = img.data;
    for (var i = 0; i < w * h; i++) {
      var v = arr[off + i] / scale;
      if (signed) divergeRGB(v, px, 4 * i); else heatRGB(v, px, 4 * i);
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(OFF, 0, 0, w, h, 0, 0, size, size);
  }
  function fmt(v, d) { return (v >= 0 ? "+" : "") + v.toFixed(d === undefined ? 2 : d); }

  function layerLabel(L, i) {
    if (L.kind === "conv") return "卷积层 " + i;
    return "全连接层 " + i;
  }

  function renderFormula() {
    var net = app.net, sp = app.spec, parts = ["输入 " + sp.input.c + "×" + sp.input.h + "×" + sp.input.w], ci = 0, fi = 0;
    net.layers.forEach(function (L, idx) {
      var last = idx === net.layers.length - 1;
      if (last) parts.push("输出 " + L.out + " 个方向的" + (sp.out === "q" ? "Q 值" : "分"));
      else if (L.kind === "conv") { ci++; parts.push("卷积 " + L.cout + "（" + L.k + "×" + L.k + "，步长 " + L.s + "）"); }
      else { fi++; parts.push("全连接 " + L.out); }
    });
    var fcW = 0;
    net.layers.forEach(function (L) { if (L.kind === "fc") fcW += net.w[L.name + ".weight"].length + net.w[L.name + ".bias"].length; });
    $("formula").innerHTML = parts.join("<span class='arrow'>→</span>") +
      (sp.v.length ? "<span class='arrow'>＋</span>价值网络 → 1 个数" : "") +
      "<span class='param'>参数 " + net.params.toLocaleString("en-US") + " 个（" + (100 * fcW / net.params).toFixed(1) + "% 在全连接层）· float16</span>";
  }

  /** Build the panel DOM for the current brain (input row + one block per hidden layer). */
  function buildPanel() {
    var net = app.net, body = $("brain-body"), h = "", geo = app.geo, hw = geo.h * geo.w;
    h += "<div class='layer' id='L-input'><div class='lt' id='lt-input'><b>输入层</b> <span class='cnt'>" + (geo.c * hw) + "</span> 维 = " +
      app.inputs.map(function (x) { return x.name + " " + hw; }).join(" + ") + "（每组一张 " + geo.h + "×" + geo.w + " 图）</div>" +
      "<div class='lrow' id='row-input'>" + app.inputs.map(function (gI, ch) {
        return "<div class='in-map' data-tip='" + gI.name + "（" + hw + " 维）：" + gI.desc + "'><canvas id='in-" + ch + "'></canvas><div class='nm'>" + gI.name +
          " <span class='dims'>" + hw + " 维</span></div><div class='dims'>归因 <span id='inv-" + ch + "'></span></div><div class='attr'><i id='ina-" + ch + "'></i></div></div>";
      }).join("") + "</div>" +
      "<div class='note'>归因 = 梯度 × 输入（对所选方向打分的一阶近似），<span style='color:var(--pos)'>橙</span>推高、<span style='color:var(--neg)'>蓝</span>压低。<b>这是模型内部的相关性解释，不代表因果。</b></div></div>";
    var ci = 0, fi = 0, hid = net.hiddenLayers();
    hid.forEach(function (L, idx) {
      var last = idx === hid.length - 1, n = L.name;
      if (L.kind === "conv") {
        ci++;
        h += "<div class='layer conv' id='L-" + n + "'><div class='lt' id='lt-" + n + "'><b>卷积层 " + ci + "</b> <span class='cnt'>" + L.cout + "</span> 个通道 × " + L.hout + "×" + L.wout + " 位置 = " +
          (L.cout * L.hout * L.wout).toLocaleString("en-US") + " 个神经元 · 展开贡献最大的 " + app.kConv + " 个通道 · 方阵 = 全部 " + L.cout + " 个（点击敲掉）</div>" +
          "<div class='lrow'><div class='tops' id='tops-" + n + "'>" + topSlots(n, L) + "</div><canvas class='sq' id='sq-" + n + "' data-layer='" + n + "' data-tip='全部通道：每个小方块是一个通道的平均激活。点一下 = 敲掉 / 恢复' data-anim='click'></canvas></div></div>";
      } else if (!last) {
        fi++;
        h += "<div class='layer' id='L-" + n + "'><div class='lt' id='lt-" + n + "'><b>全连接层 " + fi + "</b> <span class='cnt'>" + L.out + "</span> 个神经元 · 方阵 = 全部（点击敲掉）· 边框 = 对所选方向贡献最大的几个</div>" +
          "<canvas class='sq fcgrid' id='sq-" + n + "' data-layer='" + n + "' data-tip='每个小方块是一个神经元的激活。点一下 = 敲掉 / 恢复' data-anim='click'></canvas></div>";
      } else {
        fi++;
        h += "<div class='layer' id='L-fc'><div class='lt' id='lt-fc'><b>全连接层 " + fi + "</b> <span class='cnt'>" + L.out + "</span> 个神经元 · 方阵 = 全部（<span id='fc-active'></span>）· 圆点 = 贡献最大的至多 " + CFG.topK.fc +
          " 个 · 连线 <span style='color:var(--pos)'>橙</span>推高 / <span style='color:var(--neg)'>蓝</span>压低 → <b>输出 4 个方向</b>" +
          (app.spec.out === "q" ? "（% = 按 Q 换算的倾向，仅示意，DQN 总选 Q 最大的）" : "（% = 模型自己的概率）") + "</div>" +
          "<canvas id='fc-canvas' data-layer='" + n + "'></canvas></div>";
      }
    });
    body.innerHTML = h;
    bindTips(body);
    body.querySelectorAll(".lt").forEach(function (el) { el.title = el.textContent; });
  }
  function topSlots(layer, L) {
    var h = "";
    for (var i = 0; i < CFG.topK.conv; i++) h += "<div class='fm' data-layer='" + layer + "' data-slot='" + i + "' data-tip='这个通道在 " + L.hout + "×" + L.wout + " 的特征图上的激活。点一下 = 敲掉 / 恢复' data-anim='click'><canvas></canvas><div class='cap'></div></div>";
    return h;
  }

  var hits = { fc: [], sq: {} };

  function renderBrain() {
    var v = app.view;
    if (!v || !v.res || !v.res.contrib) return;
    var res = v.res, sz = app.sizes, net = app.net, geo = app.geo, hw = geo.h * geo.w;
    $("view-label").innerHTML = v.label || "";
    var kc = net.knockCount();
    $("knock-info").textContent = kc ? "已敲掉 " + kc + " 个" : "";
    $("btn-restore").disabled = kc === 0;

    // ---- input planes
    var x = envFromSnap(v.snap).obs(app.spec.obs);
    var ci = res.contrib.input, mxa = 1e-8;
    for (var c = 0; c < ci.length; c++) mxa = Math.max(mxa, Math.abs(ci[c]));
    app.inputs.forEach(function (gI, ch) {
      drawMap($("in-" + ch), x, ch * hw, geo.w, geo.h, sz.map, 1, !!gI.signed);
      var a = ci[ch] / mxa, bar = $("ina-" + ch);
      bar.style.left = (a >= 0 ? 50 : 50 + 50 * a) + "%";
      bar.style.width = (50 * Math.abs(a)) + "%";
      bar.style.background = a >= 0 ? CFG.colors.pos : CFG.colors.neg;
      $("inv-" + ch).textContent = fmt(ci[ch]);
    });

    // ---- hidden layers
    var hid = net.hiddenLayers();
    hid.forEach(function (L, idx) {
      var last = idx === hid.length - 1, name = L.name, a = res.acts[name].a, contrib = res.contrib[name], C = L.units;
      if (L.kind === "conv") renderConv(L, a, contrib);
      else if (!last) renderFCGrid(L, a, contrib);
      else renderFC(L, res);
    });
    renderValue(v);
  }

  function topOrder(contrib, k) {
    var order = [];
    for (var i = 0; i < contrib.length; i++) order.push(i);
    order.sort(function (p, q) { return Math.abs(contrib[q]) - Math.abs(contrib[p]); });
    return order.filter(function (c) { return contrib[c] !== 0; }).slice(0, k);
  }

  function renderConv(L, a, contrib) {
    var name = L.name, C = L.cout, mhw = L.hout * L.wout, sz = app.sizes, net = app.net;
    var tops = topOrder(contrib, app.kConv);
    var slots = $("tops-" + name).children;
    for (var sl = 0; sl < slots.length; sl++) {
      slots[sl].style.visibility = sl < tops.length ? "visible" : "hidden";
      slots[sl].style.display = sl < app.kConv || sl < tops.length ? "" : "none";
      if (sl >= tops.length) delete slots[sl].dataset.ch;
    }
    tops.forEach(function (ch, s) {
      var el = slots[s], cv = el.querySelector("canvas"), mx = 1e-8;
      for (var p = 0; p < mhw; p++) mx = Math.max(mx, a[ch * mhw + p]);
      drawMap(cv, a, ch * mhw, L.wout, L.hout, sz.map, mx, false);
      var cc = contrib[ch], off = net.isKnocked(name, ch);
      el.className = "fm " + (off ? "off" : (cc >= 0 ? "pos" : "neg"));
      el.dataset.ch = ch;
      var cap = el.querySelector(".cap");
      cap.className = "cap " + (cc >= 0 ? "pos" : "neg");
      cap.textContent = "#" + ch + " " + fmt(cc);
    });
    // full-layer square: mean activation per channel
    var cols = 8, rows = Math.ceil(C / cols), t = Math.max(5, Math.min(16, Math.floor((sz.map + 18) / rows)));
    var cv2 = $("sq-" + name), ctx = sizeCanvas(cv2, cols * t, rows * t), means = new Float32Array(C), mm = 1e-8;
    for (var i = 0; i < C; i++) { var s2 = 0; for (var p2 = 0; p2 < mhw; p2++) s2 += a[i * mhw + p2]; means[i] = s2 / mhw; mm = Math.max(mm, means[i]); }
    ctx.clearRect(0, 0, cols * t, rows * t);
    for (i = 0; i < C; i++) {
      var cx = (i % cols) * t, cy = Math.floor(i / cols) * t;
      if (net.isKnocked(name, i)) {
        ctx.fillStyle = "#333"; ctx.fillRect(cx + 0.5, cy + 0.5, t - 1, t - 1);
        ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(cx + 2, cy + 2); ctx.lineTo(cx + t - 2, cy + t - 2);
        ctx.moveTo(cx + t - 2, cy + 2); ctx.lineTo(cx + 2, cy + t - 2); ctx.stroke();
      } else { ctx.fillStyle = heat(means[i] / mm); ctx.fillRect(cx + 0.5, cy + 0.5, t - 1, t - 1); }
      if (tops.indexOf(i) >= 0) { ctx.strokeStyle = contrib[i] >= 0 ? CFG.colors.pos : CFG.colors.neg; ctx.lineWidth = 1.5; ctx.strokeRect(cx + 1, cy + 1, t - 2, t - 2); }
    }
    hits.sq[name] = { cols: cols, t: t, C: C };
  }

  function renderFCGrid(L, h, contrib) {
    var name = L.name, H = L.out, net = app.net, side = Math.ceil(Math.sqrt(H));
    var t = Math.max(6, Math.min(14, Math.floor((app.sizes.map + 40) / side)));
    var cv = $("sq-" + name), ctx = sizeCanvas(cv, side * t, side * t), mh = 1e-8, i;
    for (i = 0; i < H; i++) mh = Math.max(mh, h[i]);
    var tops = topOrder(contrib, Math.min(CFG.topK.fc, 12));
    ctx.clearRect(0, 0, side * t, side * t);
    for (i = 0; i < H; i++) {
      var cx = (i % side) * t, cy = Math.floor(i / side) * t;
      if (net.isKnocked(name, i)) { ctx.fillStyle = "#444"; ctx.fillRect(cx + 0.5, cy + 0.5, t - 1, t - 1); ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx + 1, cy + 1); ctx.lineTo(cx + t - 1, cy + t - 1); ctx.stroke(); }
      else { ctx.fillStyle = heat(h[i] / mh); ctx.fillRect(cx + 0.5, cy + 0.5, t - 1, t - 1); }
      if (tops.indexOf(i) >= 0) { ctx.strokeStyle = contrib[i] >= 0 ? CFG.colors.pos : CFG.colors.neg; ctx.lineWidth = 1.2; ctx.strokeRect(cx + 1, cy + 1, t - 2, t - 2); }
    }
    hits.sq[name] = { cols: side, t: t, C: H };
  }

  function renderFC(L, res) {
    var cv = $("fc-canvas"), W0 = app.sizes.brainW, H0 = app.sizes.fcH, ctx = sizeCanvas(cv, W0, H0), net = app.net;
    ctx.clearRect(0, 0, W0, H0);
    var name = L.name, H = L.out, h = res.acts[name].a, contrib = res.contrib[name], fcOut = res.fcOut;
    var inv = invMap(), act = res.action;
    hits.fc = [];
    // left: grid of all neurons (activation)
    var side = Math.ceil(Math.sqrt(H)), g = Math.min(H0 - 4, Math.floor(W0 * 0.3)), cs = g / side, mh = 1e-8, i;
    for (i = 0; i < H; i++) mh = Math.max(mh, h[i]);
    var active = 0;
    for (i = 0; i < H; i++) if (h[i] > 0) active++;
    var tops = topOrder(contrib, CFG.topK.fc), K = tops.length, top1 = tops[0];
    for (i = 0; i < H; i++) {
      var gx = (i % side) * cs, gy = Math.floor(i / side) * cs + 2;
      if (net.isKnocked(name, i)) { ctx.fillStyle = "#444"; ctx.fillRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1); ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(gx + 1, gy + 1); ctx.lineTo(gx + cs - 1, gy + cs - 1); ctx.stroke(); }
      else { ctx.fillStyle = heat(h[i] / mh); ctx.fillRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1); }
      if (tops.indexOf(i) >= 0) { ctx.strokeStyle = contrib[i] >= 0 ? CFG.colors.pos : CFG.colors.neg; ctx.lineWidth = 1; ctx.strokeRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1); }
      hits.fc.push({ x: gx, y: gy, w: cs, h: cs, j: i, top1: i === top1 });
    }
    // middle: top-K dots in 2 columns
    var compact = W0 < 560;
    var colN = Math.max(1, Math.ceil(K / 2)), rowH = (H0 - 8) / colN, r = Math.max(4, Math.min(rowH * 0.36, 9));
    var dx0 = g + (compact ? 22 : 26), dx1 = dx0 + (compact ? 40 : 54), outX = compact ? dx1 + 30 : Math.max(dx1 + 110, W0 - 230), dots = [];
    var mc = 1e-8;
    tops.forEach(function (j) { mc = Math.max(mc, Math.abs(contrib[j])); });
    tops.forEach(function (j, k) {
      var col = k < colN ? 0 : 1, row = k % colN;
      dots.push({ j: j, x: col ? dx1 : dx0, y: 4 + rowH * (row + 0.5) });
    });
    var rowsY = [], oh = (H0 - 8) / 4;
    for (var o = 0; o < 4; o++) rowsY.push(4 + oh * (o + 0.5));
    // lines: strongest |h_j * W[o,j]| among the shown dots (o = env order row, fcOut indexed by brain order)
    var lines = [];
    dots.forEach(function (d) { for (var o2 = 0; o2 < 4; o2++) lines.push({ d: d, o: o2, v: fcOut[inv[o2]][d.j] }); });
    lines.sort(function (p, q) { return Math.abs(q.v) - Math.abs(p.v); });
    var ml = lines.length ? Math.abs(lines[0].v) || 1e-8 : 1;
    lines.slice(0, CFG.topK.lines).forEach(function (Ln) {
      ctx.strokeStyle = (Ln.v >= 0 ? "rgba(255,183,77," : "rgba(79,195,247,") + (0.25 + 0.6 * Math.abs(Ln.v) / ml).toFixed(2) + ")";
      ctx.lineWidth = 0.6 + 3 * Math.abs(Ln.v) / ml;
      ctx.beginPath(); ctx.moveTo(Ln.d.x + r, Ln.d.y);
      ctx.bezierCurveTo((Ln.d.x + outX) / 2, Ln.d.y, (Ln.d.x + outX) / 2, rowsY[Ln.o], outX - 4, rowsY[Ln.o]); ctx.stroke();
    });
    dots.forEach(function (d, k) {
      var off = net.isKnocked(name, d.j), cc = contrib[d.j];
      ctx.fillStyle = off ? "#444" : heat(h[d.j] / mh);
      ctx.beginPath(); ctx.arc(d.x, d.y, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = off ? "#ff6b6b" : (cc >= 0 ? CFG.colors.pos : CFG.colors.neg);
      ctx.lineWidth = 1 + 2 * Math.abs(cc) / mc; ctx.stroke();
      ctx.fillStyle = "#8b98a8"; ctx.font = "10px Consolas, Menlo, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      if (k < colN) ctx.fillText("#" + d.j, d.x - r - 3, d.y);
      else { ctx.textAlign = "left"; ctx.fillText("#" + d.j, d.x + r + 3, d.y); }
      hits.fc.push({ x: d.x - r - 2, y: d.y - r - 2, w: 2 * r + 4, h: 2 * r + 4, j: d.j, top1: d.j === top1 });
    });
    // outputs (rows in env order: 上 下 左 右)
    var q = res.q, mask = res.mask, prob = res.probs, qhi = -Infinity, qlo = Infinity;
    for (o = 0; o < 4; o++) { qhi = Math.max(qhi, q[o]); qlo = Math.min(qlo, q[o]); }
    if (app.spec.out === "q") qlo = Math.min(0, qlo);
    if (qhi - qlo < 1e-6) qhi = qlo + 1;
    var barX = outX + (compact ? 24 : 40), barW = Math.max(40, W0 - barX - (compact ? 70 : 96));
    ctx.textBaseline = "middle";
    var opp = SnakeEnv.OPPOSITE[app.view.snap.dir];
    for (o = 0; o < 4; o++) {
      var bi = inv[o], y = rowsY[o], chosen = bi === act, il = !mask[bi];
      ctx.globalAlpha = il ? 0.35 : 1;
      ctx.fillStyle = chosen ? CFG.colors.chosen : "#dde5ee"; ctx.font = "bold 13px 'Microsoft YaHei','PingFang SC',sans-serif"; ctx.textAlign = "left";
      ctx.fillText(compact ? ACT[o].arrow : ACT[o].arrow + " " + ACT[o].name, outX, y);
      ctx.fillStyle = "#0b1016"; ctx.fillRect(barX, y - 9, barW, 18);
      var wq = barW * Math.max(0.03, (q[bi] - qlo) / (qhi - qlo));
      ctx.fillStyle = chosen ? CFG.colors.chosen : "#355d83"; ctx.fillRect(barX, y - 9, wq, 18);
      ctx.fillStyle = "#dde5ee"; ctx.font = (compact ? "11px" : "12px") + " Consolas, Menlo, monospace";
      ctx.fillText((app.spec.out === "q" ? "Q " : "分 ") + q[bi].toFixed(compact ? 2 : 3), barX + barW + 5, y - 6);
      ctx.fillStyle = "#8b98a8"; ctx.font = (compact ? "10.5px" : "11px") + " 'Microsoft YaHei','PingFang SC',sans-serif";
      var why = il ? (o === opp ? (compact ? "掉头" : "掉头·不允许") : (compact ? "护栏" : "护栏·会撞死")) : (chosen ? "✓ " : "") + Math.round(prob[bi] * 100) + "%";
      ctx.fillText(why, barX + barW + 5, y + 8);
      ctx.globalAlpha = 1;
    }
    app._fcTop1 = top1;
    $("fc-active").textContent = "此刻有 " + active + " 个神经元激活（> 0），其余为 0";
  }

  function renderValue(v) {
    var res = v.res, val = valueOf(res);
    $("value-text").innerHTML = (app.spec.v.length ? "局面价值（价值网络）<b>" : "局面价值 <b>") + val.toFixed(2) + "</b>";
    var cv = $("value-curve"), w = 110, hgt = 22;
    var ctx = sizeCanvas(cv, w, hgt), arr = v.curve || [];
    ctx.clearRect(0, 0, w, hgt);
    if (arr.length < 2) return;
    var lo = Math.min.apply(null, arr), hi = Math.max.apply(null, arr);
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    ctx.strokeStyle = CFG.colors.chosen; ctx.lineWidth = 1.5; ctx.beginPath();
    arr.forEach(function (q, i) { var px = i / (arr.length - 1) * (w - 2) + 1, py = hgt - 3 - (q - lo) / (hi - lo) * (hgt - 6); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
    ctx.stroke();
  }

  // ================================================================== knock-out
  function toggleKnock(layer, idx) {
    var off = !app.net.isKnocked(layer, idx);
    app.net.setKnock(layer, idx, off);
    reanalyze();
    var L = app.net.layer(layer);
    toast((off ? "已敲掉 " : "已恢复 ") + (L && L.kind === "conv" ? "卷积层通道 #" : "全连接层神经元 #") + idx);
  }
  function fcTop1(res) {
    var c = res.contrib[app.lastHidden.name], best = 0;
    for (var j = 1; j < c.length; j++) if (Math.abs(c[j]) > Math.abs(c[best])) best = j;
    return best;
  }
  function reanalyze() {
    if (!W.over) { W.lastRes = analyze(W.env, true); }
    else { W.lastRes = analyze(envFromSnap(W.over.snapBefore), true); }
    setWatchView();
  }
  function restoreAll() { app.net.restoreAll(); reanalyze(); toast("所有神经元已恢复"); }

  // ================================================================== layout / render loop
  /** 触屏 / 手机竖屏判断（v2 第八节：触屏 + 宽度 ≤ 980px + 竖屏；768×1024 的平板也算）。 */
  function detectTouch() {
    var touch = false;
    try { touch = (navigator.maxTouchPoints > 0) || window.matchMedia("(pointer: coarse)").matches; } catch (e) { touch = false; }
    var w = window.innerWidth, h = window.innerHeight, portrait = h >= w;
    var before = app.touchp;
    app.touch = touch;
    app.touchp = touch && w <= 980 && portrait;
    document.body.classList.toggle("touch", app.touch);
    document.body.classList.toggle("touchp", app.touchp);
    document.documentElement.classList.toggle("touchp", app.touchp);
    app.kConv = app.touchp && w < 600 ? 4 : CFG.topK.conv;
    if (!app.touchp) { $("menu").hidden = true; }
    if (before !== app.touchp && app.net) { renderOpts(); renderControls(); }
    return app.touchp;
  }

  function layoutMobile() {
    var left = $("left"), right = $("right"), inner = left.clientWidth - 16, N = app.N;
    var b = Math.min(inner - 2, 640);
    app.sizes.board = Math.max(N * 10, Math.floor(b / N) * N);
    var rw = right.clientWidth - 18;
    if (rw < 120) rw = window.innerWidth - 16 - 18;
    app.sizes.brainW = Math.max(240, rw);
    var per = Math.max(3, app.inputs.length);
    app.sizes.map = Math.max(28, Math.floor(Math.min(64, (inner - 44) / per) / 2) * 2);
    app.sizes.fcH = inner < 600 ? 250 : 300;
    if (!app.view || !app.alive) { markDirty(); return; }
    drawBoard(); renderBrain(); renderPills();
  }

  function layout() {
    if (app.touchp) { layoutMobile(); return; }
    var main = $("main"), left = $("left"), right = $("right"), N = app.N;
    var lw = left.clientWidth, mh = main.clientHeight;
    var reserve = mh < 700 ? 300 : 340;
    var b = Math.max(220, Math.min(lw - 24, mh - reserve, 680));
    app.sizes.board = Math.floor(b / N) * N;
    var rw = right.clientWidth - 22, rh = right.clientHeight;
    app.sizes.brainW = Math.max(300, rw);
    var nConv = app.net.hiddenLayers().filter(function (L) { return L.kind === "conv"; }).length;
    var rowsN = 1 + nConv + Math.max(0, app.net.hiddenLayers().length - nConv - 1);
    var s = (rh - 268 - 130) / Math.max(2, rowsN);
    var byW = (rw - 120) / (CFG.topK.conv + 2.2) - 8;
    s = Math.max(30, Math.min(s, byW, 86));
    app.sizes.map = Math.floor(s / 2) * 2;
    app.sizes.fcH = Math.max(120, Math.min(320, Math.round(rh - rowsN * app.sizes.map - 268)));
    fit();
  }

  /** Render now and shrink until neither column needs a vertical scrollbar (1366x768 etc.). */
  function fit() {
    if (!app.view || !app.alive) { markDirty(); return; }
    var left = $("left"), right = $("right"), sz = app.sizes;
    for (var i = 0; i < 14; i++) {
      drawBoard(); renderBrain(); renderPills();
      var ro = right.scrollHeight - right.clientHeight, lo = left.scrollHeight - left.clientHeight;
      if (ro <= 1 && lo <= 1) break;
      if (lo > 1 && sz.board > 200) sz.board = Math.max(200, sz.board - Math.max(10, Math.ceil(lo / 10) * 10));
      if (ro > 1) {
        if (sz.fcH > 150) sz.fcH = Math.max(150, sz.fcH - ro);
        else if (sz.map > 28) { sz.map -= 4; }
        else if (sz.fcH > 110) sz.fcH -= 10;
        else break;
      }
    }
  }

  function markDirty() {
    if (app.dirty || !app.alive) return;
    app.dirty = true;
    requestAnimationFrame(function () {
      app.dirty = false;
      if (!app.alive) return;
      drawBoard();
      renderBrain();
      renderPills();
    });
  }

  // ================================================================== tooltips with animated demos
  var DEMOS = {
    click: "<div class='demo demo-click'><i class='d-dot'></i><i class='d-dot'></i><i class='d-dot target'></i><i class='d-cursor'></i></div>",
    step: "<div class='demo demo-step'><i class='bar b1'></i><i class='bar b2'></i><i class='bar b3'></i></div>",
    fog: "<div class='demo demo-fog'><i class='cell' style='left:60px;top:20px'></i><i class='cell' style='left:78px;top:20px;animation-delay:.3s'></i><i class='cell' style='left:96px;top:20px;animation-delay:.6s'></i></div>"
  };
  function bindTips(root) {
    (root || document).querySelectorAll("[data-tip]").forEach(function (el) {
      if (el._tip) return;
      el._tip = true;
      el.addEventListener("mouseenter", function () { showTip(el); });
      el.addEventListener("mouseleave", hideTip);
    });
  }
  function showTip(el, text) {
    if (app.touch) return;   // 触屏上没有悬停，不弹说明气泡
    var tip = $("tip");
    tip.innerHTML = (text || el.dataset.tip) + (el.dataset && el.dataset.anim && DEMOS[el.dataset.anim] ? DEMOS[el.dataset.anim] : "");
    tip.hidden = false;
    var r = el.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
    var x = Math.min(window.innerWidth - tw - 8, Math.max(8, r.left + r.width / 2 - tw / 2));
    var y = r.bottom + 8;
    if (y + th > window.innerHeight - 8) y = r.top - th - 8;
    tip.style.left = x + "px"; tip.style.top = Math.max(8, y) + "px";
  }
  function showTipAt(x, y, html) {
    if (app.touch) return;
    var tip = $("tip");
    tip.innerHTML = html; tip.hidden = false;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - tw - 8, x + 14) + "px";
    tip.style.top = Math.min(window.innerHeight - th - 8, y + 14) + "px";
  }
  function hideTip() { $("tip").hidden = true; }

  // ================================================================== help / exit / big screen
  function openHelp() { $("help").hidden = false; }
  function closeHelp() { $("help").hidden = true; store("helpSeen", "1"); }

  function setBig(on) {
    if (app.touchp) return;   // 手机竖屏没有左右两栏
    document.body.classList.toggle("big", on);
    $("ck-big").checked = on;
    store("big", on ? "1" : "0");
    layout();
  }

  var listeners = [];
  function on(target, type, fn, opts) { target.addEventListener(type, fn, opts); listeners.push([target, type, fn, opts]); }

  function shutdown() {
    app.alive = false;
    app.timers.forEach(function (t) { clearTimeout(t); });
    app.timers = [];
    listeners.forEach(function (l) { l[0].removeEventListener(l[1], l[2], l[3]); });
    listeners = [];
    W.env = null; app.view = null; app.net = null;
    window.SNAKE_MODEL = null; window.SNAKE_BRAINS = null;
    try { window.close(); } catch (e) { /* ignore */ }
    setTimeout(function () {
      document.body.innerHTML = "<div class='ended'>已停止运行，所有计算都已结束，可以直接关闭这个标签页。</div>";
      document.title = "已停止 · " + CFG.shortTitle;
    }, 120);
  }

  // ================================================================== events
  function onKey(e) {
    if (e.target && /input|textarea|select/i.test(e.target.tagName) && e.target.type !== "checkbox" && e.target.type !== "range") return;
    var k = e.key;
    if (!$("help").hidden) { if (k === "Escape" || k === "Enter" || k === "h" || k === "H") { closeHelp(); e.preventDefault(); } return; }
    if (k === "h" || k === "H" || k === "?") { openHelp(); return; }
    if (k === "b" || k === "B") { setBig(!document.body.classList.contains("big")); return; }
    for (var i = 0; i < BRAINS.length; i++) if (k === BRAINS[i].key) { setBrain(BRAINS[i].id); return; }
    if (k === " ") { e.preventDefault(); setWatchPlaying(!W.playing); }
    else if (k === "ArrowRight" || k === ".") { e.preventDefault(); manualStep(); }
    else if (k === "r" || k === "R") { hideBanner(); watchRestart(); renderExplain(); }
  }

  function onClick(e) {
    var t = e.target;
    if (!t.closest("#menu") && t.id !== "menu-btn") { $("menu").hidden = true; $("menu-btn").setAttribute("aria-expanded", "false"); }
    var pre = t.closest(".preset");
    if (pre) {
      if (pre.dataset.kind === "brain") { if (pre.dataset.id !== app.brain.id) setBrain(pre.dataset.id); }
      else if (pre.dataset.kind === "speed") { var i = +pre.dataset.i; W.speedIdx = i; W.ms = CFG.watchSpeeds[i].ms; renderOpts(); scheduleWatch(); }
      else if (pre.dataset.kind === "policy") { app.policy = pre.dataset.v; renderOpts(); reanalyze(); }
      return;
    }
    var fm = t.closest(".fm");
    if (fm && fm.dataset.ch !== undefined) { toggleKnock(fm.dataset.layer, +fm.dataset.ch); return; }
    if (t.classList && t.classList.contains("sq")) { var ii = sqHit(e, t.dataset.layer); if (ii !== null) toggleKnock(t.dataset.layer, ii); return; }
    if (t.id === "fc-canvas") { var hh = canvasHit(e, hits.fc); if (hh) toggleKnock(t.dataset.layer, hh.j); return; }
    switch (t.id) {
      case "b-play": setWatchPlaying(!W.playing); break;
      case "b-step": manualStep(); break;
      case "b-reset": hideBanner(); watchRestart(); renderExplain(); scheduleWatch(); break;
      case "btn-restore": restoreAll(); break;
      case "btn-help": openHelp(); break;
      case "help-close": closeHelp(); break;
      case "btn-exit": shutdown(); break;
      case "menu-btn": var mn = $("menu"); mn.hidden = !mn.hidden; t.setAttribute("aria-expanded", mn.hidden ? "false" : "true"); break;
      case "menu-help": $("menu").hidden = true; openHelp(); break;
      case "menu-restore": $("menu").hidden = true; restoreAll(); break;
      case "menu-exit": $("menu").hidden = true; shutdown(); break;
    }
  }

  function onChange(e) {
    var t = e.target;
    if (t.id === "ck-fog") { app.fog = t.checked; setWatchView(); }
    else if (t.id === "ck-auto") { app.autoNext = t.checked; }
    else if (t.id === "ck-guard") { app.guard = t.checked; reanalyze(); }
    else if (t.id === "ck-big") { setBig(t.checked); }
  }
  function onInput(e) {
    var t = e.target;
    if (t.id === "rng-ms") {
      W.ms = +t.value; W.speedIdx = -1; $("ms-lab").textContent = W.ms + " 毫秒/步";
      document.querySelectorAll(".preset[data-kind=speed]").forEach(function (b) { b.classList.remove("on"); });
      scheduleWatch();
    }
  }

  function canvasHit(e, list) {
    var r = e.target.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    for (var i = list.length - 1; i >= 0; i--) { var h = list[i]; if (x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h) return h; }
    return null;
  }
  function sqHit(e, layer) {
    var info = hits.sq[layer]; if (!info) return null;
    var r = e.target.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    var i = Math.floor(y / info.t) * info.cols + Math.floor(x / info.t);
    return i >= 0 && i < info.C ? i : null;
  }

  function onBrainMove(e) {
    var t = e.target;
    if (!app.view || !app.view.res || !app.view.res.contrib) return;
    var res = app.view.res, envName = ACT[res.envAction].name;
    if (t.id === "fc-canvas") {
      var h = canvasHit(e, hits.fc);
      if (!h) { hideTip(); t.style.cursor = "default"; return; }
      t.style.cursor = "pointer";
      var ln = t.dataset.layer;
      showTipAt(e.clientX, e.clientY, "全连接层神经元 <b>#" + h.j + "</b>" + (app.net.isKnocked(ln, h.j) ? "（已敲掉）" : "") +
        "<br>激活 " + res.acts[ln].a[h.j].toFixed(3) + "<br>对“" + envName + "”的贡献 " + fmt(res.contrib[ln][h.j], 3) + "<br>点一下 = 敲掉 / 恢复" + DEMOS.click);
    } else if (t.classList && t.classList.contains("sq")) {
      var layer = t.dataset.layer, i = sqHit(e, layer);
      if (i === null) return;
      var L = app.net.layer(layer);
      showTipAt(e.clientX, e.clientY, (L.kind === "conv" ? "卷积层通道" : "全连接层神经元") + " <b>#" + i + "</b>" + (app.net.isKnocked(layer, i) ? "（已敲掉）" : "") +
        "<br>对“" + envName + "”的贡献 " + fmt(res.contrib[layer][i], 3) + "<br>点一下 = 敲掉 / 恢复");
    }
  }

  function renderFooter() {
    var c = CREDITS;
    $("foot").textContent = [(c.AUTHOR || "Shuyao Gao") + (c.AUTHOR_EMAIL ? "（" + c.AUTHOR_EMAIL + "）" : ""), c.UNIVERSITY_EN, c.UNIVERSITY_KO,
      (c.SERIES_NAME ? c.SERIES_NAME + "系列" : ""), "代码 MIT · 文档 CC BY 4.0 · 第三方模型见 THIRD_PARTY.md", "离线纯静态网页，关掉标签页即全部结束"]
      .filter(Boolean).join("  ·  ");
  }

  // ================================================================== self-test (used by tools/browser_check.py)
  function wait(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  async function selftest() {
    var out = { steps: [], brains: {}, ok: false };
    function note(s) { out.steps.push(s); }
    try {
      closeHelp(); openHelp(); closeHelp(); note("help");
      for (var bi = 0; bi < BRAINS.length; bi++) {
        var id = BRAINS[bi].id;
        setBrain(id); W.speedIdx = 3; W.ms = 5; renderOpts(); setWatchPlaying(true);
        await wait(1500);
        out.brains[id] = { steps: W.totalSteps, n: app.N, games: curResults().length };
        if (W.totalSteps < 10) throw new Error("brain " + id + " did not advance");
        setWatchPlaying(false); manualStep(); manualStep(); manualStep();
        var hid = app.net.hiddenLayers();
        toggleKnock(hid[0].name, 0); toggleKnock(app.lastHidden.name, fcTop1(W.lastRes));
        if (app.net.knockCount() !== 2) throw new Error("knock count " + id);
        manualStep(); restoreAll();
        if (!app.spec.mask || app.spec.mask !== "safe") { app.guard = !app.guard; reanalyze(); app.guard = !app.guard; reanalyze(); }
        app.policy = app.policy === "greedy" ? "sample" : "greedy"; reanalyze(); app.policy = BRAINS[bi].policy; reanalyze();
        W.totalSteps = 0;
        note("brain " + id + " ok");
      }
      setBig(true); await wait(50); setBig(false); note("big");
      setBrain(BRAINS[CFG.brainDefault].id);
      await wait(100);
      out.overflowX = document.documentElement.scrollWidth > window.innerWidth + 1;
      out.board = app.sizes.board; out.map = app.sizes.map;
      out.viewport = [window.innerWidth, window.innerHeight];
      var lr = $("left").getBoundingClientRect(), rr = $("right").getBoundingClientRect();
      out.columns = [Math.round(lr.width), Math.round(rr.width)];
      out.ok = window.__errors.length === 0 && !out.overflowX;
    } catch (err) {
      window.__errors.push("selftest: " + (err && err.stack || err));
    }
    out.errors = window.__errors.slice();
    var pre = $("selftest");
    pre.textContent = JSON.stringify(out);
    document.title = "SELFTEST " + (out.ok && !out.errors.length ? "PASS" : "FAIL");
  }

  /** ?shot=watch|knock&brain=<id>&steps=N — deterministic state for screenshots. */
  function prepareShot(kind, steps) {
    closeHelp();
    if (params.get("brain")) setBrain(params.get("brain"));
    setWatchPlaying(false);
    for (var j = 0; j < steps; j++) { if (W.over) { hideBanner(); watchNextGame(); } watchStep(); }
    if (!W.lastRes.contrib) reanalyze();
    if (kind === "knock") { toggleKnock(app.lastHidden.name, fcTop1(W.lastRes)); var hid = app.net.hiddenLayers(); toggleKnock(hid[Math.min(2, hid.length - 1)].name, 5); }
    $("toast").classList.remove("show");
    $("selftest").textContent = JSON.stringify({ sizes: app.sizes, rh: $("right").clientHeight, rw: $("right").clientWidth, mh: $("main").clientHeight });
  }

  // ================================================================== boot
  function boot() {
    $("app-title").textContent = CFG.title;
    $("app-title-short").textContent = CFG.mobileTitle || CFG.shortTitle;
    document.title = CFG.title;
    renderFooter();
    bindTips(document);
    on(document, "keydown", onKey);
    on(document, "click", onClick);
    on(document, "change", onChange);
    on(document, "input", onInput);
    on($("brain-body"), "mousemove", onBrainMove);
    on($("brain-body"), "mouseleave", hideTip);
    on(window, "resize", function () { detectTouch(); layout(); });
    on(window, "orientationchange", function () { setTimeout(function () { if (app.alive) { detectTouch(); layout(); } }, 150); });
    var big = store("big") === "1";
    if (big) { document.body.classList.add("big"); $("ck-big").checked = true; }
    detectTouch();
    var want = params.get("brain") || store("brain");
    setBrain(brainById(want) ? want : BRAINS[CFG.brainDefault].id);
    // 只读的调试 / 验收入口（tools/browser_check.py 的手机场景断言用）
    window.__app = {
      brain: function () { return app.brain.id; }, n: function () { return app.N; }, touchp: function () { return app.touchp; },
      watchSteps: function () { return W.totalSteps; }, knocked: function () { return app.net ? app.net.knockCount() : 0; },
      sizes: function () { return app.sizes; }, kConv: function () { return app.kConv; }, playing: function () { return W.playing; },
      guard: function () { return app.guard; }, policy: function () { return app.policy; }, brains: function () { return BRAINS.map(function (b) { return b.id; }); }
    };
    var shot = params.get("shot");
    if (params.get("selftest")) { closeHelp(); selftest(); return; }
    if (params.get("exittest")) { closeHelp(); setTimeout(function () { $("btn-exit").click(); }, 300); return; }
    if (shot) { prepareShot(shot, +(params.get("steps") || 30)); if (params.get("big") === "1") setBig(true); return; }
    if (params.get("nohelp") !== "1" && store("helpSeen") !== "1") openHelp();
    scheduleWatch();
  }

  boot();
})();
