/* 贪吃蛇 · 看懂 AI 的大脑 — page logic (plain script, works from file://).
 * Modes: 观战 (watch, default) / 我来玩 (play) / 练习 (practice).
 * Right column: brain panel computed live from the real weights (play/js/brain_net.js). */
(function () {
  "use strict";
  var CFG = window.APP_CONFIG, CREDITS = window.CREDITS || {};
  var SnakeEnv = window.SnakeEnv, BrainNet = window.BrainNet;
  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);

  function store(key, val) {
    try {
      if (val === undefined) return localStorage.getItem(CFG.storagePrefix + key);
      localStorage.setItem(CFG.storagePrefix + key, val);
    } catch (e) { /* private window / blocked storage: just forget */ }
    return null;
  }

  // ------------------------------------------------------------------ model
  if (!window.SNAKE_MODEL) {
    document.body.innerHTML = "<div class='ended'>模型文件 model/snake_cnn_fp16.js 没有加载成功。<br>请把整个 play 文件夹一起解压后再打开 index.html。</div>";
    return;
  }
  var net = new BrainNet(window.SNAKE_MODEL);
  var META = window.SNAKE_MODEL.meta;
  var N = net.n, NN = N * N;
  var SIZES = net.layerSizes();
  var ACT = CFG.actions;
  var OPP = SnakeEnv.OPPOSITE;

  // ------------------------------------------------------------------ app state
  var app = {
    mode: "watch",
    alive: true,
    fog: true,
    autoNext: true,
    timers: [],
    view: null,        // {snap, res, label, arrows, fog}
    dirty: false,
    sizes: {},
    kConv: CFG.topK.conv,   // 每个卷积层展开的通道数（手机紧凑画法下更少）
    touch: false,           // 触屏设备（粗指针）
    touchp: false,          // 触屏 + 宽度 ≤ 980 + 竖屏 → 手机竖屏布局
    touchl: false,          // 触屏 + 宽度 ≤ 980 + 横屏 → 提示竖屏
    sheet: false,           // 「看大脑」全屏面板是否打开（手机「我来玩」）
    touchCtl: null
  };

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
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast.h);
    toast.h = setTimeout(function () { t.classList.remove("show"); }, 2200);
  }

  function maxLegal(q, legal) {
    var m = -Infinity;
    for (var i = 0; i < 4; i++) if (legal[i] && q[i] > m) m = q[i];
    return m;
  }

  function envFromSnap(s) {
    var e = new SnakeEnv(s.n, 0);
    e.snake = s.snake.map(function (p) { return [p[0], p[1]]; });
    e.food = s.food ? [s.food[0], s.food[1]] : null;
    e.direction = s.dir;
    e.score = s.score; e.steps = s.steps; e.done = false;
    return e;
  }

  function analyze(env) {
    return net.forward(env.state(), env.legalMask(), true);
  }

  // ================================================================== WATCH engine (also used by 练习)
  var W = {
    env: null, game: 0, results: [], playing: true, timer: null, speedIdx: CFG.watchDefault,
    ms: CFG.watchSpeeds[CFG.watchDefault].ms, qCurve: [], over: null, knockedThisGame: false,
    lastRes: null, totalSteps: 0
  };

  function watchRestart() {
    W.env = new SnakeEnv(N, CFG.watchSeed);
    W.game = 0;
    W.results = [];
    watchNextGame();
  }

  function watchNextGame() {
    W.env.reset();
    W.game += 1;
    W.qCurve = [];
    W.over = null;
    W.knockedThisGame = net.knockCount() > 0;
    watchAnalyze();
  }

  function watchAnalyze() {
    W.lastRes = analyze(W.env);
    W.qCurve.push(maxLegal(W.lastRes.q, W.lastRes.legal));
    if (W.qCurve.length > 600) W.qCurve.shift();
    if (app.mode !== "play") setWatchView();
  }

  function setWatchView() {
    var res = W.lastRes;
    app.view = {
      snap: W.env.snap(), res: res, curve: W.qCurve,
      label: W.over ? "<b>本局结束</b>：显示的是撞上之前那一步的局面" : "<b>此刻局面</b> → 打算往 <b>" + ACT[res.action].arrow + " " + ACT[res.action].name + "</b>",
      arrows: W.over ? [] : [{ from: W.env.snake[0], dir: res.action, color: CFG.colors.chosen, dashed: false }],
      fog: app.fog ? res.saliency : null
    };
    if (W.over) app.view.snap = W.over.snapBefore;
    markDirty();
  }

  function watchStep() {
    if (W.over) return;
    var env = W.env, a = W.lastRes.action;
    var before = env.snap();
    var peeked = env.peek(a);
    var r = env.step(a);
    W.totalSteps++;
    if (net.knockCount() > 0) W.knockedThisGame = true;
    if (r.done) {
      var reason = SnakeEnv.endReason(r.info, peeked, N);
      W.over = { reason: reason, score: env.score, steps: env.steps, snapBefore: before };
      W.results.push({ score: env.score, steps: env.steps, reason: reason, knocked: W.knockedThisGame });
      setWatchView();
      showWatchBanner();
      practiceEvent("watch-over");
      if (app.autoNext && W.playing) {
        later(function () { if (W.over && app.mode !== "play") { hideBanner(); watchNextGame(); scheduleWatch(); } },
              Math.min(1300, Math.max(250, W.ms * 4)));
      }
      return;
    }
    watchAnalyze();
  }

  function scheduleWatch() {
    cancel(W.timer);
    W.timer = null;
    if (!W.playing || app.mode === "play" || !app.alive) return;
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

  function showWatchBanner() {
    var o = W.over, b = $("banner");
    b.innerHTML = "<div class='big'>本局结束：" + o.reason + "</div><div>吃到 " + o.score + " 个食物，走了 " + o.steps + " 步</div>" +
      (app.autoNext && W.playing ? "<div style='color:#aab'>马上开下一局…</div>" : "<div style='color:#aab'>按「播放」或空格开下一局</div>");
    b.hidden = false;
  }
  function hideBanner() { $("banner").hidden = true; }

  // ================================================================== PLAY engine (我来玩)
  var P = {
    episode: 0, env: null, seed: 0, levelIdx: CFG.playDefault, ms: CFG.playLevels[CFG.playDefault].ms,
    assist: false, started: false, paused: false, pending: null, timer: null, log: [], history: [],
    cur: null, prev: null, prevSnap: null, over: null, dqn: null, replay: null, qCurve: []
  };

  function playNewGame() {
    cancel(P.timer);
    P.episode += 1;
    P.seed = CFG.playSeed0 + P.episode;
    P.env = new SnakeEnv(N, P.seed);
    P.env.reset();
    P.log = []; P.started = false; P.paused = false; P.pending = null; P.over = null;
    P.prev = null; P.prevSnap = null; P.replay = null; P.qCurve = [];
    P.dqn = dqnRollout(P.seed);
    P.assistUsed = P.assist;
    P.knocked = net.knockCount() > 0;
    P.cur = analyze(P.env);
    P.qCurve.push(maxLegal(P.cur.q, P.cur.legal));
    hideBanner();
    setPlayView();
    renderExplain();
  }

  /** The DQN plays the same seed on its own (greedy), like the classroom version. */
  function dqnRollout(seed) {
    var env = new SnakeEnv(N, seed), info = {}, peeked = "";
    env.reset();
    while (!env.done && env.steps < 5000) {
      var r = net.forward(env.state(), env.legalMask(), false);
      peeked = env.peek(r.action);
      info = env.step(r.action).info;
    }
    return { score: env.score, steps: env.steps, reason: env.done ? SnakeEnv.endReason(info, peeked, N) : "达到步数上限" };
  }

  function playKey(dir) {
    if (app.mode !== "play" || P.over) return;
    if (P.replay) { P.replay = null; setPlayView(); renderExplain(); }
    if (P.ms === 0) { playStep(dir); return; }
    P.pending = dir;
    if (!P.started || P.paused) { P.started = true; P.paused = false; schedulePlay(); renderControls(); }
  }

  function schedulePlay() {
    cancel(P.timer);
    P.timer = null;
    if (app.mode !== "play" || P.ms === 0 || !P.started || P.paused || P.over || P.replay) return;
    P.timer = later(function () {
      P.timer = null;
      var d = P.pending === null ? P.env.direction : P.pending;
      P.pending = null;
      playStep(d);
      schedulePlay();
    }, P.ms);
  }

  function playStep(dir) {
    if (P.over) return;
    P.started = true;
    $("board-hint").hidden = true;
    var env = P.env, res = P.cur;
    var you = dir === OPP[env.direction] ? env.direction : dir;   // 180-degree turn = keep going (same rule as training)
    var rec = {
      step: env.steps + 1, you: you, dqn: res.action, agree: you === res.action,
      outcome: env.peek(you), dqnOutcome: env.peek(res.action), snap: env.snap(),
      q: Array.from(res.q), legal: res.legal.slice()
    };
    var r = env.step(you);
    P.log.push(rec);
    P.prev = res; P.prevSnap = rec.snap;
    if (P.assist) P.assistUsed = true;
    if (net.knockCount() > 0) P.knocked = true;
    if (r.done) {
      playFinish(SnakeEnv.endReason(r.info, rec.outcome, N), rec);
      return;
    }
    P.cur = analyze(env);
    P.qCurve.push(maxLegal(P.cur.q, P.cur.legal));
    setPlayView();
    renderPills();
  }

  function playFinish(reason, fatal) {
    cancel(P.timer);
    var env = P.env, agree = 0;
    P.log.forEach(function (r) { if (r.agree) agree++; });
    P.over = {
      ep: P.episode, seed: P.seed, you: { score: env.score, steps: env.steps, reason: reason },
      dqn: P.dqn, agree: P.log.length ? agree / P.log.length : null, disagree: P.log.length - agree,
      fatal: fatal, assist: P.assistUsed, knocked: P.knocked
    };
    P.history.push(P.over);
    var f = fatal, b = $("banner");
    b.innerHTML = "<div class='big'>本局结束：" + reason + "</div>" +
      "<div>你：吃到 <b>" + env.score + "</b> 个，走了 " + env.steps + " 步</div>" +
      "<div>致命一步：你走了 <span class='hl-you'>" + ACT[f.you].arrow + " " + ACT[f.you].name + "</span>，" +
      "DQN 会走 <span class='hl-dqn'>" + ACT[f.dqn].arrow + " " + ACT[f.dqn].name + "</span> → " + f.dqnOutcome + "</div>" +
      "<div>同一开局 DQN 自己玩：吃到 <b>" + P.dqn.score + "</b> 个，" + P.dqn.steps + " 步（" + P.dqn.reason + "）</div>" +
      (app.touchp ? "<div style='color:#aab'>点「下一局」继续（横屏或电脑上可以回放你和 DQN 意见不同的时刻）</div>"
                  : "<div style='color:#aab'>按 Enter 或「下一局」继续；左下角可以回放你和 DQN 意见不同的时刻</div>");
    b.hidden = false;
    setPlayView();
    renderPills();
    renderExplain();
    renderControls();
    practiceEvent("play-over");
  }

  function setPlayView() {
    if (app.mode !== "play") return;
    var v;
    if (P.replay) {
      var rec = P.replay, e = envFromSnap(rec.snap), res = net.forward(e.state(), e.legalMask(), true);
      v = { snap: rec.snap, res: res, curve: P.qCurve,
            label: "<b>回放第 " + rec.step + " 步</b>：你走 " + ACT[rec.you].arrow + "，DQN 会走 " + ACT[rec.dqn].arrow + "（结果：你 → " + rec.outcome + "；DQN → " + rec.dqnOutcome + "）",
            arrows: [{ from: rec.snap.snake[0], dir: rec.you, color: CFG.colors.you, dashed: false },
                     { from: rec.snap.snake[0], dir: rec.dqn, color: CFG.colors.dqn, dashed: true }],
            fog: app.fog ? res.saliency : null };
    } else if (P.assist && !P.over) {
      v = { snap: P.env.snap(), res: P.cur, curve: P.qCurve,
            label: "<b>此刻局面（辅助已开）</b>：DQN 建议往 <b>" + ACT[P.cur.action].arrow + " " + ACT[P.cur.action].name + "</b>",
            arrows: [{ from: P.env.snake[0], dir: P.cur.action, color: CFG.colors.dqn, dashed: true }],
            fog: app.fog ? P.cur.saliency : null };
    } else if (P.prev) {
      var last = P.log[P.log.length - 1];
      v = { snap: P.over ? P.prevSnap : P.env.snap(), res: P.prev, curve: P.qCurve, inputSnap: P.prevSnap,
            label: "<b>揭晓上一步</b>：那个局面里 DQN 会走 <b>" + ACT[last.dqn].arrow + "</b>，你走了 <b>" + ACT[last.you].arrow + "</b>" + (last.agree ? "（一致 ✓）" : "（不一致）"),
            arrows: [{ from: P.prevSnap.snake[0], dir: last.you, color: CFG.colors.you, dashed: false },
                     { from: P.prevSnap.snake[0], dir: last.dqn, color: CFG.colors.dqn, dashed: true }],
            fog: null };
    } else {
      v = { snap: P.env.snap(), res: P.cur, curve: P.qCurve, cover: true,
            label: "<b>等你走第一步</b>", arrows: [], fog: null };
    }
    app.view = v;
    var hint = $("board-hint");
    if (!P.started && !P.over && !P.replay) {
      if (app.touchp) hint.textContent = P.ms === 0 ? "拖下方摇杆（或在棋盘上滑动）走第一步" : "拖摇杆开始（不动就一直直走）";
      else hint.textContent = P.ms === 0 ? "按方向键（或 WASD / 在棋盘上滑动）走第一步" : "按任意方向键开始（不按键就一直直走）";
      hint.hidden = false;
    } else hint.hidden = true;
    markDirty();
  }

  // ================================================================== PRACTICE (task list)
  var TASKS = [
    { id: "step3", text: "按 3 次「单步」（或 → 键）。每一步都对照右下角：<b>Q 值最高的绿色方向</b>就是蛇下一步走的方向。", anim: "step" },
    { id: "slowfog", text: "选「慢放讲解」档并打开「注意力红雾」，看红雾怎么跟着蛇头和食物移动。", anim: "fog" },
    { id: "knockconv", text: "在任意一个卷积层右侧的方阵里点一下，<b>敲掉</b>一个通道。", anim: "click" },
    { id: "knockfc", text: "在全连接层敲掉<b>贡献最大的那个神经元</b>（圆点列表最上面那个），看 4 个 Q 值怎么变。", anim: "click" },
    { id: "restore", text: "点右上角「全部恢复」，让大脑恢复原样。", anim: "click" },
    { id: "playover", text: "切到「我来玩」，用回合制走完一局，看看你和 DQN 谁吃得多。", anim: "keys" },
    { id: "replay", text: "一局结束后，在左下角点开一个「你 ≠ DQN」的时刻回放，想想 DQN 为什么那样走。", anim: "click" }
  ];
  var PR = { done: {}, stepCount: 0 };
  (function loadTasks() {
    var s = store("tasks");
    if (s) { try { PR.done = JSON.parse(s) || {}; } catch (e) { PR.done = {}; } }
  })();

  function practiceEvent(ev, data) {
    var before = Object.keys(PR.done).length;
    if (ev === "manual-step") { PR.stepCount++; if (PR.stepCount >= 3) PR.done.step3 = 1; }
    if ((ev === "speed" || ev === "fog") && W.speedIdx === 0 && app.fog) PR.done.slowfog = 1;
    if (ev === "knock" && data && /^conv/.test(data.layer) && data.off) PR.done.knockconv = 1;
    if (ev === "knock" && data && data.layer === "fc1" && data.off && data.top1) PR.done.knockfc = 1;
    if (ev === "restore" && (PR.done.knockconv || PR.done.knockfc)) PR.done.restore = 1;
    if (ev === "play-over" && P.ms === 0) PR.done.playover = 1;
    if (ev === "replay") PR.done.replay = 1;
    if (Object.keys(PR.done).length !== before) {
      store("tasks", JSON.stringify(PR.done));
      if (app.mode === "practice") renderExplain();
      toast("练习任务完成 ✓");
    }
  }

  // ================================================================== MODES
  function setMode(m) {
    if (!/^(watch|play|practice)$/.test(m)) return;
    var prev = app.mode;
    app.mode = m;
    document.querySelectorAll(".tab").forEach(function (t) { t.classList.toggle("on", t.dataset.mode === m); });
    ["watch", "play", "practice"].forEach(function (k) { document.body.classList.toggle("m-" + k, k === m); });
    document.documentElement.classList.toggle("touchp-play", app.touchp && m === "play");
    if (m !== "play") setSheet(false);
    hideBanner();
    $("board-hint").hidden = true;
    if (m === "play") {
      cancel(W.timer); W.timer = null;
      if (!P.env) playNewGame();
      else { if (P.over) showPlayBannerAgain(); setPlayView(); }
    } else {
      cancel(P.timer); P.timer = null;
      if (P.started && !P.over && P.ms > 0) P.paused = true;
      if (m === "practice" && prev !== "practice") W.playing = false;
      if (m === "watch" && prev === "practice") W.playing = true;
      if (W.over) showWatchBanner();
      setWatchView();
      scheduleWatch();
    }
    renderOpts(); renderControls(); renderExplain(); renderPills();
    layout();
  }

  function showPlayBannerAgain() { if (P.over) { $("banner").hidden = false; } }

  // ================================================================== LEFT: pills / options / controls / explain
  function renderPills() {
    var h = [];
    if (app.mode === "play") {
      var env = P.env, agree = 0;
      P.log.forEach(function (r) { if (r.agree) agree++; });
      h.push("<span class='pill'>第 <b>" + P.episode + "</b> 局</span>");
      h.push("<span class='pill'>得分 <b>" + env.score + "</b></span>");
      h.push("<span class='pill'>长度 <b>" + env.snake.length + "</b></span>");
      h.push("<span class='pill'>步数 <b>" + env.steps + "</b></span>");
      h.push("<span class='pill'>与 DQN 一致 <b>" + (P.log.length ? Math.round(100 * agree / P.log.length) + "%" : "–") + "</b></span>");
    } else {
      var e = W.env, done = W.results.length, sum = 0;
      W.results.forEach(function (r) { sum += r.score; });
      h.push("<span class='pill'>第 <b>" + W.game + "</b> 局</span>");
      h.push("<span class='pill'>得分 <b>" + e.score + "</b></span>");
      h.push("<span class='pill'>长度 <b>" + e.snake.length + "</b></span>");
      h.push("<span class='pill'>步数 <b>" + e.steps + "</b></span>");
      h.push("<span class='pill'>已看完 <b>" + done + "</b> 局，平均 <b>" + (done ? (sum / done).toFixed(2) : "–") + "</b> 分</span>");
    }
    $("pills").innerHTML = h.join("");
    // 手机「我来玩」一屏布局：胶囊行高度变了（例如多出一行）就重新算棋盘大小，保证操作区不被挤出视口
    if (app.touchp && app.mode === "play" && app.alive) {
      var ph = $("pills").offsetHeight;
      if (app._pillsH !== undefined && ph !== app._pillsH) { app._pillsH = ph; layoutMobile(); return; }
      app._pillsH = ph;
    }
  }

  function presetButtons(list, idx, kind) {
    return list.map(function (p, i) {
      return "<button class='preset" + (i === idx ? " on" : "") + "' data-kind='" + kind + "' data-i='" + i + "' data-tip='" + p.name + "：" + p.tip + "'>" + p.name + "</button>";
    }).join("");
  }

  function renderOpts() {
    var o = $("opts"), h;
    if (app.mode === "play") {
      h = "<div class='opt-row'><span class='lab'>难度</span>" + presetButtons(CFG.playLevels, P.levelIdx, "level") +
        "<input type='range' id='rng-ms' min='80' max='800' step='10' value='" + (P.ms || 260) + "'" + (P.ms === 0 ? " disabled" : "") + " data-tip='细调每步间隔（回合制下不用）'>" +
        "<span class='ms' id='ms-lab'>" + (P.ms === 0 ? "等你按键" : P.ms + " 毫秒/步") + "</span></div>" +
        "<div class='opt-row'><label class='ck' data-tip='辅助只是把 DQN 在“同一局面”的建议提前给你看：不改规则、不改食物位置，也不替你走。开过辅助的局在历局表里会标出来，所以比分仍然公平可比。'><input type='checkbox' id='ck-assist'" + (P.assist ? " checked" : "") + "> 辅助：提前显示 DQN 的建议</label>" +
        "<label class='ck' data-tip='红雾 = 网络做这次决定时最“在意”的格子（|∂Q/∂输入|），只在辅助打开或回放时显示' data-anim='fog'><input type='checkbox' id='ck-fog'" + (app.fog ? " checked" : "") + "> 注意力红雾</label></div>";
    } else {
      h = "<div class='opt-row'><span class='lab'>速度</span>" + presetButtons(CFG.watchSpeeds, W.speedIdx, "speed") +
        "<input type='range' id='rng-ms' min='5' max='1000' step='5' value='" + W.ms + "' data-tip='细调每步间隔'>" +
        "<span class='ms' id='ms-lab'>" + W.ms + " 毫秒/步</span></div>" +
        "<div class='opt-row'><label class='ck' data-tip='红雾 = 网络做这次决定时最“在意”的格子：|∂Q(所选方向)/∂输入|，颜色越深影响越大' data-anim='fog'><input type='checkbox' id='ck-fog'" + (app.fog ? " checked" : "") + "> 注意力红雾</label>" +
        "<label class='ck'><input type='checkbox' id='ck-auto'" + (app.autoNext ? " checked" : "") + "> 死了自动开下一局</label></div>";
    }
    o.innerHTML = h;
    bindTips(o);
    renderMobileBar();
  }

  /** 手机「我来玩」操作区顶部一行：难度档位 + 看大脑 / 辅助 / 重开。 */
  function renderMobileBar() {
    $("mob-levels").innerHTML = presetButtons(CFG.playLevels, P.levelIdx, "level");
    $("tk-assist").classList.toggle("on", !!P.assist);
    $("tk-assist").setAttribute("aria-pressed", P.assist ? "true" : "false");
    $("b-mnext").textContent = P.over ? "下一局" : "重开";
  }

  /** 右下角大圆键的含义随状态变化：开始 / 暂停 / 继续 / 下一局；回合制下是「直走一步」。 */
  function renderTouchKey() {
    var b = $("tk-pause"), hint = $("tk-hint"), label, tip, go = false;
    if (P.over) { label = "下一局"; tip = "本局结束，再来一局"; go = true; }
    else if (P.ms === 0) { label = "直走"; tip = "回合制：拖一次摇杆走一步，这个键让蛇直走一步"; }
    else if (!P.started) { label = "开始"; tip = "开始后蛇会自动前进，拖摇杆改方向"; go = true; }
    else if (P.paused) { label = "继续"; tip = "已暂停"; go = true; }
    else { label = "暂停"; tip = "蛇正在自动前进"; }
    b.textContent = label; b.classList.toggle("go", go); hint.textContent = tip;
  }

  function onTouchPress(key) {
    if (app.mode !== "play" || key !== "pause") return;
    if (P.over) { playNewGame(); renderControls(); renderPills(); return; }
    if (P.ms === 0) { playKey(P.env.direction); return; }
    if (!P.started) { P.started = true; P.paused = false; schedulePlay(); renderControls(); $("board-hint").hidden = true; return; }
    P.paused = !P.paused; schedulePlay(); renderControls();
  }

  function setSheet(on) {
    app.sheet = !!on;
    document.body.classList.toggle("sheet-brain", app.sheet);
    if (app.sheet) layout();
  }

  function renderControls() {
    var c = $("controls"), h;
    if (app.mode === "play") {
      var timed = P.ms > 0;
      h = "<button class='btn go' id='b-next' data-tip='开新的一局（种子 = " + (CFG.playSeed0) + " + 局号；DQN 用同一个开局）'>" + (P.over ? "下一局（Enter）" : "重开一局") + "</button>" +
        (timed ? "<button class='btn' id='b-pause'" + (P.over || !P.started ? " disabled" : "") + ">" + (P.paused ? "▶ 继续" : "⏸ 暂停") + "</button>" : "") +
        "<span style='color:var(--muted);font-size:12px'>方向键 / WASD；手机在棋盘上滑动</span>";
    } else {
      h = "<button class='btn " + (W.playing ? "" : "go") + "' id='b-play' data-tip='播放 / 暂停（空格）'>" + (W.playing ? "⏸ 暂停" : "▶ 播放") + "</button>" +
        "<button class='btn' id='b-step' data-tip='暂停并只走一步（→ 或 . 键），方便对照右边的大脑' data-anim='step'>单步</button>" +
        "<button class='btn' id='b-reset' data-tip='从头再来：回到和评估脚本一样的第 1 局（种子 999）'>重开</button>";
    }
    c.innerHTML = h;
    bindTips(c);
    renderTouchKey();
    $("b-mnext").textContent = P.over ? "下一局" : "重开";
  }

  function renderExplain() {
    var e = $("explain"), h = "";
    if (app.mode === "watch") {
      var ev = META.eval;
      h = "<p><b>你在看什么：</b>蛇由一个卷积神经网络控制（DQN，深度 Q 网络）。它每一步给上下左右 4 个方向各打一个分——<b>Q 值</b>，意思是“往这边走，以后（打折后）大概还能拿到多少奖励”，然后选分最高的方向。</p>" +
        "<p><b>右边是它的大脑：</b>从输入（它“看到”的 5 张 10×10 图）到 3 个卷积层、1 个全连接层，再到 4 个 Q 值，全部是真实权重在你的浏览器里实时算出来的。点任何方块或圆点可以把它敲掉。</p>" +
        (ev ? "<p style='color:var(--muted)'>开局顺序和训练脚本的评估一样（种子 " + ev.seed + "）：不敲掉任何神经元时，前 " + ev.games + " 局平均吃到 " + ev.avg_score_fp16 + " 个（见 results/result.json）。</p>" : "");
    } else if (app.mode === "practice") {
      var n = 0;
      TASKS.forEach(function (t) { if (PR.done[t.id]) n++; });
      var cur = TASKS.filter(function (t) { return !PR.done[t.id]; })[0];
      h = "<h3>练习：跟着清单看懂这个网络（已完成 " + n + " / " + TASKS.length + "）</h3><div class='progress'><i style='width:" + (100 * n / TASKS.length) + "%'></i></div><ul class='tasks'>" +
        TASKS.map(function (t, i) {
          var d = !!PR.done[t.id];
          return "<li class='" + (d ? "done" : (cur === t ? "cur" : "")) + "' data-tip='" + t.text.replace(/<[^>]+>/g, "") + "' data-anim='" + t.anim + "'><span class='box'>" + (d ? "✓" : "") + "</span><span>" + (i + 1) + ". " + t.text + "</span></li>";
        }).join("") + "</ul>" +
        (n === TASKS.length ? "<p style='color:var(--green);margin-top:6px'><b>全部完成！</b>你已经会读这个网络的“大脑”了。</p>" : "") +
        "<p style='margin-top:4px'><button class='mini-btn' id='b-tasks-reset'>清空进度重来</button></p>";
    } else {
      h = "<p><b>怎么玩：</b>和 DQN 玩<b>同一局</b>（同样的起点和食物）。你每走一步，就揭晓 DQN 在<b>同一局面</b>会怎么走：<span class='hl-you'>黄色实线</span> = 你，<span class='hl-dqn'>蓝色虚线</span> = DQN。右边的大脑显示它在那个局面的想法。</p>";
      if (P.history.length) {
        h += "<table class='hist'><tr><th>局</th><th>你：得分 / 步数</th><th>DQN：得分 / 步数</th><th>一致</th><th>备注</th></tr>" +
          P.history.slice(-6).map(function (r) {
            var note = [r.assist ? "辅助" : "", r.knocked ? "有敲掉" : ""].filter(Boolean).join("、") || "–";
            return "<tr><td>" + r.ep + "</td><td>" + r.you.score + " / " + r.you.steps + "</td><td>" + r.dqn.score + " / " + r.dqn.steps + "</td><td>" +
              (r.agree === null ? "–" : Math.round(r.agree * 100) + "%") + "</td><td>" + note + "</td></tr>";
          }).join("") + "</table>";
      }
      var dis = P.log.filter(function (r) { return !r.agree; });
      if (P.over && P.over.fatal && P.over.fatal.agree) dis.push(P.over.fatal);
      if (dis.length) {
        h += "<div style='margin-top:6px'>本局「你 ≠ DQN」的时刻（点开回放）：</div><div class='chips'>" +
          dis.slice(-24).map(function (r) {
            var fatal = P.over && r === P.over.fatal;
            return "<button class='chip" + (fatal ? " fatal" : "") + (P.replay === r ? " on" : "") + "' data-step='" + r.step + "'>第 " + r.step + " 步 " + ACT[r.you].arrow + " vs " + ACT[r.dqn].arrow + (fatal ? " 致命" : "") + "</button>";
          }).join("") + (P.replay ? "<button class='chip' id='b-back'>回到当前</button>" : "") + "</div>";
      }
    }
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
    var snap = v.snap, n = snap.n, cs = s / n;
    ctx.fillStyle = "#0b1016"; ctx.fillRect(0, 0, s, s);
    ctx.strokeStyle = CFG.colors.grid; ctx.lineWidth = 1;
    for (var i = 0; i <= n; i++) {
      ctx.beginPath(); ctx.moveTo(i * cs, 0); ctx.lineTo(i * cs, s); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cs); ctx.lineTo(s, i * cs); ctx.stroke();
    }
    if (v.fog) {
      var mx = 1e-8;
      for (i = 0; i < NN; i++) if (v.fog[i] > mx) mx = v.fog[i];
      for (i = 0; i < NN; i++) {
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
  function diverge(v) {   // -1..1 -> blue / black / amber
    v = Math.max(-1, Math.min(1, v));
    if (v >= 0) return "rgb(" + (255 * v | 0) + "," + (183 * v | 0) + "," + (77 * v | 0) + ")";
    return "rgb(" + (79 * -v | 0) + "," + (195 * -v | 0) + "," + (247 * -v | 0) + ")";
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
  function drawMap(cv, arr, off, size, scale, div) {
    var ctx = sizeCanvas(cv, size, size), cs = size / N;
    for (var r = 0; r < N; r++) for (var c = 0; c < N; c++) {
      var v = arr[off + r * N + c] / scale;
      ctx.fillStyle = div ? diverge(v) : heat(v);
      ctx.fillRect(c * cs, r * cs, cs + 0.5, cs + 0.5);
    }
  }
  function fmt(v, d) { return (v >= 0 ? "+" : "") + v.toFixed(d === undefined ? 2 : d); }

  function renderFormula() {
    var s = SIZES, fcShare = (net.w["fc1.weight"].length + net.w["fc1.bias"].length) / net.params;
    $("formula").innerHTML = "输入 " + META.channels + "×" + N + "×" + N + "<span class='arrow'>→</span>卷积 " + s.conv1 +
      "<span class='arrow'>→</span>卷积 " + s.conv2 + "<span class='arrow'>→</span>卷积 " + s.conv3 +
      "<span class='arrow'>→</span>全连接 " + s.fc1 + "<span class='arrow'>→</span>输出 " + s.out + " 个方向的 Q 值" +
      "<span class='param'>参数 " + net.params.toLocaleString("en-US") + " 个（" + (100 * fcShare).toFixed(1) + "% 在全连接层）· float16</span>";
    var g = CFG.inputGroups;
    $("lt-input").innerHTML = "<b>输入层</b> <span class='cnt'>" + (META.channels * NN) + "</span> 维 = " +
      g.map(function (x) { return x.name + " " + NN; }).join(" + ") + "（每组一张 " + N + "×" + N + " 图）";
    ["conv1", "conv2", "conv3"].forEach(function (k, i) {
      $("lt-" + k).innerHTML = "<b>卷积层 " + (i + 1) + "</b> <span class='cnt'>" + s[k] + "</span> 个通道 × " + NN + " 位置 = " + (s[k] * NN).toLocaleString("en-US") +
        " 个神经元 · 展开贡献最大的 " + app.kConv + " 个通道 · 方阵 = 全部 " + s[k] + " 个（点击敲掉）";
    });
    $("lt-fc").innerHTML = "<b>全连接层</b> <span class='cnt'>" + s.fc1 + "</span> 个神经元 · 方阵 = 全部（<span id='fc-active'></span>）· 圆点 = 贡献最大的至多 " + CFG.topK.fc +
      " 个 · 连线 <span style='color:var(--pos)'>橙</span>推高 / <span style='color:var(--neg)'>蓝</span>压低 → <b>输出 " + s.out + " 个方向</b>（% = 按 Q 换算的倾向，仅示意，DQN 总选 Q 最大的）";
    ["lt-input", "lt-conv1", "lt-conv2", "lt-conv3", "lt-fc"].forEach(function (id) { $(id).title = $(id).textContent; });
  }

  function buildInputRow() {
    var row = $("row-input");
    row.innerHTML = CFG.inputGroups.map(function (g) {
      return "<div class='in-map' data-tip='" + g.name + "（" + NN + " 维）：" + g.desc + "'><canvas id='in-" + g.ch + "'></canvas><div class='nm'>" + g.name +
        " <span class='dims'>" + NN + " 维</span></div><div class='dims'>归因 <span id='inv-" + g.ch + "'></span></div><div class='attr'><i id='ina-" + g.ch + "'></i></div></div>";
    }).join("");
    bindTips(row);
  }

  function buildTops(layer) {
    var box = $("tops-" + layer), k = CFG.topK.conv, h = "";
    for (var i = 0; i < k; i++) h += "<div class='fm' data-layer='" + layer + "' data-slot='" + i + "' data-tip='这个通道在 " + N + "×" + N + " 棋盘上的激活图。点一下 = 敲掉 / 恢复' data-anim='click'><canvas></canvas><div class='cap'></div></div>";
    box.innerHTML = h;
    bindTips(box);
  }

  var hits = { fc: [], sq: {} };

  function renderBrain() {
    var v = app.view;
    if (!v || !v.res) return;
    var res = v.res, sz = app.sizes, act = res.action;
    $("view-label").innerHTML = v.label || "";
    var kc = net.knockCount();
    $("knock-info").textContent = kc ? "已敲掉 " + kc + " 个" : "";
    $("btn-restore").disabled = kc === 0;
    var cover = $("brain-cover");
    if (v.cover) { cover.hidden = false; cover.innerHTML = "走出第一步后，这里揭晓 DQN 在<b>同一局面</b>的想法。<br><span style='font-size:13px;color:var(--muted)'>想边走边看？在左边勾选「辅助：提前显示 DQN 的建议」。</span>"; }
    else cover.hidden = true;

    // ---- input
    var x = (v.inputSnap ? envFromSnap(v.inputSnap) : envFromSnap(v.snap)).state();
    var ci = res.contrib.input, mxa = 1e-8;
    for (var c = 0; c < ci.length; c++) mxa = Math.max(mxa, Math.abs(ci[c]));
    CFG.inputGroups.forEach(function (g) {
      drawMap($("in-" + g.ch), x, g.ch * NN, sz.map, 1, g.ch >= 3);
      var a = ci[g.ch] / mxa, bar = $("ina-" + g.ch);
      bar.style.left = (a >= 0 ? 50 : 50 + 50 * a) + "%";
      bar.style.width = (50 * Math.abs(a)) + "%";
      bar.style.background = a >= 0 ? CFG.colors.pos : CFG.colors.neg;
      $("inv-" + g.ch).textContent = fmt(ci[g.ch]);
    });

    // ---- conv layers
    [["conv1", res.a1], ["conv2", res.a2], ["conv3", res.a3]].forEach(function (pair) {
      var layer = pair[0], a = pair[1], C = SIZES[layer], contrib = res.contrib[layer];
      var order = [];
      for (var i = 0; i < C; i++) order.push(i);
      order.sort(function (p, q) { return Math.abs(contrib[q]) - Math.abs(contrib[p]); });
      var tops = order.filter(function (c2) { return contrib[c2] !== 0; }).slice(0, app.kConv);
      var slots = $("tops-" + layer).children;
      for (var sl = 0; sl < slots.length; sl++) {
        slots[sl].style.visibility = sl < tops.length ? "visible" : "hidden";
        slots[sl].style.display = sl < app.kConv || sl < tops.length ? "" : "none";   // 手机紧凑画法：多余的槽位不占位
        if (sl >= tops.length) delete slots[sl].dataset.ch;
      }
      tops.forEach(function (ch, s) {
        var el = slots[s], cv = el.querySelector("canvas"), mx = 1e-8;
        for (var p = 0; p < NN; p++) mx = Math.max(mx, a[ch * NN + p]);
        drawMap(cv, a, ch * NN, sz.map, mx, false);
        var cc = contrib[ch], off = net.isKnocked(layer, ch);
        el.className = "fm " + (off ? "off" : (cc >= 0 ? "pos" : "neg"));
        el.dataset.ch = ch;
        var cap = el.querySelector(".cap");
        cap.className = "cap " + (cc >= 0 ? "pos" : "neg");
        cap.textContent = "#" + ch + " " + fmt(cc);
      });
      // full-layer square: mean activation per channel
      var cols = 8, rows = Math.ceil(C / cols), t = Math.max(5, Math.min(16, Math.floor((sz.map + 18) / rows)));
      var cv2 = $("sq-" + layer), ctx = sizeCanvas(cv2, cols * t, rows * t), means = new Float32Array(C), mm = 1e-8;
      for (i = 0; i < C; i++) { var s2 = 0; for (var p2 = 0; p2 < NN; p2++) s2 += a[i * NN + p2]; means[i] = s2 / NN; mm = Math.max(mm, means[i]); }
      ctx.clearRect(0, 0, cols * t, rows * t);
      for (i = 0; i < C; i++) {
        var cx = (i % cols) * t, cy = Math.floor(i / cols) * t;
        if (net.isKnocked(layer, i)) {
          ctx.fillStyle = "#333"; ctx.fillRect(cx + 0.5, cy + 0.5, t - 1, t - 1);
          ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(cx + 2, cy + 2); ctx.lineTo(cx + t - 2, cy + t - 2);
          ctx.moveTo(cx + t - 2, cy + 2); ctx.lineTo(cx + 2, cy + t - 2); ctx.stroke();
        } else { ctx.fillStyle = heat(means[i] / mm); ctx.fillRect(cx + 0.5, cy + 0.5, t - 1, t - 1); }
        if (tops.indexOf(i) >= 0) { ctx.strokeStyle = contrib[i] >= 0 ? CFG.colors.pos : CFG.colors.neg; ctx.lineWidth = 1.5; ctx.strokeRect(cx + 1, cy + 1, t - 2, t - 2); }
      }
      hits.sq[layer] = { cols: cols, t: t, C: C };
    });

    renderFC(res, act);
    renderValue(v);
  }

  function renderFC(res, act) {
    var cv = $("fc-canvas"), W0 = app.sizes.brainW, H0 = app.sizes.fcH, ctx = sizeCanvas(cv, W0, H0);
    ctx.clearRect(0, 0, W0, H0);
    var H = SIZES.fc1, h = res.h, contrib = res.contrib.fc1, fcOut = res.fcOut;
    hits.fc = [];
    // left: 16x16 grid of all neurons (activation)
    var side = Math.ceil(Math.sqrt(H)), g = Math.min(H0 - 4, Math.floor(W0 * 0.3)), cs = g / side, mh = 1e-8, i;
    for (i = 0; i < H; i++) mh = Math.max(mh, h[i]);
    var order = [];
    for (i = 0; i < H; i++) order.push(i);
    order.sort(function (p, q) { return Math.abs(contrib[q]) - Math.abs(contrib[p]); });
    var active = 0;
    for (i = 0; i < H; i++) if (h[i] > 0) active++;
    var tops = order.filter(function (j) { return contrib[j] !== 0; }).slice(0, CFG.topK.fc), K = tops.length, top1 = tops[0];
    for (i = 0; i < H; i++) {
      var gx = (i % side) * cs, gy = Math.floor(i / side) * cs + 2;
      if (net.isKnocked("fc1", i)) { ctx.fillStyle = "#444"; ctx.fillRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1); ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(gx + 1, gy + 1); ctx.lineTo(gx + cs - 1, gy + cs - 1); ctx.stroke(); }
      else { ctx.fillStyle = heat(h[i] / mh); ctx.fillRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1); }
      if (tops.indexOf(i) >= 0) { ctx.strokeStyle = contrib[i] >= 0 ? CFG.colors.pos : CFG.colors.neg; ctx.lineWidth = 1; ctx.strokeRect(gx + 0.5, gy + 0.5, cs - 1, cs - 1); }
      hits.fc.push({ x: gx, y: gy, w: cs, h: cs, j: i, top1: i === top1 });
    }
    // middle: top-K dots in 2 columns (compact spacing on narrow screens)
    var compact = W0 < 560;
    var colN = Math.max(1, Math.ceil(K / 2)), rowH = (H0 - 8) / colN, r = Math.max(4, Math.min(rowH * 0.36, 9));
    var dx0 = g + (compact ? 22 : 26), dx1 = dx0 + (compact ? 40 : 54), outX = compact ? dx1 + 30 : Math.max(dx1 + 110, W0 - 230), dots = [];
    var mc = 1e-8;
    tops.forEach(function (j) { mc = Math.max(mc, Math.abs(contrib[j])); });
    tops.forEach(function (j, k) {
      var col = k < colN ? 0 : 1, row = k % colN;
      dots.push({ j: j, x: col ? dx1 : dx0, y: 4 + rowH * (row + 0.5) });
    });
    // output rows
    var rowsY = [], oh = (H0 - 8) / 4;
    for (var o = 0; o < 4; o++) rowsY.push(4 + oh * (o + 0.5));
    // lines: strongest |h_j * W2[o,j]| among the shown dots
    var lines = [];
    dots.forEach(function (d) { for (var o2 = 0; o2 < 4; o2++) lines.push({ d: d, o: o2, v: fcOut[o2][d.j] }); });
    lines.sort(function (p, q) { return Math.abs(q.v) - Math.abs(p.v); });
    var ml = lines.length ? Math.abs(lines[0].v) || 1e-8 : 1;
    lines.slice(0, CFG.topK.lines).forEach(function (L) {
      ctx.strokeStyle = (L.v >= 0 ? "rgba(255,183,77," : "rgba(79,195,247,") + (0.25 + 0.6 * Math.abs(L.v) / ml).toFixed(2) + ")";
      ctx.lineWidth = 0.6 + 3 * Math.abs(L.v) / ml;
      ctx.beginPath(); ctx.moveTo(L.d.x + r, L.d.y);
      ctx.bezierCurveTo((L.d.x + outX) / 2, L.d.y, (L.d.x + outX) / 2, rowsY[L.o], outX - 4, rowsY[L.o]); ctx.stroke();
    });
    dots.forEach(function (d, k) {
      var off = net.isKnocked("fc1", d.j), cc = contrib[d.j];
      ctx.fillStyle = off ? "#444" : heat(h[d.j] / mh);
      ctx.beginPath(); ctx.arc(d.x, d.y, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = off ? "#ff6b6b" : (cc >= 0 ? CFG.colors.pos : CFG.colors.neg);
      ctx.lineWidth = 1 + 2 * Math.abs(cc) / mc; ctx.stroke();
      ctx.fillStyle = "#8b98a8"; ctx.font = "10px Consolas, Menlo, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      if (k < colN) ctx.fillText("#" + d.j, d.x - r - 3, d.y);
      else { ctx.textAlign = "left"; ctx.fillText("#" + d.j, d.x + r + 3, d.y - r - 1 > 0 ? d.y : d.y); }
      hits.fc.push({ x: d.x - r - 2, y: d.y - r - 2, w: 2 * r + 4, h: 2 * r + 4, j: d.j, top1: d.j === top1 });
    });
    // outputs
    var q = res.q, legal = res.legal, prob = BrainNet.softmaxLegal(q, legal), qhi = -Infinity, qlo = 0;
    for (o = 0; o < 4; o++) { qhi = Math.max(qhi, q[o]); qlo = Math.min(qlo, q[o]); }
    if (qhi - qlo < 1e-6) qhi = qlo + 1;
    var barX = outX + (compact ? 24 : 40), barW = Math.max(40, W0 - barX - (compact ? 66 : 90));
    ctx.textBaseline = "middle";
    for (o = 0; o < 4; o++) {
      var y = rowsY[o], chosen = o === act, il = !legal[o];
      ctx.globalAlpha = il ? 0.35 : 1;
      ctx.fillStyle = chosen ? CFG.colors.chosen : "#dde5ee"; ctx.font = "bold 13px 'Microsoft YaHei','PingFang SC',sans-serif"; ctx.textAlign = "left";
      ctx.fillText(compact ? ACT[o].arrow : ACT[o].arrow + " " + ACT[o].name, outX, y);
      ctx.fillStyle = "#0b1016"; ctx.fillRect(barX, y - 9, barW, 18);
      var wq = barW * Math.max(0.03, (q[o] - qlo) / (qhi - qlo));
      ctx.fillStyle = chosen ? CFG.colors.chosen : "#355d83"; ctx.fillRect(barX, y - 9, wq, 18);
      ctx.fillStyle = "#dde5ee"; ctx.font = (compact ? "11px" : "12px") + " Consolas, Menlo, monospace";
      ctx.fillText("Q " + q[o].toFixed(compact ? 2 : 3), barX + barW + 5, y - 6);
      ctx.fillStyle = "#8b98a8"; ctx.font = (compact ? "10.5px" : "11px") + " 'Microsoft YaHei','PingFang SC',sans-serif";
      ctx.fillText(il ? (compact ? "不允许" : "掉头·不允许") : (chosen ? "✓ " : "") + Math.round(prob[o] * 100) + "%", barX + barW + 5, y + 8);
      ctx.globalAlpha = 1;
    }
    app._fcTop1 = top1;
    $("fc-active").textContent = "此刻有 " + active + " 个神经元激活（> 0），其余为 0";
  }

  function renderValue(v) {
    var res = v.res, val = maxLegal(res.q, res.legal);
    $("value-text").innerHTML = "局面价值 <b>" + val.toFixed(2) + "</b>";
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
  function toggleKnock(layer, idx, top1) {
    var off = !net.isKnocked(layer, idx);
    net.setKnock(layer, idx, off);
    reanalyze();
    toast((off ? "已敲掉 " : "已恢复 ") + (layer === "fc1" ? "全连接层神经元 #" : layer.replace("conv", "卷积层 ") + " 通道 #") + idx);
    practiceEvent("knock", { layer: layer, idx: idx, off: off, top1: !!top1 });
  }
  function fcTop1(res) {
    var c = res.contrib.fc1, best = 0;
    for (var j = 1; j < c.length; j++) if (Math.abs(c[j]) > Math.abs(c[best])) best = j;
    return best;
  }
  function reanalyze() {
    if (app.mode === "play") {
      if (P.env && !P.over) { P.cur = analyze(P.env); }
      if (P.prev && P.prevSnap) { var e = envFromSnap(P.prevSnap); P.prev = net.forward(e.state(), e.legalMask(), true); }
      setPlayView();
    } else {
      if (!W.over) { W.lastRes = analyze(W.env); }
      else { var e2 = envFromSnap(W.over.snapBefore); W.lastRes = net.forward(e2.state(), e2.legalMask(), true); }
      setWatchView();
    }
  }

  // ================================================================== layout / render loop
  /** 触屏 / 手机竖屏判断（v2 第八节：触屏 + 宽度 ≤ 980px + 竖屏；768×1024 的平板也算）。 */
  function detectTouch() {
    var touch = false;
    try { touch = (navigator.maxTouchPoints > 0) || window.matchMedia("(pointer: coarse)").matches; } catch (e) { touch = false; }
    var w = window.innerWidth, h = window.innerHeight, portrait = h >= w;
    var before = app.touchp;
    app.touch = touch;
    app.touchp = touch && w <= 980 && portrait;
    app.touchl = touch && w <= 980 && !portrait;
    document.body.classList.toggle("touch", app.touch);
    document.body.classList.toggle("touchp", app.touchp);
    document.body.classList.toggle("touchl", app.touchl);
    document.documentElement.classList.toggle("touchp", app.touchp);
    document.documentElement.classList.toggle("touchp-play", app.touchp && app.mode === "play");
    $("rotate-hint").hidden = !app.touchl;
    app.kConv = app.touchp && w < 600 ? 4 : CFG.topK.conv;
    if (!app.touchp) { setSheet(false); $("menu").hidden = true; }
    if (before !== app.touchp) { renderFormula(); renderOpts(); renderControls(); if (app.mode === "play") setPlayView(); }
    return app.touchp;
  }

  function layoutMobile() {
    var main = $("main"), left = $("left"), right = $("right"), inner = left.clientWidth - 16;
    var b;
    if (app.mode === "play") {
      var avail = main.clientHeight - $("touchctl").offsetHeight - $("pills").offsetHeight - 36;
      b = Math.max(150, Math.min(inner - 2, avail));
    } else b = Math.min(inner - 2, 640);
    app.sizes.board = Math.max(N * 10, Math.floor(b / N) * N);
    var rw = right.clientWidth - 18;
    if (rw < 120) rw = window.innerWidth - 16 - 18;
    app.sizes.brainW = Math.max(240, rw);
    app.sizes.map = Math.max(28, Math.floor(Math.min(64, (inner - 44) / 5) / 2) * 2);
    app.sizes.fcH = inner < 600 ? 250 : 300;
    if (!app.view || !app.alive) { markDirty(); return; }
    drawBoard(); renderBrain(); renderPills();
  }

  function layout() {
    if (app.touchp) { layoutMobile(); return; }
    var main = $("main"), left = $("left"), right = $("right");
    var lw = left.clientWidth, mh = main.clientHeight;
    var reserve = app.mode === "practice" ? 420 : (mh < 700 ? 255 : (app.mode === "play" ? 330 : 300));
    var b = Math.max(220, Math.min(lw - 24, mh - reserve, 680));
    app.sizes.board = Math.floor(b / N) * N;
    var rw = right.clientWidth - 22, rh = right.clientHeight;
    app.sizes.brainW = Math.max(300, rw);
    var s = (rh - 268 - 130) / 4;
    var byW = (rw - 120) / (CFG.topK.conv + 2.2) - 8;
    s = Math.max(30, Math.min(s, byW, 86));
    app.sizes.map = Math.floor(s / 2) * 2;
    app.sizes.fcH = Math.max(120, Math.min(320, Math.round(rh - 4 * app.sizes.map - 268)));
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
    keys: "<div class='demo demo-keys'><i class='k k-up'>↑</i><i class='k k-left'>←</i><i class='k k-down'>↓</i><i class='k k-right'>→</i></div>",
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
    if (app.touch) return;   // 触屏上没有悬停，不弹说明气泡（轻点会触发模拟的 mouseenter）
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
    if (app.touchCtl) { app.touchCtl.unbind(); app.touchCtl = null; }
    W.env = null; P.env = null; app.view = null;
    net = null;
    window.SNAKE_MODEL = null;
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
    var dirMap = { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3, w: 0, s: 1, a: 2, d: 3, W: 0, S: 1, A: 2, D: 3 };
    if (app.mode === "play" && k in dirMap) { e.preventDefault(); playKey(dirMap[k]); return; }
    if (k === "h" || k === "H" || k === "?") { openHelp(); return; }
    if (k === "b" || k === "B") { setBig(!document.body.classList.contains("big")); return; }
    if (k === "1") return setMode("watch");
    if (k === "2") return setMode("play");
    if (k === "3") return setMode("practice");
    if (app.mode === "play") {
      if (k === "Enter" && P.over) { playNewGame(); renderControls(); renderPills(); }
      else if ((k === "r" || k === "R")) { playNewGame(); renderControls(); renderPills(); }
      else if (k === " " && P.ms > 0 && P.started && !P.over) { e.preventDefault(); P.paused = !P.paused; schedulePlay(); renderControls(); }
      return;
    }
    if (k === " ") { e.preventDefault(); setWatchPlaying(!W.playing); }
    else if (k === "ArrowRight" || k === ".") { e.preventDefault(); manualStep(); }
    else if (k === "r" || k === "R") { hideBanner(); watchRestart(); }
  }

  function manualStep() {
    if (W.playing) setWatchPlaying(false);
    if (W.over) { hideBanner(); watchNextGame(); }
    else watchStep();
    practiceEvent("manual-step");
  }

  function onClick(e) {
    var t = e.target;
    if (!t.closest("#menu") && t.id !== "menu-btn") { $("menu").hidden = true; $("menu-btn").setAttribute("aria-expanded", "false"); }
    var tab = t.closest(".tab");
    if (tab) { setMode(tab.dataset.mode); return; }
    var pre = t.closest(".preset");
    if (pre) {
      var i = +pre.dataset.i;
      if (pre.dataset.kind === "speed") { W.speedIdx = i; W.ms = CFG.watchSpeeds[i].ms; renderOpts(); scheduleWatch(); practiceEvent("speed"); }
      else {
        P.levelIdx = i; P.ms = CFG.playLevels[i].ms; renderOpts(); renderControls();
        if (P.ms === 0) { cancel(P.timer); P.timer = null; } else if (P.started && !P.over && !P.paused) schedulePlay();
        setPlayView();
      }
      return;
    }
    var fm = t.closest(".fm");
    if (fm && fm.dataset.ch !== undefined) { toggleKnock(fm.dataset.layer, +fm.dataset.ch, false); return; }
    var chip = t.closest(".chip");
    if (chip) {
      if (chip.id === "b-back") { P.replay = null; }
      else {
        var st = +chip.dataset.step;
        P.replay = P.log.filter(function (r) { return r.step === st; })[0] || null;
        if (P.replay) { cancel(P.timer); P.timer = null; if (P.ms > 0 && P.started && !P.over) P.paused = true; practiceEvent("replay"); }
      }
      setPlayView(); renderExplain(); renderControls();
      return;
    }
    switch (t.id) {
      case "b-play": setWatchPlaying(!W.playing); break;
      case "b-step": manualStep(); break;
      case "b-reset": hideBanner(); watchRestart(); scheduleWatch(); break;
      case "b-next": playNewGame(); renderControls(); renderPills(); break;
      case "b-pause": P.paused = !P.paused; schedulePlay(); renderControls(); break;
      case "btn-restore": net.restoreAll(); reanalyze(); toast("所有神经元已恢复"); practiceEvent("restore"); break;
      case "btn-help": openHelp(); break;
      case "help-close": closeHelp(); break;
      case "btn-exit": shutdown(); break;
      case "b-tasks-reset": PR.done = {}; PR.stepCount = 0; store("tasks", "{}"); renderExplain(); break;
      // 手机竖屏：☰ 菜单、看大脑、操作区顶部一行、横屏提示
      case "menu-btn": var mn = $("menu"); mn.hidden = !mn.hidden; t.setAttribute("aria-expanded", mn.hidden ? "false" : "true"); break;
      case "menu-help": $("menu").hidden = true; openHelp(); break;
      case "menu-restore": $("menu").hidden = true; net.restoreAll(); reanalyze(); toast("所有神经元已恢复"); practiceEvent("restore"); break;
      case "menu-exit": $("menu").hidden = true; shutdown(); break;
      case "btn-brain": setSheet(true); break;
      case "brain-close": setSheet(false); break;
      case "rot-ok": document.body.classList.add("rot-ok"); break;
      case "tk-assist": P.assist = !P.assist; if (P.assist && P.env && !P.over) P.assistUsed = true; renderOpts(); setPlayView(); break;
      case "b-mnext": playNewGame(); renderControls(); renderPills(); break;
    }
  }

  function onChange(e) {
    var t = e.target;
    if (t.id === "ck-fog") { app.fog = t.checked; if (app.mode === "play") setPlayView(); else setWatchView(); practiceEvent("fog"); }
    else if (t.id === "ck-auto") { app.autoNext = t.checked; }
    else if (t.id === "ck-assist") { P.assist = t.checked; if (P.assist && P.env && !P.over) P.assistUsed = true; setPlayView(); }
    else if (t.id === "ck-big") { setBig(t.checked); }
  }
  function onInput(e) {
    var t = e.target;
    if (t.id === "rng-ms") {
      var v = +t.value;
      if (app.mode === "play") { P.ms = v; P.levelIdx = -1; $("ms-lab").textContent = v + " 毫秒/步"; document.querySelectorAll(".preset").forEach(function (b) { b.classList.remove("on"); }); if (P.started && !P.over && !P.paused) schedulePlay(); }
      else { W.ms = v; W.speedIdx = -1; $("ms-lab").textContent = v + " 毫秒/步"; document.querySelectorAll(".preset").forEach(function (b) { b.classList.remove("on"); }); scheduleWatch(); }
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

  function bindCanvases() {
    var fc = $("fc-canvas");
    on(fc, "click", function (e) { var h = canvasHit(e, hits.fc); if (h) toggleKnock("fc1", h.j, h.top1); });
    on(fc, "mousemove", function (e) {
      var h = canvasHit(e, hits.fc);
      if (!h || !app.view || !app.view.res) { hideTip(); fc.style.cursor = "default"; return; }
      fc.style.cursor = "pointer";
      var res = app.view.res;
      showTipAt(e.clientX, e.clientY, "全连接层神经元 <b>#" + h.j + "</b>" + (net.isKnocked("fc1", h.j) ? "（已敲掉）" : "") +
        "<br>激活 " + res.h[h.j].toFixed(3) + "<br>对“" + ACT[res.action].name + "”的贡献 " + fmt(res.contrib.fc1[h.j], 3) + "<br>点一下 = 敲掉 / 恢复" + DEMOS.click);
    });
    on(fc, "mouseleave", hideTip);
    ["conv1", "conv2", "conv3"].forEach(function (layer) {
      var cv = $("sq-" + layer);
      on(cv, "click", function (e) { var i = sqHit(e, layer); if (i !== null) toggleKnock(layer, i, false); });
      on(cv, "mousemove", function (e) {
        var i = sqHit(e, layer);
        if (i === null || !app.view || !app.view.res) return;
        var res = app.view.res;
        showTipAt(e.clientX, e.clientY, layer.replace("conv", "卷积层 ") + " 通道 <b>#" + i + "</b>" + (net.isKnocked(layer, i) ? "（已敲掉）" : "") +
          "<br>对“" + ACT[res.action].name + "”的贡献 " + fmt(res.contrib[layer][i], 3) + "<br>点一下 = 敲掉 / 恢复");
      });
      on(cv, "mouseleave", hideTip);
    });
    // swipe on the board (phones / touchpads)
    var bw = $("board-wrap"), start = null;
    on(bw, "pointerdown", function (e) { start = [e.clientX, e.clientY]; });
    on(bw, "pointerup", function (e) {
      if (!start || app.mode !== "play") { start = null; return; }
      var dx = e.clientX - start[0], dy = e.clientY - start[1];
      start = null;
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
      playKey(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 3 : 2) : (dy > 0 ? 1 : 0));
    });
  }

  function renderFooter() {
    var c = CREDITS;
    $("foot").textContent = [(c.AUTHOR || "Shuyao Gao") + (c.AUTHOR_EMAIL ? "（" + c.AUTHOR_EMAIL + "）" : ""), c.UNIVERSITY_EN, c.UNIVERSITY_KO,
      (c.SERIES_NAME ? c.SERIES_NAME + "系列" : ""), "代码 MIT · 文档 CC BY 4.0", "离线纯静态网页，关掉标签页即全部结束"]
      .filter(Boolean).join("  ·  ");
  }

  // ================================================================== self-test (used by tools/browser_check.py)
  function wait(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  async function selftest() {
    var out = { steps: [], ok: false };
    function note(s) { out.steps.push(s); }
    try {
      closeHelp(); openHelp(); closeHelp(); note("help");
      setMode("watch"); W.speedIdx = 3; W.ms = 5; renderOpts(); setWatchPlaying(true);
      await wait(1500);
      note("watch steps=" + W.totalSteps + " games=" + W.results.length);
      if (W.totalSteps < 20) throw new Error("watch mode did not advance");
      setWatchPlaying(false); manualStep(); manualStep(); manualStep(); note("manual steps");
      toggleKnock("conv2", 0, false); toggleKnock("fc1", fcTop1(W.lastRes), true); toggleKnock("conv1", 3, false);
      if (net.knockCount() !== 3) throw new Error("knock count");
      manualStep(); net.restoreAll(); reanalyze(); practiceEvent("restore"); note("knock+restore");
      setBig(true); await wait(50); setBig(false); note("big");
      setMode("play");
      for (var i = 0; i < 400 && !P.over; i++) playKey(i < 25 ? P.cur.action : P.env.direction);
      note("play steps=" + P.log.length + " over=" + !!P.over);
      if (!P.over) throw new Error("play game did not end");
      var dis = P.log.filter(function (r) { return !r.agree; })[0];
      if (dis) { P.replay = dis; setPlayView(); renderExplain(); P.replay = null; setPlayView(); note("replay"); }
      $("ck-assist").click(); playNewGame(); playKey(P.cur.action); playKey(P.cur.action); note("assist");
      P.levelIdx = 3; P.ms = CFG.playLevels[3].ms; renderOpts(); playNewGame(); playKey(3); await wait(800); note("timed steps=" + P.log.length);
      setMode("practice"); manualStep(); note("practice");
      setMode("watch");
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

  /** ?shot=watch|play|practice&steps=N — deterministic state for screenshots. */
  function prepareShot(kind, steps) {
    closeHelp();
    if (kind === "play") {
      setMode("play");
      for (var i = 0; i < steps; i++) playKey(i === steps - 1 ? (P.cur.action + (P.cur.legal[(P.cur.action + 2) % 4] ? 2 : 1)) % 4 : P.cur.action);
    } else {
      if (kind === "practice") setMode("practice");
      setWatchPlaying(false);
      for (var j = 0; j < steps; j++) { if (W.over) { hideBanner(); watchNextGame(); } watchStep(); }
      if (kind === "knock") { toggleKnock("fc1", fcTop1(W.lastRes), true); toggleKnock("conv3", 5, false); }
      $("toast").classList.remove("show");
    }
    $("selftest").textContent = JSON.stringify({ sizes: app.sizes, rh: $("right").clientHeight, rw: $("right").clientWidth, mh: $("main").clientHeight });
  }

  // ================================================================== boot
  function boot() {
    $("app-title").textContent = CFG.title;
    $("app-title-short").textContent = CFG.mobileTitle || CFG.shortTitle;
    document.title = CFG.title;
    renderFormula();
    buildInputRow();
    ["conv1", "conv2", "conv3"].forEach(buildTops);
    renderFooter();
    bindTips(document);
    bindCanvases();
    on(document, "keydown", onKey);
    on(document, "click", onClick);
    on(document, "change", onChange);
    on(document, "input", onInput);
    on(window, "resize", function () { detectTouch(); layout(); });
    on(window, "orientationchange", function () { setTimeout(function () { if (app.alive) { detectTouch(); layout(); } }, 150); });
    app.touchCtl = new window.SnakeTouch($("touchctl"), { onDir: function (d) { playKey(d); }, onPress: onTouchPress });
    var big = store("big") === "1";
    if (big) { document.body.classList.add("big"); $("ck-big").checked = true; }
    detectTouch();
    watchRestart();
    setMode("watch");
    layout();
    // 只读的调试 / 验收入口（tools/browser_check.py 的手机场景断言用）
    window.__app = {
      mode: function () { return app.mode; }, touchp: function () { return app.touchp; }, sheet: function () { return app.sheet; },
      playDir: function () { return P.env ? P.env.direction : null; }, playSteps: function () { return P.env ? P.env.steps : null; },
      playMs: function () { return P.ms; }, playLevel: function () { return P.levelIdx; },
      watchSteps: function () { return W.totalSteps; }, knocked: function () { return net ? net.knockCount() : 0; },
      sizes: function () { return app.sizes; }, kConv: function () { return app.kConv; }
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
