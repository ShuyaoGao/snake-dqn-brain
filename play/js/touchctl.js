/* 手机竖屏操控区（「我来玩」模式）：左下浮动摇杆 + 右下大圆键。
 * 摇杆：手指按下处就是摇杆中心，中间有死区，死区外按主轴分成上 / 下 / 左 / 右 4 个方向，松手回中；
 *       方向每变一次就通知页面一次（回合制下 = 走一步，限时模式下 = 改变下一步方向）。
 * 大圆键：暂停 / 继续 / 开始 / 下一局（回合制下是「直走一步」），由页面决定含义。
 * 多点触控：摇杆和每个按键各自跟踪自己的 pointerId，可以同时按。按下有视觉反馈 + 轻微振动（不支持时静默）。
 * 防误触：长按菜单、文字选中、双击缩放都关掉；所有手指离开或窗口失焦时全部复位。 */
(function (g) {
  "use strict";
  function vibrate(ms) { try { if (g.navigator && g.navigator.vibrate) g.navigator.vibrate(ms); } catch (e) { /* 不支持时静默 */ } }

  /** root = #touchctl；opts.onDir(dir) 方向 0=上 1=下 2=左 3=右；opts.onPress(key) 按键。 */
  function SnakeTouch(root, opts) {
    this.root = root; this.opts = opts || {};
    this.zone = root.querySelector("#stick-zone"); this.base = root.querySelector("#stick-base"); this.knob = root.querySelector("#stick-knob");
    this.stickId = null; this.origin = null; this.dir = -1; this.bound = [];
    this.R = 52;        // 摇杆半径（CSS px）
    this.dead = 0.35;   // 死区 = 半径的 35%
    var self = this;
    function on(el, ev, fn, o) { el.addEventListener(ev, fn, o); self.bound.push([el, ev, fn, o]); }
    on(root, "contextmenu", function (e) { e.preventDefault(); });
    // 只在摇杆区和大圆键上拦掉原生触摸（防滚动 / 缩放 / 长按）；顶部一行的普通按钮要保留轻点产生的 click
    var isCtl = function (e) { return e.target && e.target.closest && e.target.closest("#stick-zone, [data-tk]"); };
    on(root, "touchstart", function (e) { if (e.cancelable && isCtl(e)) e.preventDefault(); }, { passive: false });
    on(root, "touchmove", function (e) { if (e.cancelable && isCtl(e)) e.preventDefault(); }, { passive: false });
    // 保险：所有手指都离开屏幕时摇杆回中、按键全部松开（个别设备会丢失某根手指的抬起事件）
    var allUp = function (e) { if (!e.touches || e.touches.length === 0) self.reset(); };
    on(root, "touchend", allUp); on(root, "touchcancel", allUp);
    on(g, "blur", function () { self.reset(); });
    // ---- 浮动摇杆 ----
    on(this.zone, "pointerdown", function (e) {
      if (self.stickId !== null) return;
      self.stickId = e.pointerId;
      try { self.zone.setPointerCapture(e.pointerId); } catch (err) { /* 合成事件可能没有捕获能力 */ }
      var r = self.zone.getBoundingClientRect();
      self.origin = { x: e.clientX, y: e.clientY };
      self.base.classList.add("live");
      self.base.style.left = (e.clientX - r.left) + "px"; self.base.style.top = (e.clientY - r.top) + "px";
      self.knob.style.transition = "none";
      self._move(e.clientX, e.clientY);
      e.preventDefault();
    });
    on(this.zone, "pointermove", function (e) { if (e.pointerId === self.stickId) { self._move(e.clientX, e.clientY); e.preventDefault(); } });
    var end = function (e) { if (e.pointerId === self.stickId) self._release(); };
    on(this.zone, "pointerup", end); on(this.zone, "pointercancel", end); on(this.zone, "lostpointercapture", end);
    // ---- 按键（每个键独立跟踪按下它的手指） ----
    root.querySelectorAll("[data-tk]").forEach(function (btn) {
      var k = btn.getAttribute("data-tk"), pid = null;
      on(btn, "pointerdown", function (e) {
        e.preventDefault(); pid = e.pointerId;
        try { btn.setPointerCapture(e.pointerId); } catch (err) { /* 忽略 */ }
        btn.classList.add("on"); vibrate(8);
        if (self.opts.onPress) self.opts.onPress(k);
      });
      var up = function (e) { if (e.pointerId !== pid) return; pid = null; btn.classList.remove("on"); };
      on(btn, "pointerup", up); on(btn, "pointercancel", up); on(btn, "lostpointercapture", up);
    });
  }
  SnakeTouch.prototype._move = function (x, y) {
    var dx = x - this.origin.x, dy = y - this.origin.y, d = Math.sqrt(dx * dx + dy * dy), R = this.R;
    var k = d > R ? R / d : 1;
    this.knob.style.transform = "translate(" + (dx * k) + "px," + (dy * k) + "px)";
    var dir = -1;
    if (d > R * this.dead) dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 3 : 2) : (dy > 0 ? 1 : 0);   // 主轴决定方向（屏幕坐标 y 向下）
    this._set(dir);
  };
  SnakeTouch.prototype._set = function (dir) {
    if (dir === this.dir) return;
    this.dir = dir;
    this.root.setAttribute("data-dir", dir);
    if (dir >= 0) { vibrate(4); if (this.opts.onDir) this.opts.onDir(dir); }
  };
  SnakeTouch.prototype._release = function () {
    this.stickId = null; this._set(-1);
    this.knob.style.transition = "transform .12s ease-out"; this.knob.style.transform = "translate(0px,0px)";   // 回中动画
    this.base.classList.remove("live");
  };
  SnakeTouch.prototype.reset = function () {
    this._release();
    this.root.querySelectorAll("[data-tk].on").forEach(function (b) { b.classList.remove("on"); });   // 只松开大圆键，不碰顶部一行的档位 / 开关
  };
  SnakeTouch.prototype.unbind = function () {
    this.bound.forEach(function (b) { b[0].removeEventListener(b[1], b[2], b[3]); });
    this.bound = [];
  };
  g.SnakeTouch = SnakeTouch;
})(typeof window !== "undefined" ? window : globalThis);
