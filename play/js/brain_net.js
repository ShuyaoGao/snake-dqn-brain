/* SnakeCNN (model.py) in plain JavaScript: forward pass + the gradient of the chosen
 * action's Q value w.r.t. every layer, computed by hand-written backpropagation.
 *
 *   input 5xNxN -> conv3x3(32)+ReLU -> conv3x3(64)+ReLU -> conv3x3(64)+ReLU
 *               -> FC 256 + ReLU -> FC 4 (Q values: up, down, left, right)
 *
 * Weights come from window.SNAKE_MODEL (float16, exported by tools/export_model.py).
 * "Knock-out" masks multiply a unit's post-ReLU activation by 0 (conv: whole channel).
 * Locked against PyTorch by tests/test_play_parity.py.
 */
(function (g) {
  "use strict";

  function halfToFloat(h) {
    var s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * f * Math.pow(2, -24);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * Math.pow(2, e - 15);
  }

  function decodeFp16(b64) {
    var bin = (typeof atob === "function") ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    var n = bin.length >> 1, out = new Float32Array(n), lut = new Float32Array(65536);
    for (var i = 0; i < 65536; i++) lut[i] = halfToFloat(i);
    for (var k = 0; k < n; k++) out[k] = lut[bin.charCodeAt(2 * k) | (bin.charCodeAt(2 * k + 1) << 8)];
    return out;
  }

  function BrainNet(payload) {
    var flat = decodeFp16(payload.b64);
    this.meta = payload.meta;
    this.n = payload.meta.n;
    this.w = {};
    this.shapes = {};
    for (var i = 0; i < payload.tensors.length; i++) {
      var t = payload.tensors[i];
      this.w[t.name] = flat.subarray(t.offset, t.offset + t.length);
      this.shapes[t.name] = t.shape;
    }
    this.C = [payload.meta.channels, this.shapes["conv1.weight"][0],
              this.shapes["conv2.weight"][0], this.shapes["conv3.weight"][0]];
    this.H = this.shapes["fc1.weight"][0];
    this.A = this.shapes["fc2.weight"][0];
    this.masks = {
      conv1: new Float32Array(this.C[1]).fill(1), conv2: new Float32Array(this.C[2]).fill(1),
      conv3: new Float32Array(this.C[3]).fill(1), fc1: new Float32Array(this.H).fill(1)
    };
    this.params = 0;
    for (var k in this.w) this.params += this.w[k].length;
  }

  BrainNet.prototype.layerSizes = function () {
    return { conv1: this.C[1], conv2: this.C[2], conv3: this.C[3], fc1: this.H, out: this.A };
  };

  BrainNet.prototype.setKnock = function (layer, idx, off) { this.masks[layer][idx] = off ? 0 : 1; };
  BrainNet.prototype.isKnocked = function (layer, idx) { return this.masks[layer][idx] === 0; };
  BrainNet.prototype.knockCount = function () {
    var c = 0;
    for (var k in this.masks) for (var i = 0; i < this.masks[k].length; i++) if (this.masks[k][i] === 0) c++;
    return c;
  };
  BrainNet.prototype.restoreAll = function () { for (var k in this.masks) this.masks[k].fill(1); };

  // 3x3 conv, padding 1, NxN. z = conv(x) + b; a = relu(z) * mask
  BrainNet.prototype._conv = function (x, cin, w, b, cout, mask) {
    var n = this.n, nn = n * n, z = new Float32Array(cout * nn), a = new Float32Array(cout * nn);
    for (var co = 0; co < cout; co++) {
      var zo = co * nn, bias = b[co];
      for (var p = 0; p < nn; p++) z[zo + p] = bias;
      for (var ci = 0; ci < cin; ci++) {
        var xo = ci * nn, wo = (co * cin + ci) * 9;
        for (var ky = 0; ky < 3; ky++) {
          for (var kx = 0; kx < 3; kx++) {
            var wv = w[wo + ky * 3 + kx];
            if (wv === 0) continue;
            var dy = ky - 1, dx = kx - 1;
            var y0 = dy < 0 ? 1 : 0, y1 = dy > 0 ? n - 1 : n;
            var x0 = dx < 0 ? 1 : 0, x1 = dx > 0 ? n - 1 : n;
            for (var y = y0; y < y1; y++) {
              var zr = zo + y * n, xr = xo + (y + dy) * n + dx;
              for (var xx = x0; xx < x1; xx++) z[zr + xx] += wv * x[xr + xx];
            }
          }
        }
      }
      var m = mask[co];
      for (var q = 0; q < nn; q++) { var v = z[zo + q]; a[zo + q] = v > 0 ? v * m : 0; }
    }
    return { z: z, a: a };
  };

  // gradient of conv input given gradient of conv pre-activation dz
  BrainNet.prototype._convBack = function (dz, cin, w, cout) {
    var n = this.n, nn = n * n, dx = new Float32Array(cin * nn);
    for (var co = 0; co < cout; co++) {
      var zo = co * nn;
      for (var ci = 0; ci < cin; ci++) {
        var xo = ci * nn, wo = (co * cin + ci) * 9;
        for (var ky = 0; ky < 3; ky++) {
          for (var kx = 0; kx < 3; kx++) {
            var wv = w[wo + ky * 3 + kx];
            if (wv === 0) continue;
            var dy = ky - 1, ddx = kx - 1;
            var y0 = dy < 0 ? 1 : 0, y1 = dy > 0 ? n - 1 : n;
            var x0 = ddx < 0 ? 1 : 0, x1 = ddx > 0 ? n - 1 : n;
            for (var y = y0; y < y1; y++) {
              var zr = zo + y * n, xr = xo + (y + dy) * n + ddx;
              for (var xx = x0; xx < x1; xx++) dx[xr + xx] += wv * dz[zr + xx];
            }
          }
        }
      }
    }
    return dx;
  };

  /** Forward only (fast path for rollouts). Returns Float32Array(4) of Q values. */
  BrainNet.prototype.q = function (x) {
    return this.forward(x, null, false).q;
  };

  /** Greedy action with the 180-degree turn masked (Q + log(mask)); ties -> lowest index. */
  BrainNet.argmaxLegal = function (q, legal) {
    var best = -1, bv = -Infinity;
    for (var i = 0; i < q.length; i++) {
      if (!legal[i]) continue;
      if (q[i] > bv) { bv = q[i]; best = i; }
    }
    return best;
  };

  /**
   * Full analysis of one state.
   * legal: [1,1,0,1]-style mask; grad: also backpropagate dQ[action]/d(everything).
   * Returns q, action, activations (a1,a2,a3,h), and if grad: dx (input gradient),
   * ga1..ga3/gh (gradients w.r.t. activations), contrib.{conv1,conv2,conv3,fc1}
   * (activation x gradient summed per unit = contribution to Q[action]) and fcOut
   * (h_j * W2[o, j] for every output o — exact decomposition of the last layer).
   */
  BrainNet.prototype.forward = function (x, legal, grad) {
    var n = this.n, nn = n * n, w = this.w, C = this.C, H = this.H, A = this.A;
    var l1 = this._conv(x, C[0], w["conv1.weight"], w["conv1.bias"], C[1], this.masks.conv1);
    var l2 = this._conv(l1.a, C[1], w["conv2.weight"], w["conv2.bias"], C[2], this.masks.conv2);
    var l3 = this._conv(l2.a, C[2], w["conv3.weight"], w["conv3.bias"], C[3], this.masks.conv3);
    var D = C[3] * nn, W1 = w["fc1.weight"], b1 = w["fc1.bias"], zh = new Float32Array(H), h = new Float32Array(H);
    var a3 = l3.a, mh = this.masks.fc1;
    for (var j = 0; j < H; j++) {
      var s = b1[j], ro = j * D;
      for (var i = 0; i < D; i++) s += W1[ro + i] * a3[i];
      zh[j] = s;
      h[j] = s > 0 ? s * mh[j] : 0;
    }
    var W2 = w["fc2.weight"], b2 = w["fc2.bias"], q = new Float32Array(A);
    for (var o = 0; o < A; o++) {
      var t = b2[o];
      for (var jj = 0; jj < H; jj++) t += W2[o * H + jj] * h[jj];
      q[o] = t;
    }
    var res = { q: q, a1: l1.a, a2: l2.a, a3: a3, h: h };
    legal = legal || [1, 1, 1, 1];
    res.legal = legal;
    res.action = BrainNet.argmaxLegal(q, legal);
    if (!grad) return res;

    var act = res.action;
    // FC layer: exact per-output contributions h_j * W2[o, j]
    var fcOut = [];
    for (o = 0; o < A; o++) {
      var row = new Float32Array(H);
      for (jj = 0; jj < H; jj++) row[jj] = h[jj] * W2[o * H + jj];
      fcOut.push(row);
    }
    // dQ[act]/dh, then through ReLU and mask
    var gh = new Float32Array(H), dzh = new Float32Array(H);
    for (jj = 0; jj < H; jj++) {
      gh[jj] = W2[act * H + jj];
      dzh[jj] = zh[jj] > 0 ? gh[jj] * mh[jj] : 0;
    }
    var ga3 = new Float32Array(D);
    for (j = 0; j < H; j++) {
      var dv = dzh[j];
      if (dv === 0) continue;
      var r0 = j * D;
      for (i = 0; i < D; i++) ga3[i] += W1[r0 + i] * dv;
    }
    var self = this;
    function reluBack(ga, z, mask, ch) {
      var dz = new Float32Array(ch * nn);
      for (var c = 0; c < ch; c++) {
        var m = mask[c];
        if (m === 0) continue;
        for (var p = 0; p < nn; p++) { var k = c * nn + p; dz[k] = z[k] > 0 ? ga[k] * m : 0; }
      }
      return dz;
    }
    var dz3 = reluBack(ga3, l3.z, this.masks.conv3, C[3]);
    var ga2 = self._convBack(dz3, C[2], w["conv3.weight"], C[3]);
    var dz2 = reluBack(ga2, l2.z, this.masks.conv2, C[2]);
    var ga1 = self._convBack(dz2, C[1], w["conv2.weight"], C[2]);
    var dz1 = reluBack(ga1, l1.z, this.masks.conv1, C[1]);
    var dx = self._convBack(dz1, C[0], w["conv1.weight"], C[1]);

    function chContrib(a, ga, ch) {
      var out = new Float32Array(ch);
      for (var c = 0; c < ch; c++) {
        var s2 = 0;
        for (var p = 0; p < nn; p++) s2 += a[c * nn + p] * ga[c * nn + p];
        out[c] = s2;
      }
      return out;
    }
    var fcC = new Float32Array(H);
    for (jj = 0; jj < H; jj++) fcC[jj] = h[jj] * gh[jj];
    res.dx = dx;
    res.ga1 = ga1; res.ga2 = ga2; res.ga3 = ga3; res.gh = gh;
    res.contrib = { input: chContrib(x, dx, C[0]), conv1: chContrib(l1.a, ga1, C[1]),
                    conv2: chContrib(l2.a, ga2, C[2]), conv3: chContrib(a3, ga3, C[3]), fc1: fcC };
    res.fcOut = fcOut;
    // saliency like server.py: |dQ/dx| summed over channels, per cell
    var sal = new Float32Array(nn);
    for (var c = 0; c < C[0]; c++) for (var p = 0; p < nn; p++) sal[p] += Math.abs(dx[c * nn + p]);
    res.saliency = sal;
    return res;
  };

  /** Softmax over legal actions (temperature 1), illegal -> 0. Port of vote_core.softmax_legal. */
  BrainNet.softmaxLegal = function (q, legal) {
    var m = -Infinity, i, out = [0, 0, 0, 0], z = 0;
    for (i = 0; i < 4; i++) if (legal[i] && q[i] > m) m = q[i];
    for (i = 0; i < 4; i++) if (legal[i]) { out[i] = Math.exp(q[i] - m); z += out[i]; }
    for (i = 0; i < 4; i++) out[i] = legal[i] ? out[i] / z : 0;
    return out;
  };

  g.BrainNet = BrainNet;
  g.halfToFloat = halfToFloat;
})(typeof window !== "undefined" ? window : globalThis);
