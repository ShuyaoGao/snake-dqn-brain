/* Generic policy network in plain JavaScript: forward pass + the gradient of the chosen output
 * w.r.t. every layer, by hand-written backpropagation (no libraries, runs from file://).
 *
 * A model file (play/model/*.js) carries float16 weights (base64) and a `spec`:
 *   input  {c, h, w}
 *   trunk  conv / fc layers, ReLU after each            (shared by the policy and the value head)
 *   pi     fc layers; ReLU after all but the last        (last = 4 action scores: logits or Q values)
 *   v      fc layers; ReLU after all but the last        (last = 1 state value; may be empty)
 *   actionMap  brain action index -> env action (0=up 1=down 2=left 3=right)
 *   mask   "turn" | "safe" | "none"   the action mask the model was trained with
 * The original DQN export (format fp16-b64-v1) only has n / channels; its spec is rebuilt here.
 * "Knock-out" multiplies a unit's post-ReLU activation by 0 (conv: a whole channel).
 * Locked against the NumPy reference brains.py by tests/test_play_parity.py.
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

  /** Spec for the v1 DQN export: 3 x conv3x3(pad 1) -> fc 256 -> fc 4 (Q values). */
  function specFromV1(meta) {
    var n = meta.n, ch = meta.channels;
    function conv(name, cin, cout) { return { name: name, kind: "conv", cin: cin, cout: cout, k: 3, s: 1, p: 1, hin: n, win: n, hout: n, wout: n }; }
    return { obs: "dqn5", n: n, input: { c: ch, h: n, w: n },
             trunk: [conv("conv1", ch, 32), conv("conv2", 32, 64), conv("conv3", 64, 64), { name: "fc1", kind: "fc", "in": 64 * n * n, out: 256 }],
             pi: [{ name: "fc2", kind: "fc", "in": 256, out: 4 }], v: [], actionMap: [0, 1, 2, 3], mask: "turn", out: "q" };
  }

  function BrainNet(payload) {
    var flat = decodeFp16(payload.b64);
    this.meta = payload.meta || {};
    this.spec = payload.spec || specFromV1(payload.meta);
    this.n = this.spec.n;
    this.w = {};
    this.shapes = {};
    for (var i = 0; i < payload.tensors.length; i++) {
      var t = payload.tensors[i];
      this.w[t.name] = flat.subarray(t.offset, t.offset + t.length);
      this.shapes[t.name] = t.shape;
    }
    this.params = 0;
    for (var k in this.w) this.params += this.w[k].length;
    // layers that take part in the policy decision, in order; `hidden` = ReLU + knock-out mask
    this.layers = [];
    var sp = this.spec, j;
    for (j = 0; j < sp.trunk.length; j++) this.layers.push(withMeta(sp.trunk[j], true));
    for (j = 0; j < sp.pi.length; j++) this.layers.push(withMeta(sp.pi[j], j < sp.pi.length - 1));
    this.vlayers = [];
    for (j = 0; j < sp.v.length; j++) this.vlayers.push(withMeta(sp.v[j], j < sp.v.length - 1));
    this.masks = {};
    for (j = 0; j < this.layers.length; j++) {
      var L = this.layers[j];
      if (L.hidden) this.masks[L.name] = new Float32Array(L.units).fill(1);
    }
    this.outLayer = this.layers[this.layers.length - 1];
    this.A = this.outLayer.units;
  }

  function withMeta(L, hidden) {
    var o = {};
    for (var k in L) o[k] = L[k];
    o.hidden = hidden;
    o.units = L.kind === "conv" ? L.cout : L.out;             // knock-out granularity: conv channel / fc unit
    o.size = L.kind === "conv" ? L.cout * L.hout * L.wout : L.out;
    o.mapH = L.kind === "conv" ? L.hout : 1;
    o.mapW = L.kind === "conv" ? L.wout : 1;
    return o;
  }

  BrainNet.prototype.hiddenLayers = function () { return this.layers.filter(function (L) { return L.hidden; }); };
  BrainNet.prototype.layer = function (name) {
    for (var i = 0; i < this.layers.length; i++) if (this.layers[i].name === name) return this.layers[i];
    return null;
  };
  BrainNet.prototype.setKnock = function (layer, idx, off) { if (this.masks[layer]) this.masks[layer][idx] = off ? 0 : 1; };
  BrainNet.prototype.isKnocked = function (layer, idx) { return !!this.masks[layer] && this.masks[layer][idx] === 0; };
  BrainNet.prototype.knockCount = function () {
    var c = 0;
    for (var k in this.masks) for (var i = 0; i < this.masks[k].length; i++) if (this.masks[k][i] === 0) c++;
    return c;
  };
  BrainNet.prototype.restoreAll = function () { for (var k in this.masks) this.masks[k].fill(1); };

  // ---- conv (kernel k, stride s, zero padding p). z = conv(x) + b ; a = relu(z) * mask (if hidden)
  BrainNet.prototype._conv = function (L, x, mask) {
    var cin = L.cin, cout = L.cout, k = L.k, s = L.s, p = L.p, hin = L.hin, win = L.win, ho = L.hout, wo = L.wout;
    var w = this.w[L.name + ".weight"], b = this.w[L.name + ".bias"];
    var hw = ho * wo, z = new Float32Array(cout * hw), a = new Float32Array(cout * hw);
    var ihw = hin * win;
    for (var co = 0; co < cout; co++) {
      var zo = co * hw, bias = b[co];
      for (var q = 0; q < hw; q++) z[zo + q] = bias;
      for (var ci = 0; ci < cin; ci++) {
        var xo = ci * ihw, wo0 = (co * cin + ci) * k * k;
        for (var ky = 0; ky < k; ky++) {
          for (var kx = 0; kx < k; kx++) {
            var wv = w[wo0 + ky * k + kx];
            if (wv === 0) continue;
            for (var oy = 0; oy < ho; oy++) {
              var iy = oy * s + ky - p;
              if (iy < 0 || iy >= hin) continue;
              var zr = zo + oy * wo, xr = xo + iy * win;
              for (var ox = 0; ox < wo; ox++) {
                var ix = ox * s + kx - p;
                if (ix < 0 || ix >= win) continue;
                z[zr + ox] += wv * x[xr + ix];
              }
            }
          }
        }
      }
      var m = mask ? mask[co] : 1;
      if (L.hidden) for (q = 0; q < hw; q++) { var v = z[zo + q]; a[zo + q] = v > 0 ? v * m : 0; }
      else for (q = 0; q < hw; q++) a[zo + q] = z[zo + q];
    }
    return { z: z, a: a };
  };

  // gradient of the conv input given dz (gradient of the pre-activation)
  BrainNet.prototype._convBack = function (L, dz) {
    var cin = L.cin, cout = L.cout, k = L.k, s = L.s, p = L.p, hin = L.hin, win = L.win, ho = L.hout, wo = L.wout;
    var w = this.w[L.name + ".weight"], hw = ho * wo, ihw = hin * win, dx = new Float32Array(cin * ihw);
    for (var co = 0; co < cout; co++) {
      var zo = co * hw;
      for (var ci = 0; ci < cin; ci++) {
        var xo = ci * ihw, wo0 = (co * cin + ci) * k * k;
        for (var ky = 0; ky < k; ky++) {
          for (var kx = 0; kx < k; kx++) {
            var wv = w[wo0 + ky * k + kx];
            if (wv === 0) continue;
            for (var oy = 0; oy < ho; oy++) {
              var iy = oy * s + ky - p;
              if (iy < 0 || iy >= hin) continue;
              var zr = zo + oy * wo, xr = xo + iy * win;
              for (var ox = 0; ox < wo; ox++) {
                var ix = ox * s + kx - p;
                if (ix < 0 || ix >= win) continue;
                dx[xr + ix] += wv * dz[zr + ox];
              }
            }
          }
        }
      }
    }
    return dx;
  };

  BrainNet.prototype._fc = function (L, x, mask) {
    var W = this.w[L.name + ".weight"], b = this.w[L.name + ".bias"], D = L["in"], H = L.out;
    var z = new Float32Array(H), a = new Float32Array(H);
    for (var j = 0; j < H; j++) {
      var sum = b[j], ro = j * D;
      for (var i = 0; i < D; i++) sum += W[ro + i] * x[i];
      z[j] = sum;
      a[j] = L.hidden ? (sum > 0 ? sum * (mask ? mask[j] : 1) : 0) : sum;
    }
    return { z: z, a: a };
  };

  BrainNet.prototype._fcBack = function (L, dz) {
    var W = this.w[L.name + ".weight"], D = L["in"], H = L.out, dx = new Float32Array(D);
    for (var j = 0; j < H; j++) {
      var dv = dz[j];
      if (dv === 0) continue;
      var ro = j * D;
      for (var i = 0; i < D; i++) dx[i] += W[ro + i] * dv;
    }
    return dx;
  };

  BrainNet.prototype._run = function (L, x) {
    return L.kind === "conv" ? this._conv(L, x, this.masks[L.name]) : this._fc(L, x, this.masks[L.name]);
  };

  /** Greedy choice among allowed actions (mask > 0); ties -> lowest index. */
  BrainNet.argmaxMasked = function (scores, mask) {
    var best = -1, bv = -Infinity;
    for (var i = 0; i < scores.length; i++) {
      if (mask && !mask[i]) continue;
      if (scores[i] > bv) { bv = scores[i]; best = i; }
    }
    return best;
  };

  /** Softmax over allowed actions (temperature 1), disallowed -> 0. */
  BrainNet.softmaxMasked = function (scores, mask) {
    var m = -Infinity, i, out = new Array(scores.length), z = 0;
    for (i = 0; i < scores.length; i++) { out[i] = 0; if ((!mask || mask[i]) && scores[i] > m) m = scores[i]; }
    for (i = 0; i < scores.length; i++) if (!mask || mask[i]) { out[i] = Math.exp(scores[i] - m); z += out[i]; }
    for (i = 0; i < scores.length; i++) out[i] = (!mask || mask[i]) ? out[i] / z : 0;
    return out;
  };

  /** Sample an index from probabilities with a uniform u in [0,1) (cumulative sum, same as brains.py). */
  BrainNet.sample = function (probs, u) {
    var c = 0, last = probs.length - 1;
    for (var i = 0; i < probs.length; i++) {
      if (probs[i] <= 0) continue;
      c += probs[i];
      last = i;
      if (u < c) return i;
    }
    return last;
  };

  /**
   * Full analysis of one observation.
   *   x     Float32Array input (c*h*w)
   *   mask  4 numbers in BRAIN action order (1 = allowed)
   *   grad  also backpropagate d scores[action] / d(everything)
   * Returns { q (scores, Float32Array 4), value (number|null), action (brain index, greedy), probs,
   *           acts: {layerName: {a, z}}, h (last hidden), a1 (first layer), and if grad:
   *           dx, contrib: {input, <each hidden layer>}, saliency (per input pixel, |dx| summed over channels),
   *           fcOut (per output o: h_j * W[o, j] for the last hidden layer) }.
   */
  BrainNet.prototype.forward = function (x, mask, grad) {
    var layers = this.layers, acts = {}, cur = x, i, L;
    for (i = 0; i < layers.length; i++) {
      L = layers[i];
      var r = this._run(L, cur);
      acts[L.name] = r;
      cur = r.a;
    }
    var scores = cur, value = null;
    if (this.vlayers.length) {
      var cv = this.spec.trunk.length ? acts[this.spec.trunk[this.spec.trunk.length - 1].name].a : x;
      for (i = 0; i < this.vlayers.length; i++) cv = this._run(this.vlayers[i], cv).a;
      value = cv[0];
    }
    mask = mask || [1, 1, 1, 1];
    var action = BrainNet.argmaxMasked(scores, mask);
    var lastHidden = layers.length >= 2 ? layers[layers.length - 2] : null;
    var res = { q: scores, value: value, action: action, mask: mask, probs: BrainNet.softmaxMasked(scores, mask), acts: acts,
                h: lastHidden ? acts[lastHidden.name].a : x, a1: acts[layers[0].name].a };
    if (!grad) return res;

    // ---- backward: d scores[action]
    var contrib = {}, gUp = null;             // gUp = gradient w.r.t. the current layer's output activation
    for (i = layers.length - 1; i >= 0; i--) {
      L = layers[i];
      var z = acts[L.name].z, dz;
      if (i === layers.length - 1) {
        dz = new Float32Array(z.length);
        dz[action] = 1;
      } else {
        var m = this.masks[L.name], a = acts[L.name].a;
        dz = new Float32Array(z.length);
        var c = new Float32Array(L.units), per = L.size / L.units;
        for (var k = 0; k < z.length; k++) {
          var u = Math.floor(k / per);
          dz[k] = z[k] > 0 ? gUp[k] * m[u] : 0;
          c[u] += a[k] * gUp[k];
        }
        contrib[L.name] = c;
      }
      gUp = L.kind === "conv" ? this._convBack(L, dz) : this._fcBack(L, dz);
    }
    var cin = this.spec.input.c, hw = this.spec.input.h * this.spec.input.w, cI = new Float32Array(cin), sal = new Float32Array(hw);
    for (var ch = 0; ch < cin; ch++) for (var q = 0; q < hw; q++) { cI[ch] += x[ch * hw + q] * gUp[ch * hw + q]; sal[q] += Math.abs(gUp[ch * hw + q]); }
    contrib.input = cI;
    res.dx = gUp;
    res.contrib = contrib;
    res.saliency = sal;
    if (lastHidden) {
      var W2 = this.w[this.outLayer.name + ".weight"], H = lastHidden.size, h = res.h, fcOut = [];
      for (var o = 0; o < this.A; o++) {
        var row = new Float32Array(H);
        for (var jj = 0; jj < H; jj++) row[jj] = h[jj] * W2[o * H + jj];
        fcOut.push(row);
      }
      res.fcOut = fcOut;
    }
    return res;
  };

  g.BrainNet = BrainNet;
  g.halfToFloat = halfToFloat;
})(typeof window !== "undefined" ? window : globalThis);
