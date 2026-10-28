/* Python-compatible random.Random (MT19937) — only what snake_env.py uses.
 *
 * new PyRandom(seed) reproduces CPython's `random.Random(seed)` for a non-negative
 * integer seed (< 2**53): same init_by_array seeding, same genrand_uint32 stream,
 * `random()` and `randbelow(n)` (= what `choice(seq)` uses internally).
 * This lets the browser replay exactly the same food positions as SnakeEnv(n, seed).
 * Locked by tests/test_play_parity.py::test_pyrandom_matches_cpython.
 */
(function (g) {
  "use strict";
  var N = 624, M = 397;

  function PyRandom(seed) {
    this.mt = new Uint32Array(N);
    this.mti = N + 1;
    if (seed === undefined || seed === null) {
      seed = Math.floor(Math.random() * 4294967296);
    }
    seed = Math.abs(Math.floor(seed));
    var lo = seed % 4294967296, hi = Math.floor(seed / 4294967296);
    this.initByArray(hi > 0 ? [lo >>> 0, hi >>> 0] : [lo >>> 0]);
  }

  PyRandom.prototype.initGenrand = function (s) {
    var mt = this.mt;
    mt[0] = s >>> 0;
    for (var i = 1; i < N; i++) {
      var p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, p) + i) >>> 0;
    }
    this.mti = N;
  };

  PyRandom.prototype.initByArray = function (key) {
    var mt = this.mt, i = 1, j = 0, k, len = key.length;
    this.initGenrand(19650218);
    for (k = Math.max(N, len); k > 0; k--) {
      var p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(p, 1664525)) >>> 0) + key[j] + j;
      mt[i] = mt[i] >>> 0;
      i++; j++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= len) j = 0;
    }
    for (k = N - 1; k > 0; k--) {
      var q = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (((mt[i] ^ Math.imul(q, 1566083941)) >>> 0) - i) >>> 0;
      i++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
  };

  PyRandom.prototype.genrandUint32 = function () {
    var y, mt = this.mt, kk;
    if (this.mti >= N) {
      for (kk = 0; kk < N - M; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + M] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      }
      for (; kk < N - 1; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + (M - N)] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      }
      y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff);
      mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
      this.mti = 0;
    }
    y = mt[this.mti++];
    y ^= (y >>> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >>> 18);
    return y >>> 0;
  };

  /** random.random(): 53-bit float in [0, 1). */
  PyRandom.prototype.random = function () {
    var a = this.genrandUint32() >>> 5, b = this.genrandUint32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  };

  /** getrandbits(k) for 0 <= k <= 32. */
  PyRandom.prototype.getrandbits = function (k) {
    if (k <= 0) return 0;
    return this.genrandUint32() >>> (32 - k);
  };

  /** _randbelow_with_getrandbits(n): uniform int in [0, n). */
  PyRandom.prototype.randbelow = function (n) {
    if (!n) return 0;
    var k = 32 - Math.clz32(n);
    var r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  };

  /** random.choice(seq) */
  PyRandom.prototype.choice = function (seq) {
    if (!seq.length) throw new Error("Cannot choose from an empty sequence");
    return seq[this.randbelow(seq.length)];
  };

  g.PyRandom = PyRandom;
})(typeof window !== "undefined" ? window : globalThis);
