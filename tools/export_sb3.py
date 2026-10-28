# -*- coding: utf-8 -*-
"""Export a Stable-Baselines3 PPO / MaskablePPO policy (policy.pth inside the model .zip) to
play/model/<name>_fp16.js — float16 weights, base64, plus the network spec the page and
brains.py need. No PyTorch required: the checkpoint is read with a small torch-free unpickler.

    python -m tools.export_sb3 ray   path/to/AI-Snake-Game/model/model.zip
    python -m tools.export_sb3 linyi path/to/snake-ai/main/trained_models_cnn/ppo_snake_final.zip
    python -m tools.export_sb3 linyi_mlp path/to/snake-ai/main/trained_models_mlp/ppo_snake_final.zip

The three presets describe the original projects' observation wrappers (board 12x12, image scale,
action order 0=UP 1=LEFT 2=RIGHT 3=DOWN, action mask) — see brains.py and THIRD_PARTY.md.
"""

import base64
import io
import json
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "play" / "model"

# brain action index (0=UP 1=LEFT 2=RIGHT 3=DOWN in both upstream projects) -> env action (0=up 1=down 2=left 3=right)
UPSTREAM_ACTION_MAP = [0, 2, 3, 1]

PRESETS = {
    "ray": {"id": "ppo_ray", "file": "ppo_cnn_ray_fp16.js", "obs": "ray_cnn", "n": 12,
            "input": {"c": 3, "h": 60, "w": 60}, "mask": "none",
            "source": {"project": "Part-time-Ray/AI-Snake-Game", "url": "https://github.com/Part-time-Ray/AI-Snake-Game",
                       "file": "model/model.zip", "license": "MIT", "algorithm": "PPO (Stable-Baselines3 2.3.2), CnnPolicy (NatureCNN)",
                       "reported": "README: 12x12 board, average snake length 74 after 50M steps, 98 after 100M steps"}},
    "linyi": {"id": "ppo_linyi", "file": "ppo_cnn_linyi_fp16.js", "obs": "linyi_cnn", "n": 12,
              "input": {"c": 3, "h": 84, "w": 84}, "mask": "safe",
              "source": {"project": "linyiLYi/snake-ai", "url": "https://github.com/linyiLYi/snake-ai",
                         "file": "main/trained_models_cnn/ppo_snake_final.zip", "license": "Apache-2.0",
                         "algorithm": "MaskablePPO (sb3-contrib, Stable-Baselines3 1.8.0), CnnPolicy (NatureCNN)",
                         "reported": "README: trained 100M steps; no numeric score published"}},
    "linyi_mlp": {"id": "ppo_mlp_linyi", "file": "ppo_mlp_linyi_fp16.js", "obs": "linyi_mlp", "n": 12,
                  "input": {"c": 1, "h": 12, "w": 12}, "mask": "safe",
                  "source": {"project": "linyiLYi/snake-ai", "url": "https://github.com/linyiLYi/snake-ai",
                             "file": "main/trained_models_mlp/ppo_snake_final.zip", "license": "Apache-2.0",
                             "algorithm": "MaskablePPO (sb3-contrib, Stable-Baselines3 1.8.0), MlpPolicy (64-64)",
                             "reported": "README: trained 100M steps; CNN agent scores higher on average"}},
}


# ---------------------------------------------------------------- torch-free checkpoint reader
DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
          "LongStorage": np.int64, "IntStorage": np.int32, "ByteStorage": np.uint8, "BoolStorage": np.bool_}


def read_state_dict(zip_path, member="policy.pth"):
    """Return OrderedDict name -> np.ndarray from the policy.pth inside an SB3 model zip."""
    outer = zipfile.ZipFile(zip_path)
    inner_bytes = outer.read(member) if member in outer.namelist() else Path(zip_path).read_bytes()
    z = zipfile.ZipFile(io.BytesIO(inner_bytes))
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl[: -len("data.pkl")]

    class StorageRef:
        def __init__(self, dtype, key):
            self.dtype, self.key, self._arr = dtype, key, None

        def arr(self):
            if self._arr is None:
                self._arr = np.frombuffer(z.read(prefix + "data/" + self.key), dtype=self.dtype)
            return self._arr

    def rebuild(storage, offset, size, stride, *rest):
        a = storage.arr()
        size, stride = tuple(size), tuple(stride)
        if not size:
            return np.array(a[offset], dtype=a.dtype)
        it = a.dtype.itemsize
        return np.lib.stride_tricks.as_strided(a[offset:], shape=size, strides=tuple(s * it for s in stride)).copy()

    class Storage:
        def __init__(self, name):
            self.name = name

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            if name.endswith("Storage") and module.startswith("torch"):
                return Storage(name)
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            raise pickle.UnpicklingError("refusing %s.%s" % (module, name))

        def persistent_load(self, pid):
            typename, storage_type, key = pid[0], pid[1], pid[2]
            assert typename == "storage"
            return StorageRef(DTYPES[storage_type.name], key)

    return U(io.BytesIO(z.read(pkl))).load()


# ---------------------------------------------------------------- spec + tensors
def conv_out(hin, k, s, p=0):
    return (hin + 2 * p - k) // s + 1


def build(preset, sd):
    P = PRESETS[preset]
    tensors, spec = OrderedDict(), {"obs": P["obs"], "n": P["n"], "input": P["input"], "trunk": [], "pi": [], "v": [],
                                    "actionMap": UPSTREAM_ACTION_MAP, "mask": P["mask"], "out": "logits"}
    c, h, w = P["input"]["c"], P["input"]["h"], P["input"]["w"]
    if "pi_features_extractor.cnn.0.weight" in sd or "features_extractor.cnn.0.weight" in sd:
        fx = "pi_features_extractor." if "pi_features_extractor.cnn.0.weight" in sd else "features_extractor."
        for other in ("features_extractor.", "vf_features_extractor."):
            for k in sd:
                if k.startswith(other) and fx + k[len(other):] in sd:
                    assert np.array_equal(sd[k], sd[fx + k[len(other):]]), "feature extractor is not shared: " + k
        # NatureCNN: conv 8/4 -> conv 4/2 -> conv 3/1 -> flatten -> linear 512, ReLU everywhere
        cur_c, cur_h, cur_w = c, h, w
        for name, idx, k, s in (("conv1", 0, 8, 4), ("conv2", 2, 4, 2), ("conv3", 4, 3, 1)):
            wt, b = sd[fx + "cnn.%d.weight" % idx], sd[fx + "cnn.%d.bias" % idx]
            assert wt.shape[2] == k and wt.shape[1] == cur_c, (name, wt.shape)
            ho, wo = conv_out(cur_h, k, s), conv_out(cur_w, k, s)
            spec["trunk"].append({"name": name, "kind": "conv", "cin": cur_c, "cout": int(wt.shape[0]), "k": k, "s": s, "p": 0,
                                  "hin": cur_h, "win": cur_w, "hout": ho, "wout": wo})
            tensors[name + ".weight"], tensors[name + ".bias"] = wt, b
            cur_c, cur_h, cur_w = int(wt.shape[0]), ho, wo
        wt, b = sd[fx + "linear.0.weight"], sd[fx + "linear.0.bias"]
        assert wt.shape[1] == cur_c * cur_h * cur_w, (wt.shape, cur_c, cur_h, cur_w)
        spec["trunk"].append({"name": "fc1", "kind": "fc", "in": int(wt.shape[1]), "out": int(wt.shape[0])})
        tensors["fc1.weight"], tensors["fc1.bias"] = wt, b
        feat = int(wt.shape[0])
        assert not any(k.startswith("mlp_extractor.") for k in sd), "unexpected mlp_extractor layers with a CNN"
        spec["pi"].append({"name": "pi_out", "kind": "fc", "in": feat, "out": 4})
        tensors["pi_out.weight"], tensors["pi_out.bias"] = sd["action_net.weight"], sd["action_net.bias"]
        spec["v"].append({"name": "v_out", "kind": "fc", "in": feat, "out": 1})
        tensors["v_out.weight"], tensors["v_out.bias"] = sd["value_net.weight"], sd["value_net.bias"]
    else:
        # MlpPolicy: Flatten -> policy_net (64, 64) -> action_net; value_net branch alike
        d = c * h * w
        for branch, prefix, out_key, out_name in (("pi", "mlp_extractor.policy_net.", "action_net", "pi_out"),
                                                   ("v", "mlp_extractor.value_net.", "value_net", "v_out")):
            cur, i, j = d, 0, 1
            while prefix + "%d.weight" % i in sd:
                wt, b = sd[prefix + "%d.weight" % i], sd[prefix + "%d.bias" % i]
                assert wt.shape[1] == cur
                name = "%s%d" % (branch, j)
                spec[branch].append({"name": name, "kind": "fc", "in": cur, "out": int(wt.shape[0])})
                tensors[name + ".weight"], tensors[name + ".bias"] = wt, b
                cur, i, j = int(wt.shape[0]), i + 2, j + 1
            wt, b = sd[out_key + ".weight"], sd[out_key + ".bias"]
            spec[branch].append({"name": out_name, "kind": "fc", "in": cur, "out": int(wt.shape[0])})
            tensors[out_name + ".weight"], tensors[out_name + ".bias"] = wt, b
    assert spec["pi"][-1]["out"] == 4
    return spec, tensors


def encode(tensors):
    table, chunks, off = [], [], 0
    for name, a in tensors.items():
        a16 = np.ascontiguousarray(a, dtype=np.float32).astype("<f2")
        table.append({"name": name, "shape": list(a16.shape), "offset": off, "length": int(a16.size)})
        chunks.append(a16.tobytes())
        off += int(a16.size)
    return table, base64.b64encode(b"".join(chunks)).decode("ascii")


def export(preset, zip_path, out=None):
    P = PRESETS[preset]
    sd = read_state_dict(zip_path)
    spec, tensors = build(preset, sd)
    table, b64 = encode(tensors)
    params = sum(t["length"] for t in table)
    meta = {"id": P["id"], "source": P["source"], "params": int(params), "dtype": "float16",
            "n": P["n"], "obs": P["obs"], "exported_from": Path(zip_path).name}
    payload = {"format": "fp16-b64-v2", "meta": meta, "spec": spec, "tensors": table, "b64": b64}
    out = Path(out) if out else MODEL_DIR / P["file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    head = ("/* Generated by tools/export_sb3.py -- do not edit.\n"
            " * Source: %s (%s), file %s, license %s.\n"
            " * Weights converted to float16; network spec + observation format embedded below (see brains.py). */\n"
            % (P["source"]["project"], P["source"]["url"], P["source"]["file"], P["source"]["license"]))
    text = head + 'window.SNAKE_BRAINS = window.SNAKE_BRAINS || {};\nwindow.SNAKE_BRAINS["%s"] = ' % P["id"] + \
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
    out.write_text(text, encoding="utf-8")
    return out, params, spec


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in PRESETS:
        print(__doc__)
        return 2
    out, params, spec = export(argv[0], argv[1], argv[2] if len(argv) > 2 else None)
    print("wrote %s: %s params, %s bytes" % (out.relative_to(ROOT) if out.is_relative_to(ROOT) else out, f"{params:,}", f"{out.stat().st_size:,}"))
    for L in spec["trunk"] + spec["pi"]:
        print("  ", L)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
