# -*- coding: utf-8 -*-
"""Double DQN 训练贪吃蛇 CNN。

用法: python train.py --max-minutes 15 --grid 12
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 绕开 torch 与 conda/MKL 的 OpenMP 重复加载冲突；不经 run.ps1 启动也不崩

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from snake_env import SnakeEnv
from model import SnakeCNN

CKPT = Path(__file__).parent / "checkpoints"
CKPT.mkdir(exist_ok=True)


class Replay:
    """uint8 量化存储，省内存。"""

    def __init__(self, cap, shape):
        self.cap = cap
        self.s = np.zeros((cap, *shape), dtype=np.uint8)
        self.s2 = np.zeros((cap, *shape), dtype=np.uint8)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.i = 0
        self.full = False

    @staticmethod
    def q(x):                                    # [-1,1] → uint8
        return ((x + 1.0) * 127.5).astype(np.uint8)

    @staticmethod
    def dq(x):
        return x.astype(np.float32) / 127.5 - 1.0

    def push(self, s, a, r, s2, d):
        self.s[self.i] = self.q(s)
        self.s2[self.i] = self.q(s2)
        self.a[self.i] = a
        self.r[self.i] = r
        self.d[self.i] = d
        self.i = (self.i + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, bs, rng):
        idx = rng.integers(0, len(self), bs)
        return (torch.from_numpy(self.dq(self.s[idx])),
                torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.dq(self.s2[idx])),
                torch.from_numpy(self.d[idx]))


def evaluate(net, n, games=20, seed=999):
    env = SnakeEnv(n, seed)
    scores, lens = [], []
    with torch.no_grad():
        for _ in range(games):
            s = env.reset()
            while not env.done:
                q = net(torch.from_numpy(s).unsqueeze(0))[0]
                q = q + torch.tensor(env.legal_mask()).log()   # 掩码回头
                s, _, done, _ = env.step(int(q.argmax()))
            scores.append(env.score)
            lens.append(env.steps)
    return float(np.mean(scores)), float(np.mean(lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=15.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.96)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--replay", type=int, default=50000)
    ap.add_argument("--eps-decay-steps", type=int, default=60000)
    ap.add_argument("--target-sync", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = SnakeEnv(args.grid, args.seed)
    net = SnakeCNN(args.grid, SnakeEnv.CHANNELS)
    tgt = SnakeCNN(args.grid, SnakeEnv.CHANNELS)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = Replay(args.replay, (SnakeEnv.CHANNELS, args.grid, args.grid))
    pyrng = random.Random(args.seed)

    t0 = time.time()
    deadline = t0 + args.max_minutes * 60
    step = 0
    ep = 0
    recent = deque(maxlen=100)
    best = -1.0
    print(f"DQN 训练开始：{args.grid}×{args.grid} 网格", flush=True)

    while time.time() < deadline:
        s = env.reset()
        ep += 1
        while not env.done:
            eps = max(0.05, 1.0 - step / args.eps_decay_steps)
            if pyrng.random() < eps:
                legal = [a for a in range(4) if env.legal_mask()[a] > 0]
                a = pyrng.choice(legal)
            else:
                with torch.no_grad():
                    q = net(torch.from_numpy(s).unsqueeze(0))[0]
                    q = q + torch.tensor(env.legal_mask()).log()
                    a = int(q.argmax())
            s2, r, done, _ = env.step(a)
            buf.push(s, a, r, s2, float(done))
            s = s2
            step += 1

            if len(buf) >= 2000 and step % 3 == 0:
                bs_, ba, br, bs2, bd = buf.sample(args.batch, rng)
                with torch.no_grad():
                    a2 = net(bs2).argmax(1)                       # Double DQN
                    q2 = tgt(bs2).gather(1, a2.unsqueeze(1)).squeeze(1)
                    y = br + args.gamma * (1 - bd) * q2
                qv = net(bs_).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(qv, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step % args.target_sync == 0:
                tgt.load_state_dict(net.state_dict())
            if time.time() > deadline:
                break
        recent.append(env.score)

        if ep % 100 == 0:
            el = time.time() - t0
            avg = np.mean(recent) if recent else 0
            print(f"[{el:5.0f}s] 局数={ep} 步数={step} eps={eps:.2f} "
                  f"近100局均分={avg:.2f} 速度={step / el:.0f}步/s", flush=True)
        if ep % 500 == 0:
            sc, ln = evaluate(net, args.grid, 20)
            print(f"    >>> 贪婪评估: 平均吃食 {sc:.1f}，平均存活 {ln:.0f} 步", flush=True)
            torch.save({"model": net.state_dict(), "n": args.grid,
                        "channels": SnakeEnv.CHANNELS, "avg_score": sc},
                       CKPT / "dqn_snake.pt")
            if sc > best:
                best = sc
                torch.save({"model": net.state_dict(), "n": args.grid,
                            "channels": SnakeEnv.CHANNELS, "avg_score": sc},
                           CKPT / "dqn_snake_best.pt")

    sc, ln = evaluate(net, args.grid, 30)
    torch.save({"model": net.state_dict(), "n": args.grid,
                "channels": SnakeEnv.CHANNELS, "avg_score": sc},
               CKPT / "dqn_snake.pt")
    if sc > best:
        torch.save({"model": net.state_dict(), "n": args.grid,
                    "channels": SnakeEnv.CHANNELS, "avg_score": sc},
                   CKPT / "dqn_snake_best.pt")
    print(f"\n训练完成：{ep} 局 / {step} 步。最终贪婪评估：平均吃食 {sc:.1f}"
          f"（30 局），模型已保存。", flush=True)


if __name__ == "__main__":
    main()
