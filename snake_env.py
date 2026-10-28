# -*- coding: utf-8 -*-
"""贪吃蛇环境：N×N 网格，CNN 友好的多通道状态编码。

状态 (5, N, N)：
  ch0 蛇头(=1)
  ch1 身体亮度梯度：紧邻头部最亮、向尾部渐暗（头尾深浅标记，前端同源渲染）
  ch2 食物(=1)
  ch3 当前方向 dr 平面（整版填充 -1/0/1）
  ch4 当前方向 dc 平面

动作：0=上 1=下 2=左 3=右；选择 180° 回头时保持原方向（反向掩码）。
奖励：吃到 +1；死亡 -1；每步 -0.01；接近/远离食物 ±0.03（距离塑形）；
      饥饿超时（2N² 步未吃到）判死 -0.5，防转圈不死。
"""

import random

import numpy as np

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # 上 下 左 右
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
ACTION_NAMES = ["上", "下", "左", "右"]


class SnakeEnv:
    CHANNELS = 5

    def __init__(self, n=12, seed=None):
        self.n = n
        self.rng = random.Random(seed)

    def reset(self):
        n = self.n
        r, c = n // 2, n // 2
        self.snake = [(r, c), (r, c - 1), (r, c - 2)]   # 头在前
        self.direction = 3                               # 向右
        self.score = 0
        self.steps = 0
        self.hunger = 0
        self._place_food()
        self.done = False
        return self.state()

    def _place_food(self):
        empty = [(r, c) for r in range(self.n) for c in range(self.n)
                 if (r, c) not in self.snake]
        self.food = self.rng.choice(empty) if empty else None

    def state(self):
        n = self.n
        s = np.zeros((self.CHANNELS, n, n), dtype=np.float32)
        L = len(self.snake)
        hr, hc = self.snake[0]
        s[0, hr, hc] = 1.0
        for i, (r, c) in enumerate(self.snake[1:], start=1):
            s[1, r, c] = max(0.15, 1.0 - i / L)          # 渐暗
        if self.food:
            s[2, self.food[0], self.food[1]] = 1.0
        dr, dc = DIRS[self.direction]
        s[3, :, :] = dr
        s[4, :, :] = dc
        return s

    def legal_mask(self):
        """4 维 0/1：180° 回头为非法（被掩码）。"""
        m = [1.0] * 4
        m[OPPOSITE[self.direction]] = 0.0
        return m

    def step(self, action):
        assert not self.done
        if action == OPPOSITE[self.direction]:
            action = self.direction                       # 回头无效，继续直行
        self.direction = action
        dr, dc = DIRS[action]
        hr, hc = self.snake[0]
        nr, nc = hr + dr, hc + dc
        self.steps += 1
        self.hunger += 1

        reward = -0.01
        if self.food:
            old_d = abs(hr - self.food[0]) + abs(hc - self.food[1])
            new_d = abs(nr - self.food[0]) + abs(nc - self.food[1])
            reward += 0.03 if new_d < old_d else -0.03

        body_no_tail = self.snake[:-1]                    # 尾巴本步移开，可进
        if not (0 <= nr < self.n and 0 <= nc < self.n) or (nr, nc) in body_no_tail:
            self.done = True
            return self.state(), -1.0, True, {"reason": "撞墙/撞身"}

        self.snake.insert(0, (nr, nc))
        if (nr, nc) == self.food:
            self.score += 1
            self.hunger = 0
            reward = 1.0
            self._place_food()
            if self.food is None:                         # 占满全场 → 通关
                self.done = True
                return self.state(), 5.0, True, {"reason": "通关！"}
        else:
            self.snake.pop()

        if self.hunger > 2 * self.n * self.n:
            self.done = True
            return self.state(), -0.5, True, {"reason": "饥饿超时"}
        return self.state(), reward, False, {}
