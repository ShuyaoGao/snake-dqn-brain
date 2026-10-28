# 第三方组件与来源说明 / Third-party notices

## 网页 play/（下载后双击即玩的部分）

| 内容 | 说明 |
|---|---|
| 第三方 JavaScript 库 | **没有**。页面只用浏览器自带的 Canvas / DOM，全部代码在 `play/js/` |
| 字体 | **不内嵌任何字体**。只按名字引用操作系统自带字体：Windows 为 Microsoft YaHei，macOS 为 PingFang SC，其余回退到 `system-ui`。字体文件本身不随本仓库分发 |
| 表情符号（🐍 等） | Unicode 字符，由操作系统的表情字体显示，不随本仓库分发 |
| 图片、音效 | 没有。棋盘、动画演示都由代码实时绘制；页面没有声音 |
| 模型权重 `play/model/snake_cnn_fp16.js` | 由本仓库的 `checkpoints/dqn_snake_best.pt` 转成 float16 导出（`tools/export_model.py`），作者自己用本仓库 `train.py` 训练，许可证同代码（MIT） |
| 网络请求 | 没有。不用 CDN、不用 fetch / XHR、不用 service worker，`file://` 下即可运行 |

## Python（只在想自己训练 / 评估 / 跑测试时需要，不随仓库分发）

| 包 | 用途 | 许可证（来自 PyPI 元数据） |
|---|---|---|
| PyTorch (`torch`) | 训练与评估 | BSD-3-Clause 风格（PyTorch 的 LICENSE）；wheel 元数据还列出所含组件的 Apache-2.0、BSD-2-Clause、BSL-1.0、MIT 等 |
| NumPy (`numpy`) | 环境状态、经验回放 | BSD-3-Clause（另含 0BSD、MIT、Zlib、CC0-1.0 组件） |
| pytest | 测试 | MIT |

## 测试工具（不随仓库分发）

| 工具 | 用途 |
|---|---|
| Node.js | `tests/js/harness.js`：在 Node 里运行 `play/js/*.js`，与 Python / PyTorch 逐项比对 |
| 本机已安装的 Microsoft Edge 或 Google Chrome | `tools/browser_check.py`：无头打开页面跑自检和截图；不下载任何浏览器 |

## 文本

- `LICENSE-docs` 中的 CC BY 4.0 法律文本来自 Creative Commons（creativecommons.org），按其规定原样附上。

## 代码来源

- 全部代码为作者原创（由 Claude Code 辅助编写，经作者审阅），没有从外部项目复制代码。
- `play/js/pyrandom.js` 按公开的 MT19937 算法（Matsumoto & Nishimura, 1998）和 CPython `random` 模块的行为
  重新实现，用于在浏览器里复现与 Python 相同的随机序列；没有复制 CPython 源代码。
- 算法思想参考（只用了思想，未复制代码）：
  - V. Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 518, 529–533 (2015) — DQN、经验回放、目标网络。
  - H. van Hasselt, A. Guez, D. Silver, "Deep Reinforcement Learning with Double Q-learning", AAAI 2016 — Double DQN。
  - K. Simonyan, A. Vedaldi, A. Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", 2013 — 梯度显著性图（“注意力红雾”）。

## 名称与商标

- “贪吃蛇 / Snake” 是通用的游戏类型名。本项目不使用任何商业游戏的名称、角色、标志或外观。
- 学校名称只以文字形式出现在署名中；没有使用学校校标（logo）图片。
