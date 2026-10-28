/* All display names, presets and colors in one place (no third-party trademarks; open-source projects
 * are named by their repository, as their licenses require). The `brains` list is also read by
 * brains.py (id + file) and locked by tests/test_play_parity.py. */
(function (g) {
  "use strict";
  g.APP_CONFIG = {
    title: "贪吃蛇 · 看懂 AI 的大脑",
    shortTitle: "贪吃蛇 AI 大脑",
    mobileTitle: "贪吃蛇 · AI 大脑",   // 手机竖屏头部只有一行，用短标题
    storagePrefix: "snake-dqn-brain.",

    // env action index -> name (0=上 1=下 2=左 3=右, same as snake_env.ACTION_NAMES)
    actions: [
      { name: "上", arrow: "↑" },
      { name: "下", arrow: "↓" },
      { name: "左", arrow: "←" },
      { name: "右", arrow: "→" }
    ],

    // The brains you can switch between. `file` is loaded by index.html; the network spec, observation
    // format, action order and training-time action mask live inside the model file (tools/export_sb3.py).
    //   policy  "greedy" = always the highest score;  "sample" = draw by probability (the MLP loops forever when greedy)
    //   guard   true = add the "no suicide" mask on top of the model's own mask (shown as a switch); false = none
    brains: [
      { id: "ppo_linyi", file: "model/ppo_cnn_linyi_fp16.js", key: "1",
        name: "PPO 卷积网 A", title: "PPO 卷积网 A（12×12，掩码训练）",
        source: "linyiLYi/snake-ai", license: "Apache-2.0", url: "https://github.com/linyiLYi/snake-ai",
        algo: "MaskablePPO + CNN（Nature DQN 结构）", trained: "作者训练 1 亿步，带“不撞死”动作掩码",
        policy: "greedy", guard: false,
        blurb: "看到的是一张 84×84 的彩图（每格 7×7 像素）：身体灰色从头到尾变暗，蛇头绿、蛇尾红、食物蓝。训练时就带着“不能撞死”的动作掩码，所以我们也照样给它。" },
      { id: "ppo_ray", file: "model/ppo_cnn_ray_fp16.js", key: "2",
        name: "PPO 卷积网 B", title: "PPO 卷积网 B（12×12，无掩码训练）",
        source: "Part-time-Ray/AI-Snake-Game", license: "MIT", url: "https://github.com/Part-time-Ray/AI-Snake-Game",
        algo: "PPO + CNN（Nature DQN 结构）", trained: "作者训练 1 亿步，无动作掩码",
        policy: "greedy", guard: true,
        blurb: "看到的是一张 60×60 的彩图（每格 5×5 像素，而且是转置的：图的行 = 棋盘的列）。训练时没有掩码，偶尔会很自信地撞死；勾上「护栏」就只在不撞死的方向里选。" },
      { id: "ppo_mlp_linyi", file: "model/ppo_mlp_linyi_fp16.js", key: "3",
        name: "PPO 全连接网", title: "PPO 全连接网（12×12，没有卷积）",
        source: "linyiLYi/snake-ai", license: "Apache-2.0", url: "https://github.com/linyiLYi/snake-ai",
        algo: "MaskablePPO + 两层 64 的全连接网", trained: "作者训练 1 亿步，带动作掩码",
        policy: "sample", guard: false,
        blurb: "只看一张 12×12 的数表（身体 0.8→0.2，蛇头 1，食物 −1），没有卷积层。总选最大概率时它会原地绕圈饿死，所以默认按概率抽样走。" },
      { id: "dqn_own", file: "model/snake_cnn_fp16.js", key: "4", global: "SNAKE_MODEL",
        name: "Double DQN（本仓库）", title: "Double DQN（10×10，本仓库 15 分钟训练）",
        source: "本仓库 train.py", license: "MIT", url: "",
        algo: "Double DQN + 三层 3×3 卷积", trained: "纯 CPU 训练约 15 分钟（步数未记录）",
        policy: "greedy", guard: false,
        blurb: "看到的是 5 张 10×10 的图（蛇头、身体、食物、方向上下、方向左右）。训练时间很短，常常 20 来步就撞死——留着当对照：同样的大脑面板，看它和强大脑有什么不同。" }
    ],
    brainDefault: 0,

    // input planes shown on the brain panel, per observation kind (brains.py OBS); order = channel order
    inputs: {
      dqn5: [
        { name: "蛇头", desc: "蛇头所在的格子 = 1，其余 = 0" },
        { name: "身体", desc: "身体：紧挨蛇头最亮（接近 1），往尾巴逐渐变暗（最低 0.15）" },
        { name: "食物", desc: "食物所在的格子 = 1，其余 = 0" },
        { name: "方向·上下", desc: "整张图填同一个数：往上走 = −1，往下走 = +1，左右走 = 0", signed: true },
        { name: "方向·左右", desc: "整张图填同一个数：往左走 = −1，往右走 = +1，上下走 = 0", signed: true }
      ],
      linyi_cnn: [
        { name: "红通道", desc: "身体灰色（头 200 → 尾 50，除以 255），蛇尾 255，蛇头 0，食物 0" },
        { name: "绿通道", desc: "身体灰色，蛇头 255，蛇尾 0，食物 0" },
        { name: "蓝通道", desc: "身体灰色，食物 255，蛇头 0，蛇尾 0" }
      ],
      ray_cnn: [
        { name: "红通道", desc: "身体灰色（头 200 → 尾 50，除以 255），蛇头 0，食物 0。注意整张图是转置的：图的行 = 棋盘的列" },
        { name: "绿通道", desc: "身体灰色，蛇头 255，食物 0" },
        { name: "蓝通道", desc: "身体灰色，食物 255，蛇头 0" }
      ],
      linyi_mlp: [
        { name: "数表", desc: "12×12 的一张表：身体 0.8（头）→ 0.2（尾），蛇头 1，食物 −1，其余 0", signed: true }
      ]
    },

    // 观看 speed presets (ms per step). The slider keeps fine control.
    watchSpeeds: [
      { name: "慢放讲解", ms: 700, tip: "每步 0.7 秒：适合边讲边看右边的大脑怎么变" },
      { name: "正常", ms: 160, tip: "每步 0.16 秒：看蛇的整体策略" },
      { name: "快", ms: 60, tip: "每步 0.06 秒：快速看完一局" },
      { name: "极速", ms: 5, tip: "电脑能算多快就走多快：用来统计平均分" }
    ],
    watchDefault: 1,

    watchSeed: 999,     // same seed as the evaluation scripts, so the first games match results/

    topK: { conv: 6, fc: 24, lines: 40 },

    colors: {
      bg: "#0f141b", panel: "#161d27", grid: "rgba(255,255,255,.05)",
      head: "#52ffa8", food: "#ff5252",
      pos: "#ffb74d",   // contribution that pushes the chosen score up
      neg: "#4fc3f7",   // contribution that pushes it down
      chosen: "#2dbd6e", guard: "#ff8a80"
    }
  };
})(typeof window !== "undefined" ? window : globalThis);
