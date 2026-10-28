/* All display names, presets and colors in one place (no third-party names or trademarks).
 * inputGroups must match snake_env.py's state channels one-to-one — locked by
 * tests/test_play_parity.py::test_input_groups_match_env_channels. */
(function (g) {
  "use strict";
  g.APP_CONFIG = {
    title: "贪吃蛇 · 看懂 AI 的大脑",
    shortTitle: "贪吃蛇 DQN 大脑",
    mobileTitle: "贪吃蛇 · AI 大脑",   // 手机竖屏头部只有一行，用短标题
    storagePrefix: "snake-dqn-brain.",

    // action index -> name (0=上 1=下 2=左 3=右, same as snake_env.ACTION_NAMES)
    actions: [
      { name: "上", arrow: "↑" },
      { name: "下", arrow: "↓" },
      { name: "左", arrow: "←" },
      { name: "右", arrow: "→" }
    ],

    // network input channels (5 x N x N), in the exact order of SnakeEnv.state()
    inputGroups: [
      { ch: 0, key: "head", name: "蛇头", desc: "蛇头所在的格子 = 1，其余 = 0" },
      { ch: 1, key: "body", name: "身体", desc: "身体：紧挨蛇头最亮（接近 1），往尾巴逐渐变暗（最低 0.15）" },
      { ch: 2, key: "food", name: "食物", desc: "食物所在的格子 = 1，其余 = 0" },
      { ch: 3, key: "dir_dr", name: "方向·上下", desc: "整张图填同一个数：往上走 = −1，往下走 = +1，左右走 = 0" },
      { ch: 4, key: "dir_dc", name: "方向·左右", desc: "整张图填同一个数：往左走 = −1，往右走 = +1，上下走 = 0" }
    ],

    // 观战 speed presets (ms per step). The slider keeps fine control.
    watchSpeeds: [
      { name: "慢放讲解", ms: 700, tip: "每步 0.7 秒：适合边讲边看右边的大脑怎么变" },
      { name: "正常", ms: 160, tip: "每步 0.16 秒：看蛇的整体策略" },
      { name: "快", ms: 60, tip: "每步 0.06 秒：快速看完一局" },
      { name: "极速", ms: 5, tip: "电脑能算多快就走多快：用来统计平均分" }
    ],
    watchDefault: 1,

    // 我来玩 difficulty presets (ms per step; 0 = turn-based)
    playLevels: [
      { name: "回合制", ms: 0, tip: "你按一次方向键，蛇才走一步。不比反应，只比判断——和 DQN 最公平的比法" },
      { name: "慢", ms: 450, tip: "每 0.45 秒自动前进一格，不按键就直走" },
      { name: "正常", ms: 260, tip: "每 0.26 秒前进一格，接近常见的贪吃蛇速度" },
      { name: "快", ms: 150, tip: "每 0.15 秒前进一格：考验反应" }
    ],
    playDefault: 0,

    watchSeed: 999,     // same seed as train.py's evaluation, so the first games match results/result.json
    playSeed0: 1000,    // game k of 我来玩 uses seed 1000 + k for both you and the DQN

    topK: { conv: 6, fc: 24, lines: 40 },

    colors: {
      bg: "#0f141b", panel: "#161d27", grid: "rgba(255,255,255,.05)",
      head: "#52ffa8", food: "#ff5252",
      pos: "#ffb74d",   // contribution that pushes the chosen Q up
      neg: "#4fc3f7",   // contribution that pushes it down
      chosen: "#2dbd6e", you: "#ffd54f", dqn: "#4fc3f7"
    }
  };
})(typeof window !== "undefined" ? window : globalThis);
