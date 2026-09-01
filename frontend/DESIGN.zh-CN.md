# BookTalk 设计说明 v3 —— "Studio Light"

[English](DESIGN.md) · **中文**

第三个方向，依据客户对 v2 的反馈：**任何地方都不用深色表面** —— 深色书架侧栏取消。
整个应用改为跟随客户给的仪表盘参考图：冷调浅色画布、大圆角加极柔阴影的悬浮白卡、
一个用得明确的蓝色主题色、柔和的粉彩渐变色块，以及**毛玻璃覆盖表面**。
字体不变（Literata / Instrument Sans / Spline Sans Mono）。

关于 skill 之间的冲突：客户明确要求毛玻璃和更强的主题色。按 impeccable 自己的
"以需求说明为准"规则，这一条覆盖 minimalist-ui 对毛玻璃的禁用 —— 但毛玻璃只用在
真正覆盖内容之上的表面（顶栏、侧栏、输入区、toast、移动端弹层），绝不作为装饰
出现在行内卡片上，而 craft-floor 的对比度下限在每一个毛玻璃表面上都不可让步。

## Token（替换 v2 的书架/书脊 token；其余沿用）

```css
--canvas: #E9EBF2;                 /* 页面底色，比 v2 更偏冷 */
--canvas-glow-a: radial-gradient(...#FFD9C7 at ~8% opacity);  /* <body> 上两层静态的  */
--canvas-glow-b: radial-gradient(...#BFD8FF at ~10% opacity); /* 环境色晕，毛玻璃表面 */
                                   /* 才能明显地虚化出东西                            */
--card: #FFFFFF;
--ink: #171A20;  --ink-soft: #5F6672;  --line: #E2E5EB;  --line-strong: #CCD2DC;
--blue: #3D63F2;  --blue-deep: #2C4BD8;  --blue-soft: #DCE7FF;
--live: #F0604D;  --live-strong: #D93C23;  --live-ink: #B8301E;   /* 仅用于录音 */
--grad-warm: linear-gradient(135deg,#FFD9C7,#F6B9D2);
--grad-cool: linear-gradient(135deg,#BFD8FF,#A3E7DB);
--glass: rgba(255,255,255,0.72);
--glass-blur: saturate(1.4) blur(18px);
--glass-line: rgba(255,255,255,0.55);   /* 毛玻璃表面顶部的 1px 边线 */
--radius-card: 20px; --radius-inner: 12px; --radius-pill: 999px;
--shadow-card: 0 8px 28px rgba(23,26,32,0.07);
--shadow-lift: 0 14px 36px rgba(23,26,32,0.12);
--err: #B3402F; --err-bg: #FBE9E4;
```

规则：只有当毛玻璃盖在浅色画布上、alpha ≥0.7 时才允许在它上面放文字（要实测有效对比度；
把 blur 关掉时 ink 也必须保持 ≥4.5:1 —— @supports 的降级方案是纯色
`rgba(255,255,255,0.92)`）。珊瑚色只用于录音。不做渐变文字。

## 主题色，用得明确

v2 把蓝色用少了；v3 让它成为界面的主导色，和参考图完全一致：
- 主操作按钮用 --blue 填充，图标为白色（麦克风空闲态、发送按钮就绪时）。
- 选中的书卡加一圈 --blue 选中描边，再加一个蓝色的小对勾 pill（参考图里"Sep"那颗 pill
  的做法）；TOC 选中行 = 蓝色文字 + blue-soft pill。
- 入库进度条、阶段计量条、朗读状态下的均衡器/波形、链接、焦点环、"Used N sources"
  触发按钮：全部用 --blue。
- 耗时/统计 pill 保持白底加等宽的 ink 色文字，数字允许用蓝色。

## 各类表面

- **画布**：--canvas，两层粉彩环境色晕固定在所有内容之后（position: fixed,
  pointer-events: none，不做动画 —— operate.md 仍然禁止环境动效）。正是它让毛玻璃表面
  看起来是玻璃，而不是一块灰。
- **顶栏**（新增）：横跨 studio 列顶部的细长 sticky 条 —— 书名、章节/页码信息 pill、
  健康状态。`background: var(--glass); backdrop-filter: var(--glass-blur)`；底部一条
  60% 的 --line 发丝线；内容从它下面滚过。
- **左侧栏，浅色**：一块毛玻璃面板（--glass + blur，1px --glass-line 内顶边，
  --shadow-card），距画布边缘 12px 悬浮，radius-card。里面放：
  - **用书卡取代书脊**：每本书 = 一张白色内层卡片，radius-inner，左边一块小封面块
    （44×58px，圆角 8px）：由书名哈希出来的确定性粉彩渐变 —— 色系冷暖交替 —— 上面是
    书名首字母，Literata，白色或 ink 色（按对比度选）。标题（Literata，两行截断），
    元信息行用等宽（"12 ch · 226 pp"）。选中卡片：蓝色描边 + 对勾 pill。
    入库中：封面块 60% 不透明度，一条细的蓝色进度下划线，阶段文案用等宽。
    失败：err 色调的文案。
  - "Add a book" = 浅色虚线占位块，中间一个蓝色加号。
  - **TOC**：浅色行，等宽、前面补零的编号用 --ink-soft，当前章节用 blue-soft pill，
    章节梗概的展开行照常工作（改成浅色样式）。
- **studio 列**：结构不变（62rem 的列），卡片白色 radius-card。
- **输入区**：毛玻璃条（--glass + blur），sticky 在底部，距边缘 12px 悬浮，radius-card
  加 --shadow-lift —— 对话内容从它下面滚过。里面的输入框仍是纯白实心的
  （文字绝不直接落在毛玻璃上）。
- **统计块**（入库）：v2 的行为不变，渐变取自那两组色系。
- **空状态主区**：白卡、大标题，拖放区铺 12% 的 --grad-cool；再加一条细的 --grad-warm
  装饰条或一角淡色，让这块不至于完全没有颜色。
- **引用/边注、来源追溯、低置信度提示条**：结构和 token 保持不变（白色小卡片、蓝色等宽
  序号、--blue-soft 高亮、err 提示条）。
- **toast 与移动端弹层**：毛玻璃（--glass + blur），radius-card。

## 动效与状态

v2 的规则全部继续有效：120/180/240ms，只在状态变化时做动效，加载过程不编排动画，
遵守 prefers-reduced-motion，每个控件都保留 default/hover/focus/active/
disabled/loading/error 状态。卡片 hover 时用 --shadow-lift 抬起。选中描边过渡 180ms。
除此之外不再有任何动画。

## 验收标准（不变，且有约束力）

craft-floor 的 Verify 清单；用 redesign-existing-projects 对做完的成品做一次审查；
每一组上线的配色都要实测对比度，包括降级背景下的毛玻璃上文字；`npm run build` 零 TS
错误；`npx vitest run` 全绿；浏览器自带的表面（::selection、光标、滚动条、焦点环）
全部改到浅色调色板 —— 不能留下 v2 书架的深色滚动条残余。
