# 射流轴交换 · Axis Switching — 关键帧动画工作台

水流通过非圆形小口（椭圆 / 矩形 / 方形）喷出后，因表面张力振荡形成"锁链状"水柱的物理模拟器，重构为**全参数关键帧动画工作台**。

基于原版 [axis-switching-simulator](https://github.com/) 的物理模型（`jetPhysics.ts` 移植为纯 JS），叠加 Three.js 可视化与时间轴动画编排能力。

## 功能特性

- **物理模型**：超椭圆（p=6）孔口、保面积收缩、3.5 次轴交换、粘性衰减、韦伯数/雷诺数驱动
- **关键帧动画**：10 条相机轨道 + 9 条参数轨道，Catmull-Rom 平滑 / 线性 / 阶梯插值
- **时间轴编辑器**：拖拽 scrub 自动创建关键帧、菱形关键帧拖拽/编辑、播放头 scrub
- **多视角渲染**：玻璃材质喷射、热力图着色、扫描平面截面、24 线笼、带实时孔口的孔板
- **导出**：PNG 序列（ZIP）、WebM 视频（Whammy）、实时语音解说（MediaRecorder 混音）

## 使用

```bash
# 方式一：直接打开（需本地静态服务器，因使用 ES Modules）
python3 -m http.server 8080 --directory .

# 方式二：双击 index.html（部分浏览器因 CORS 限制不可用，推荐方式一）
```

浏览器访问 `http://localhost:8080`。

## 操作指南

| 操作 | 说明 |
|---|---|
| 点击时间轴 | 定位播放头 |
| 拖拽参数值 | 自动创建关键帧（scrub） |
| 拖拽菱形关键帧 | 调整时间 / 数值 |
| 双击菱形 | 编辑数值 |
| 播放 / 暂停 | 空格 或 播放按钮 |
| 撤回 / 重做 | ⌘Z / Ctrl+Z 撤回，⌘⇧Z / Ctrl+Shift+Z 或 Ctrl+Y 重做（或工具栏 ↩ 撤回 / ↪ 重做按钮）；支持全部关键帧修改操作 |
| 自由视角 → 📌 同步视角 | 自由视角摆好机位（旋转/缩放/右键平移）后点击，把当前视口机位写入播放头时间的关键帧；取消"看向视觉中心"时还会写入旋转角 |
| 导出 | 顶部工具栏 → 导出（PNG 序列 / WebM / 语音解说） |

## 技术栈

- Three.js r160（本地 vendored，无需 npm）
- 纯 vanilla JS 关键帧引擎（Catmull-Rom / 线性 / 阶梯）
- JSZip（PNG 序列导出）+ Whammy（WebM 导出）+ MediaRecorder（语音混音）

## 文件结构

```
axis-switching-workbench/
├── index.html          # 工作台 UI（时间轴 / 关键帧编辑器 / 导出面板）
├── app.js              # 核心：物理模型移植 + 场景重建 + 关键帧系统 + 导出管线
└── assets/vendor/      # 离线依赖：three.module.js / OrbitControls / JSZip / Whammy
```

## 在线演示

GitHub Pages：`https://<username>.github.io/axis-switching-workbench/`
