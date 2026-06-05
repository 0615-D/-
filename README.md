# OpenCV 智能视觉系统

基于计算机视觉技术实现的智能视觉系统，集成车道线检测与手势识别功能。

## 可视化Web界面

基于 Gradio 构建的交互式可视化平台：

### 🚗 车道线检测
| 功能 | 说明 |
|------|------|
| 🖼️ 图片检测 | 上传图片实时检测车道线，显示曲率和偏移量 |
| 🔬 Pipeline可视化 | 8阶段处理过程逐步展示 |
| 📊 阈值方法对比 | HLS/LAB/HSV/Sobel等6种方法效果对比 |
| 🔄 透视变换 | 源点标记 + 鸟瞰图可视化 |
| 🎬 视频检测 | 上传视频逐帧处理，输出带车道线叠加的视频 |

### 🤚 手势识别
| 功能 | 说明 |
|------|------|
| 🖼️ 图片识别 | 手部关键点检测、手势分类、手指计数 |
| 🎬 视频识别 | 逐帧手势识别，统计手势分布 |
| 🎨 虚拟画板 | 食指绘画、双指选择模式 |
| 📹 实时识别 | 摄像头实时手势识别 |
| 🎨 实时画板 | 摄像头实时虚拟绘画 |

启动Web界面：
```bash
python app.py
# 浏览器访问 http://localhost:7862
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 前端界面 | Gradio Web框架 |
| 图像处理 | OpenCV (色彩空间转换、边缘检测、形态学操作) |
| 手势识别 | MediaPipe Hands (21关键点检测) |
| 数值计算 | NumPy (多项式拟合、矩阵运算) |
| 数据可视化 | Matplotlib |
| 视频转码 | FFmpeg (H.264编码) |

## 环境要求

```
Python >= 3.8
opencv-python >= 4.5.0
numpy >= 1.20.0
matplotlib >= 3.3.0
gradio >= 3.0.0
imageio-ffmpeg >= 0.4.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
shijian/
├── app.py                      # Web可视化界面
├── run.py                      # 命令行主程序
├── requirements.txt            # 依赖清单
├── README.md                   # 项目文档
├── test_images/                # 测试图片
│   ├── solidWhiteCurve.jpg
│   ├── solidWhiteRight.jpg
│   ├── solidYellowCurve.jpg
│   ├── solidYellowCurve2.jpg
│   ├── solidYellowLeft.jpg
│   └── whiteCarLaneSwitch.jpg
├── test_videos/                # 测试视频
│   ├── solidWhiteRight.mp4
│   └── challenge.mp4
├── src/
│   ├── basic_lane.py           # 基础车道线检测
│   ├── advanced_lane.py        # 高级7阶段管线
│   ├── video_processor.py      # 视频处理模块
│   ├── visualization.py        # 可视化模块
│   └── gesture_recognition.py  # 手势识别模块
└── output/                     # 输出结果
```

## 算法Pipeline

| 阶段 | 功能 | 核心技术 |
|------|------|----------|
| Stage 1 | 相机标定与去畸变 | 棋盘格角点检测 + 畸变系数计算 |
| Stage 2 | 透视变换(鸟瞰图) | 源点/目标点映射矩阵计算 |
| Stage 3 | 多阈值融合 | HLS/LAB/HSV色彩空间分离 + Sobel梯度算子 |
| Stage 4 | 直方图峰值检测 | 像素列统计 + 极值定位 |
| Stage 5 | 滑动窗口多项式拟合 | 二阶多项式最小二乘拟合 |
| Stage 6 | 曲率半径 & 偏移量 | 曲率公式计算 + 车道中心偏移 |
| Stage 7 | 可视化叠加 | 逆透视变换 + 图像加权融合 |

## 多阈值融合方法

| 方法 | 色彩空间 | 用途 |
|------|----------|------|
| Sobel X/Y 梯度 | 灰度 | 检测垂直/水平边缘 |
| Sobel 幅值 | 灰度 | 梯度强度筛选 |
| Sobel 方向 | 灰度 | 边缘角度过滤 |
| HLS S 通道 | HLS | 抗阴影，检测黄色/白色线 |
| LAB B 通道 | LAB | 黄色线专用检测 |
| HSV 白色掩码 | HSV | 白色线专用检测 |

## 手势识别功能

| 功能 | 技术 | 说明 |
|------|------|------|
| 手部关键点检测 | MediaPipe Hands | 21个关键点实时追踪 |
| 手势识别 | 距离计算 + 手指状态分析 | 石头/剪刀/布、数字1-5、OK等 |
| 手指计数 | 关键点坐标比较 | 统计竖起的手指数量 |
| 虚拟画板 | 手势模式切换 | 食指绘画、双指选择 |
| 实时摄像头 | Gradio Webcam + Streaming | 实时手势识别和绘画 |

### 支持的手势

| 手势 | 描述 | 手指状态 |
|------|------|----------|
| ✊ 石头 (Rock) | 拳头 | 全部弯曲 |
| ✌️ 剪刀 (Scissors) | 二 | 食指+中指竖起 |
| 🖐️ 布 (Paper) | 张开手掌 | 全部竖起 |
| ☝️ 数字 1-5 | 单手数字 | 对应手指竖起 |
| 👍 竖大拇指 | 拇指竖起 | 只有拇指 |
| 👌 OK手势 | 拇指食指圆圈 | 特殊距离检测 |

## 使用方法

### Web界面 (推荐)
```bash
python app.py
```

### 命令行
```bash
# 运行完整演示
python run.py

# 只运行特定任务
python run.py --task 1          # 视频帧提取
python run.py --task 2          # ROI提取
python run.py --task 3          # 直线检测
python run.py --task 4          # 弯道检测
python run.py --task advanced   # 高级管线(图片)
python run.py --task video_basic # 基础方法处理视频
python run.py --task video_adv   # 高级方法处理视频
python run.py --task visualize   # 生成各阶段可视化图
```

## 可视化输出

运行 `python run.py --task visualize` 生成以下图片：

- **pipeline_stages.jpg** - 8阶段处理流程总览 (2x4网格)
- **perspective_transform.jpg** - 透视变换前后对比
- **threshold_comparison.jpg** - 多种阈值方法效果对比 (2x4网格)
- **sliding_window.jpg** - 滑动窗口搜索过程可视化
