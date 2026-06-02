# 智能车辆检测分析系统

基于 Python + YOLOv8 的实时车辆检测分析系统，支持视频上传处理，输出带标注的可视化视频。

## 功能模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 车辆检测 | `src/vehicle_detector.py` | YOLOv8 + ByteTrack 跟踪 |
| 车速计算 | `src/speed_calculator.py` | 双参考线测速法 |
| 车距预警 | `src/distance_estimator.py` | 单目测距 + 安全车距 |
| 车道检测 | `src/lane_detector.py` | 边缘检测 + 霍夫变换 |
| 密度热力图 | `src/heatmap.py` | 累积密度 + 高斯模糊 |
| 黑烟检测 | `src/smoke_detector.py` | 林格曼黑度算法 |
| 视频处理 | `src/video_processor.py` | 核心调度模块 |
| Web 应用 | `app.py` | Flask 上传与展示 |

## 安装依赖

```bash
pip install ultralytics opencv-python numpy scipy flask
```

> ByteTrack 跟踪器还需要 `lap` 包:
> ```bash
> pip install lap
> ```

## 运行方式

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:5000`，上传视频即可开始检测。

## 参数配置

所有可调参数集中在 `src/config.py` 的 `CONFIG` 字典中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `detect_conf` | 0.5 | 检测置信度阈值 |
| `line_real_dist` | 20.0 | 测速参考线实际距离 (米) |
| `speed_limit` | 80 | 限速值 (km/h) |
| `safe_dist_base` | 20 | 基础安全车距 (米) |
| `heatmap_alpha` | 0.3 | 热力图透明度 |
| `smoke_sensitivity` | medium | 黑烟检测灵敏度 (low/medium/high) |
| `smoke_alert_grade` | 2 | 林格曼黑度警报阈值 |

修改后重启程序即可生效。

## 输出说明

- **处理后视频**: `output/` 目录下的 `.mp4` 文件
- **热力图**: `output/` 目录下的 `_heatmap.png` 文件
- 视频中包含: 检测框、车辆ID、车速、车距、热力图、黑烟标注、HUD仪表盘
