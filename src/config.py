# -*- coding: utf-8 -*-
"""
全局配置参数
"""

CONFIG = {
    # ==================== 车辆检测 ====================
    'model_path': 'yolov8n.pt',
    'detect_conf': 0.35,
    'detect_iou': 0.45,
    'detect_half': False,           # CPU用False，GPU用True
    'detect_imgsz': 640,
    'detect_classes': [2, 3, 5, 7],    # car, motorcycle, bus, truck
    'detect_interval': 2,              # 每N帧做一次YOLO
    'max_vehicles': 30,

    # ==================== 测速（逐帧像素位移法） ====================
    'speed_track_len': 20,             # 每个ID保留最近N帧轨迹
    'speed_smooth_window': 5,          # 用最后N帧位移算速度
    'speed_static_frames': 5,          # 连续N帧位移<阈值 → 静止
    'speed_static_px': 1.0,            # 静止判定像素阈值
    'speed_ema_alpha': 0.3,            # EMA平滑系数（0.3当前+0.7历史）
    'speed_clamp_max': 150.0,          # 速度上限 km/h
    'speed_limit': 80,                 # 限速 km/h
    # 透视校正：画面底部（近处）和顶部（远处）的每米像素数
    'ppm_bottom': 18.0,
    'ppm_top': 4.0,

    # ==================== 车距检测 ====================
    'camera_height': 1.5,
    'focal_length': 1000,
    'safe_dist_base': 20,
    'safe_dist_per_kmh': 0.3,
    'dist_danger_m': 2.5,
    'dist_smooth_alpha': 0.3,

    # ==================== 车道检测 ====================
    'roi_top_ratio': 0.30,

    # ==================== 黑烟检测 ====================
    'smoke_confidence': 0.7,
    'smoke_debounce': 3,
    'smoke_alert_grade': 2,

    # ==================== 视频输出 ====================
    'output_codec': 'mp4v',
    'max_upload_mb': 200,
}
