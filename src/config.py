# -*- coding: utf-8 -*-
"""
全局配置参数
所有可调参数集中在此文件，修改后重启程序即可生效
"""

CONFIG = {
    # ==================== 车辆检测 ====================
    'model_path': 'yolov8n.pt',       # YOLO 模型路径
    'detect_conf': 0.5,                # 检测置信度阈值 (0~1)
    'detect_classes': [2, 3, 5, 7],    # COCO 类别: car, motorcycle, bus, truck
    'detect_interval': 2,              # 每 N 帧做一次 YOLO 检测 (性能优化)
    'max_vehicles': 30,                # 单帧最大检测车辆数

    # ==================== 测速 (双参考线法) ====================
    'speed_line1_y_ratio': 0.35,       # 第一条参考线 Y 坐标占帧高的比例
    'speed_line2_y_ratio': 0.65,       # 第二条参考线 Y 坐标占帧高的比例
    'line_real_dist': 20.0,            # 两条参考线之间的实际距离 (米)
    'speed_limit': 80,                 # 限速值 (km/h)
    'speed_smooth_alpha': 0.7,         # 速度平滑系数 (越大越平滑, 0~1)
    'speed_cooldown': 8,               # 测速冷却帧数 (防止重复测量)
    'speed_min': 5,                    # 最低有效速度 (km/h, 过滤噪声)
    'speed_max': 200,                  # 最高有效速度 (km/h)

    # ==================== 车距检测 ====================
    'camera_height': 1.5,              # 摄像头离地高度 (米)
    'camera_pitch_deg': 10,            # 摄像头向下倾斜角度 (度)
    'focal_length': 1000,              # 等效焦距 (像素)
    'safe_dist_base': 20,              # 基础安全车距 (米)
    'safe_dist_per_kmh': 0.3,          # 每 km/h 车速增加的安全距离

    # ==================== 车道检测 ====================
    'roi_top_ratio': 0.45,             # 车道检测 ROI 顶部 Y 比例
    'lane_merge_angle': 15,            # 线段合并角度容差 (度)
    'lane_merge_dist': 50,             # 线段合并距离容差 (像素)

    # ==================== 热力图 ====================
    'heatmap_alpha': 0.3,              # 热力图叠加透明度 (0~1)
    'heatmap_update_interval': 5,      # 热力图每 N 帧更新一次
    'heatmap_blur_kernel': 51,         # 高斯模糊核大小 (奇数)
    'heatmap_scale': 0.25,             # 热力图计算分辨率缩放比

    # ==================== 黑烟检测 (林格曼黑度) ====================
    'smoke_sensitivity': 'medium',     # 灵敏度: low / medium / high
    'smoke_roi_expand': 0.5,           # 尾气 ROI 向下扩展比例 (相对车高)
    'smoke_roi_width_ratio': 0.6,      # 尾气 ROI 宽度比例 (相对车宽)
    'smoke_alert_grade': 2,            # 林格曼黑度 >= 此值触发警报

    # ==================== 视频输出 ====================
    'output_codec': 'mp4v',            # 输出视频编码
    'max_upload_mb': 200,              # 最大上传文件大小 (MB)
}
