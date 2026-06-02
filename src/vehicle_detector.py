# -*- coding: utf-8 -*-
"""
模块: 车辆检测器
功能: 使用 YOLOv8 + ByteTrack 进行车辆检测与跟踪
输入: 视频帧 (BGR numpy array)
输出: 车辆列表 [{id, class_name, bbox, center, conf}, ...]
"""

from ultralytics import YOLO
from .config import CONFIG


class VehicleDetector:
    """YOLOv8 车辆检测器，集成 ByteTrack 跟踪器"""

    # COCO 数据集中车辆类别 ID 与名称映射
    CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def __init__(self):
        # 加载 YOLO 模型 (首次运行会自动下载)
        self.model = YOLO(CONFIG['model_path'])
        self.frame_count = 0       # 当前帧计数
        self.last_vehicles = []    # 上一次检测结果 (用于间隔帧复用)

    def detect(self, frame):
        """
        检测视频帧中的车辆

        参数:
            frame: BGR 格式的视频帧 (numpy array)

        返回:
            车辆列表, 每个元素为 dict:
            {
                'id': int,          # ByteTrack 分配的跟踪 ID
                'class_name': str,  # 车辆类型名称
                'bbox': (x1,y1,x2,y2),  # 边界框坐标
                'center': (cx, cy),      # 中心点坐标
                'conf': float       # 检测置信度
            }
        """
        self.frame_count += 1

        # 按间隔做检测，中间帧复用上次结果
        if self.frame_count % CONFIG['detect_interval'] != 1:
            return self.last_vehicles

        # YOLOv8 推理 + ByteTrack 跟踪
        # persist=True 保持跨帧跟踪状态
        results = self.model.track(
            frame,
            conf=CONFIG['detect_conf'],
            classes=CONFIG['detect_classes'],
            tracker='bytetrack.yaml',
            persist=True,
            verbose=False
        )

        vehicles = []
        result = results[0]

        # 无检测结果或无跟踪 ID
        if result.boxes is None or result.boxes.id is None:
            self.last_vehicles = vehicles
            return vehicles

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            track_id = int(boxes.id[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # 过滤过小的检测框
            bw, bh = x2 - x1, y2 - y1
            if bw < 20 or bh < 20:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            vehicles.append({
                'id': track_id,
                'class_name': self.CLASS_NAMES.get(cls_id, 'vehicle'),
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'conf': conf,
            })

        # 限制最大车辆数，按置信度排序
        vehicles.sort(key=lambda v: v['conf'], reverse=True)
        vehicles = vehicles[:CONFIG['max_vehicles']]

        self.last_vehicles = vehicles
        return vehicles
