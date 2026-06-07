# -*- coding: utf-8 -*-
"""
模块: 车辆检测器
功能: YOLOv8s + ByteTrack 稳定检测与跟踪
"""

import os
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)
from ultralytics import YOLO
from .config import CONFIG

_TRACKER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bytetrack_custom.yaml')


class VehicleDetector:

    CLASS_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def __init__(self):
        self.model = YOLO(CONFIG['model_path'])
        self.last_vehicles = []

    def detect(self, frame):
        results = self.model.track(
            frame,
            imgsz=CONFIG['detect_imgsz'],
            conf=CONFIG['detect_conf'],
            iou=CONFIG['detect_iou'],
            half=CONFIG['detect_half'],
            classes=CONFIG['detect_classes'],
            tracker=_TRACKER_PATH,
            persist=True,
            verbose=False,
        )

        vehicles = []
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            self.last_vehicles = vehicles
            return vehicles

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            track_id = int(boxes.id[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            bw, bh = x2 - x1, y2 - y1
            if bw < 20 or bh < 20:
                continue

            vehicles.append({
                'id': track_id,
                'class_name': self.CLASS_NAMES.get(cls_id, 'vehicle'),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'conf': conf,
            })

        vehicles.sort(key=lambda v: v['conf'], reverse=True)
        vehicles = vehicles[:CONFIG['max_vehicles']]
        self.last_vehicles = vehicles
        return vehicles
