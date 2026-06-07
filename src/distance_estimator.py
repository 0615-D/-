# -*- coding: utf-8 -*-
"""
模块: 车距估算器
功能: 单目几何测距，分车型标定，EMA平滑
"""

import numpy as np
from .config import CONFIG

VEHICLE_HEIGHTS = {
    'car': 1.5, 'motorcycle': 1.2, 'bus': 3.2, 'truck': 3.0, 'vehicle': 1.5,
}


class DistanceEstimator:

    def __init__(self, frame_h):
        self.frame_h = frame_h
        self.focal = CONFIG['focal_length']
        self.camera_h = CONFIG['camera_height']
        self.alpha = CONFIG['dist_smooth_alpha']
        self._smooth = {}

    def estimate(self, box_h, box_y, class_name='vehicle'):
        if box_h < 10 or box_y < 1:
            return 999.0
        real_h = VEHICLE_HEIGHTS.get(class_name, 1.5)
        d = self.focal * real_h / box_h
        return max(1.0, min(d, 100.0))

    def estimate_smooth(self, vid, raw):
        prev = self._smooth.get(vid, raw)
        smoothed = self.alpha * raw + (1 - self.alpha) * prev
        self._smooth[vid] = smoothed
        return smoothed

    def cleanup(self, active_ids):
        stale = [v for v in self._smooth if v not in active_ids]
        for v in stale:
            del self._smooth[v]

    def find_leading(self, vehicles, distances, lane_map):
        leading = {}
        for v in vehicles:
            vid = v['id']
            v_lane = lane_map.get(vid, -1)
            if v_lane < 0:
                leading[vid] = None
                continue
            best, best_gap = None, float('inf')
            for u in vehicles:
                if u['id'] == vid:
                    continue
                if lane_map.get(u['id'], -1) != v_lane:
                    continue
                if u['center'][1] >= v['center'][1]:
                    continue
                gap = distances.get(vid, 999) - distances.get(u['id'], 999)
                if 0 < gap < best_gap:
                    best_gap = gap
                    best = u['id']
            leading[vid] = best
        return leading
