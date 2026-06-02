# -*- coding: utf-8 -*-
"""
模块: 车距估算器
功能: 基于单目相机测距原理，估算车辆到摄像头的距离以及同车道前后车距离
原理: 利用针孔相机模型，通过边界框高度与距离的反比关系估算距离
"""

import numpy as np
from .config import CONFIG


class DistanceEstimator:
    """
    单目测距器

    使用经验公式: 距离 = K / 框高
    其中 K 为标定常数，通过已知距离和对应框高确定
    """

    def __init__(self, frame_h):
        self.frame_h = frame_h
        # 标定常数: 在参考距离处，框高 * 距离 = 常数
        # 假设在 10m 处，一辆普通轿车的框高约为 120 像素
        self.K = CONFIG['focal_length'] * 1.5  # 经验标定值

    def estimate(self, box_h, box_y):
        """
        根据边界框高度和位置估算距离

        参数:
            box_h: 边界框高度 (像素)
            box_y: 边界框底部 Y 坐标 (像素)

        返回:
            估算距离 (米)
        """
        if box_h < 10:
            return 999.0  # 框太小，认为很远

        # 基础距离 = K / 框高
        base_dist = self.K / box_h

        # 透视修正: 框在画面底部 (近处) 时修正小，顶部 (远处) 时修正大
        # 利用 Y 坐标位置做线性修正
        y_ratio = box_y / self.frame_h
        perspective_factor = 0.7 + 0.6 * (1 - y_ratio)  # 底部=0.7, 顶部=1.3

        dist = base_dist * perspective_factor
        return max(1.0, min(dist, 100.0))  # 限制在 1~100m 范围

    def compute_safe_distance(self, speed_kmh):
        """
        根据当前车速计算安全车距

        公式: 安全距离 = 基础距离 + 每km/h增量 × 速度
        例如: 60km/h 时安全距离 = 20 + 0.3*60 = 38m

        参数:
            speed_kmh: 当前车速 (km/h)

        返回:
            安全距离 (米)
        """
        return CONFIG['safe_dist_base'] + CONFIG['safe_dist_per_kmh'] * speed_kmh

    def find_leading_vehicle(self, vehicles, distances, lane_map):
        """
        对于每个车辆，找到同车道内前方最近的车辆

        参数:
            vehicles: 车辆列表
            distances: 距离字典 {vid: dist_m}
            lane_map: 车道映射 {vid: lane_id}

        返回:
            前车关系字典 {vid: leading_vid 或 None}
        """
        leading = {}
        for v in vehicles:
            vid = v['id']
            v_lane = lane_map.get(vid, -1)
            v_cy = v['center'][1]

            # 在同车道车辆中，找 Y 坐标更小 (更远) 且最近的
            best = None
            best_gap = float('inf')
            for u in vehicles:
                uid = u['id']
                if uid == vid:
                    continue
                if lane_map.get(uid, -1) != v_lane:
                    continue
                # 前车应该在画面上方 (Y 更小 = 距离更远)
                if u['center'][1] < v_cy:
                    gap = distances.get(vid, 999) - distances.get(uid, 999)
                    if 0 < gap < best_gap:
                        best_gap = gap
                        best = uid

            leading[vid] = best

        return leading
