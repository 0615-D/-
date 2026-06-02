# -*- coding: utf-8 -*-
"""
模块: 黑烟排放检测器
功能: 基于林格曼黑度算法检测车辆尾气黑烟
原理: 在车辆底部提取尾气 ROI 区域，进行灰度化和阈值分割，
      计算暗色像素占比，匹配林格曼黑度等级 (0~5 级)

林格曼黑度等级:
  0 级: 全白 (无烟)
  1 级: 浅灰 (20% 黑)
  2 级: 中灰 (40% 黑)   ← 触发警报阈值
  3 级: 深灰 (60% 黑)
  4 级: 浓黑 (80% 黑)
  5 级: 全黑 (100% 黑)
"""

import cv2
import numpy as np
from .config import CONFIG


class SmokeDetector:
    """
    林格曼黑度黑烟检测器

    检测流程:
    1. 以车辆检测框底部为中心，向下扩展区域作为尾气 ROI
    2. 对 ROI 区域灰度化、高斯模糊去噪
    3. 阈值分割提取暗色像素 (黑烟)
    4. 计算暗色像素占比，匹配林格曼黑度等级
    5. 等级 >= 2 时判定为黑烟排放
    """

    # 不同灵敏度对应的参数
    SENSITIVITY = {
        'low':    {'dark_thresh': 40,  'min_ratio': 0.35, 'grade_scale': [0, 0.20, 0.40, 0.60, 0.80, 1.0]},
        'medium': {'dark_thresh': 50,  'min_ratio': 0.25, 'grade_scale': [0, 0.15, 0.30, 0.50, 0.70, 1.0]},
        'high':   {'dark_thresh': 60,  'min_ratio': 0.18, 'grade_scale': [0, 0.10, 0.20, 0.35, 0.55, 1.0]},
    }

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # 加载灵敏度参数
        sensitivity = CONFIG['smoke_sensitivity']
        params = self.SENSITIVITY.get(sensitivity, self.SENSITIVITY['medium'])
        self.dark_thresh = params['dark_thresh']
        self.min_ratio = params['min_ratio']
        self.grade_scale = params['grade_scale']

        # 每辆车的黑烟历史 (用于时间平滑，过滤短暂干扰)
        self.history = {}  # {vid: [grade, grade, ...]}
        self.history_len = 5

    def detect(self, vehicle):
        """
        检测单辆车的尾气黑烟

        参数:
            vehicle: 车辆信息 dict (需包含 'id', 'bbox', 'center')

        返回:
            {
                'grade': int,       # 林格曼黑度等级 (0~5)
                'ratio': float,     # 暗色像素占比 (0~1)
                'has_smoke': bool,  # 是否触发黑烟警报
                'roi': tuple        # 尾气 ROI 区域 (x1, y1, x2, y2) 或 None
            }
        """
        vid = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        bw = x2 - x1
        bh = y2 - y1

        # ===== Step 1: 提取尾气 ROI =====
        # 以车辆底部为中心，向下扩展
        expand_h = int(bh * CONFIG['smoke_roi_expand'])
        roi_w = int(bw * CONFIG['smoke_roi_width_ratio'])

        # ROI 中心: 车辆底部中心
        roi_cx = (x1 + x2) // 2
        roi_cy = y2  # 车辆框底部

        # 计算 ROI 坐标
        rx1 = max(0, roi_cx - roi_w // 2)
        rx2 = min(self.frame_w, roi_cx + roi_w // 2)
        ry1 = max(0, roi_cy)
        ry2 = min(self.frame_h, roi_cy + expand_h)

        # ROI 太小则跳过
        if ry2 - ry1 < 10 or rx2 - rx1 < 10:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': None}

        roi = (rx1, ry1, rx2, ry2)

        # ===== Step 2: 图像处理 =====
        # 注意: 这里只接收 bbox 坐标，实际图像需要在调用方裁剪后传入
        # 为了模块化，返回 ROI 坐标，由调用方完成裁剪和分析
        return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

    def analyze_roi(self, roi_image, vid):
        """
        分析尾气 ROI 图像，计算黑烟等级

        参数:
            roi_image: 裁剪后的尾气区域图像 (BGR)
            vid: 车辆 ID (用于历史平滑)

        返回:
            (grade, ratio, has_smoke)
        """
        if roi_image is None or roi_image.size == 0:
            return 0, 0.0, False

        # 灰度化
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊去噪
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 计算暗色像素比例
        # 暗色阈值: 低于此灰度值的像素认为是黑烟
        dark_pixels = np.sum(gray < self.dark_thresh)
        total_pixels = gray.size
        ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

        # 匹配林格曼黑度等级
        grade = 0
        for g in range(5, -1, -1):
            if ratio >= self.grade_scale[g]:
                grade = g
                break

        # 时间平滑: 使用历史记录过滤短暂干扰
        if vid not in self.history:
            self.history[vid] = []

        self.history[vid].append(grade)
        if len(self.history[vid]) > self.history_len:
            self.history[vid].pop(0)

        # 使用历史中位数作为最终等级 (过滤突变)
        sorted_history = sorted(self.history[vid])
        smoothed_grade = sorted_history[len(sorted_history) // 2]

        # 判断是否触发黑烟警报
        has_smoke = smoothed_grade >= CONFIG['smoke_alert_grade']

        return smoothed_grade, ratio, has_smoke

    def cleanup(self, active_ids):
        """清理不再活跃的车辆历史记录"""
        stale = [vid for vid in self.history if vid not in active_ids]
        for vid in stale:
            del self.history[vid]
