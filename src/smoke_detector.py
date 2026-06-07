# -*- coding: utf-8 -*-
"""
模块: 黑烟检测器
功能: 车尾区域黑烟检测，林格曼黑度分级，连续帧确认
核心防误报: 车身灰度对比 + HSV饱和度检测 + 路面对比 + 连续帧确认
"""

import cv2
import numpy as np
from .config import CONFIG


class SmokeDetector:

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.debounce = CONFIG['smoke_debounce']
        self.threshold = CONFIG['smoke_confidence']  # 0.7
        self._counters = {}  # {vid: consecutive_frames}

    def detect(self, frame, vehicle):
        """
        检测单辆车的尾气黑烟
        返回: {'grade': int, 'ratio': float, 'has_smoke': bool, 'roi': tuple|None}
        """
        vid = vehicle['id']
        x1, y1, x2, y2 = vehicle['bbox']
        bw, bh = x2 - x1, y2 - y1

        if bw < 20 or bh < 20:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': None}

        # ── 车尾区域：底部1/4高度，水平居中 ──
        tail_h = max(1, bh // 4)
        ry1 = max(0, y2 - tail_h)
        ry2 = min(self.frame_h, y2 + tail_h // 3)
        pad_x = max(1, int(bw * 0.10))
        rx1 = max(0, x1 + pad_x)
        rx2 = min(self.frame_w, x2 - pad_x)

        if ry2 - ry1 < 10 or rx2 - rx1 < 10:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': None}

        roi_img = frame[ry1:ry2, rx1:rx2]
        if roi_img.size == 0:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': None}

        roi = (rx1, ry1, rx2, ry2)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # ══════════════════════════════════════════
        # 深色车身过滤（核心防误报逻辑）
        # ══════════════════════════════════════════

        # ── 过滤1: 车身本体对比 ──
        # 采样车辆上半部分车身灰度，如果尾部和车身一样暗，说明是车身不是烟
        body_mean = self._sample_body_gray(frame, x1, y1, x2, y2)
        if body_mean > 0:
            # 尾部灰度 >= 车身灰度的 90% → 尾部并不比车身暗，是车身
            if mean_val >= body_mean * 0.90:
                return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

        # ── 过滤2: 深色车身（宽范围） ──
        # 灰度均值 < 70 → 很可能是深色车身/阴影
        if mean_val < 70:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

        # ── 过滤3: HSV饱和度检测 ──
        # 深色车漆即使很暗也有颜色饱和度，黑烟是无彩色的（饱和度极低）
        # 如果饱和度较高 → 是有颜色的物体（车漆），不是烟
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        if sat_mean > 35:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

        # ── 过滤4: 均匀区域（阴影/路面） ──
        if std_val < 12:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

        # ── 过滤5: 亮色反光 ──
        if mean_val > 200:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

        # ── 过滤6: 路面对比 — 尾部必须明显暗于路面 ──
        road_mean = self._sample_road_gray(frame, rx1, ry1, rx2, ry2, bw, bh)
        if road_mean > 0 and mean_val > road_mean * 0.80:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

        # ── 过滤7: 路面污渍/水渍 ──
        if std_val < 20 and mean_val > 80:
            return {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': roi}

        # ══════════════════════════════════════════
        # 黑烟检测核心
        # ══════════════════════════════════════════
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 动态阈值：基于局部均值
        dark_thresh = max(50, int(mean_val * 0.45))
        _, dark_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)

        # 形态学开运算去噪点
        kernel = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        total = dark_mask.shape[0] * dark_mask.shape[1]
        ratio = cv2.countNonZero(dark_mask) / total if total > 0 else 0.0

        # ── 置信度门槛：ratio >= 0.7 才认为有烟 ──
        if ratio < self.threshold:
            self._counters.pop(vid, None)
            return {'grade': 0, 'ratio': ratio, 'has_smoke': False, 'roi': roi}

        # 林格曼黑度等级
        RINGELMANN = [(0.75, 1), (0.82, 2), (0.88, 3), (0.94, 4), (1.01, 5)]
        grade = 1
        for thresh, level in RINGELMANN:
            if ratio < thresh:
                grade = level
                break

        # 连续帧确认（grade >= 2 才报警）
        if grade >= 2:
            cnt = self._counters.get(vid, 0) + 1
            self._counters[vid] = cnt
            has_smoke = cnt >= self.debounce
        else:
            self._counters.pop(vid, None)
            has_smoke = False

        return {'grade': grade, 'ratio': ratio, 'has_smoke': has_smoke, 'roi': roi}

    def _sample_body_gray(self, frame, x1, y1, x2, y2):
        """采样车辆上半部分车身的平均灰度（用于与尾部对比）"""
        fh, fw = frame.shape[:2]
        bh = y2 - y1
        # 车身上半部分：从顶部往下 1/3 ~ 2/3 区域
        body_top = max(0, y1 + bh // 4)
        body_bot = max(0, y1 + bh * 2 // 3)
        pad_x = max(1, int((x2 - x1) * 0.10))
        bx1 = max(0, x1 + pad_x)
        bx2 = min(fw, x2 - pad_x)

        if body_bot <= body_top or bx2 <= bx1:
            return 0.0

        patch = frame[body_top:body_bot, bx1:bx2]
        if patch.size == 0:
            return 0.0
        return float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))

    def _sample_road_gray(self, frame, rx1, ry1, rx2, ry2, bw, bh):
        """采样车辆两侧路面区域的平均灰度，用于对比"""
        fh, fw = frame.shape[:2]
        lx1 = max(0, rx1 - bw)
        lx2 = max(0, rx1 - 5)
        rlx1 = min(fw, rx2 + 5)
        rlx2 = min(fw, rx2 + bw)

        samples = []
        if lx2 - lx1 > 10:
            patch = frame[ry1:ry2, lx1:lx2]
            if patch.size > 0:
                samples.append(float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))))
        if rlx2 - rlx1 > 10:
            patch = frame[ry1:ry2, rlx1:rlx2]
            if patch.size > 0:
                samples.append(float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))))

        return sum(samples) / len(samples) if samples else 0.0

    def cleanup(self, active_ids):
        stale = [v for v in self._counters if v not in active_ids]
        for v in stale:
            del self._counters[v]
