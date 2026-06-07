# -*- coding: utf-8 -*-
"""
模块: 车道检测器
功能: 边缘检测 + 霍夫变换检测车道线，按X坐标分配车道
"""

import cv2
import numpy as np
from .config import CONFIG


class LaneDetector:

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.left_line = None
        self.right_line = None
        self.left_history = []
        self.right_history = []
        self.history_len = 6

    def detect(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        roi_top = int(h * CONFIG['roi_top_ratio'])
        mask = np.zeros_like(edges)
        roi_pts = np.array([[(0, h), (w // 2 - 50, roi_top), (w // 2 + 50, roi_top), (w, h)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, rho=2, theta=np.pi / 180, threshold=80, minLineLength=60, maxLineGap=30)
        if lines is None:
            return self.left_line, self.right_line

        left_s, left_i, right_s, right_i = [], [], [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            angle = abs(np.degrees(np.arctan(slope)))
            if angle < 15 or angle > 75:
                continue
            if slope < -0.3:
                left_s.append(slope)
                left_i.append(intercept)
            elif slope > 0.3:
                right_s.append(slope)
                right_i.append(intercept)

        if left_s:
            self.left_line = (np.mean(left_s), np.mean(left_i))
            self.left_history.append(self.left_line)
            if len(self.left_history) > self.history_len:
                self.left_history.pop(0)
            self.left_line = (np.mean([l[0] for l in self.left_history]),
                              np.mean([l[1] for l in self.left_history]))

        if right_s:
            self.right_line = (np.mean(right_s), np.mean(right_i))
            self.right_history.append(self.right_line)
            if len(self.right_history) > self.history_len:
                self.right_history.pop(0)
            self.right_line = (np.mean([r[0] for r in self.right_history]),
                               np.mean([r[1] for r in self.right_history]))

        return self.left_line, self.right_line

    def assign_lane(self, vehicles):
        lane_map = {}
        for v in vehicles:
            cx = v['center'][0]
            vid = v['id']
            if self.left_line and self.right_line and self.left_line[0] != 0 and self.right_line[0] != 0:
                y_b = self.frame_h - 1
                lx = (y_b - self.left_line[1]) / self.left_line[0]
                rx = (y_b - self.right_line[1]) / self.right_line[0]
                mid = (lx + rx) / 2
                if cx < lx + (mid - lx) * 0.3:
                    lane_map[vid] = 0
                elif cx > rx - (rx - mid) * 0.3:
                    lane_map[vid] = 2
                else:
                    lane_map[vid] = 1
            else:
                if cx < self.frame_w * 0.33:
                    lane_map[vid] = 0
                elif cx < self.frame_w * 0.66:
                    lane_map[vid] = 1
                else:
                    lane_map[vid] = 2
        return lane_map
