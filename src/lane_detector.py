# -*- coding: utf-8 -*-
"""
模块: 车道检测器
功能: 使用边缘检测 + 霍夫变换检测车道线，将车辆分配到对应车道
"""

import cv2
import numpy as np
from .config import CONFIG


class LaneDetector:
    """
    车道线检测器

    流程:
    1. 灰度化 + 高斯模糊
    2. Canny 边缘检测
    3. ROI 区域裁剪 (画面下半部分)
    4. 霍夫直线检测
    5. 按斜率分类为左车道线和右车道线
    6. 根据车辆中心 X 坐标分配车道
    """

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # 检测到的车道线参数 (斜率, 截距)
        self.left_line = None   # 左车道线 (负斜率)
        self.right_line = None  # 右车道线 (正斜率)

        # 车道线历史 (用于平滑)
        self.left_history = []
        self.right_history = []
        self.history_len = 6

    def detect(self, frame):
        """
        检测车道线

        参数:
            frame: BGR 视频帧

        返回:
            (left_line, right_line) 每个为 (slope, intercept) 或 None
        """
        h, w = frame.shape[:2]

        # Step 1: 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Step 2: ROI 裁剪 (梯形区域，画面下半部分)
        roi_top = int(h * CONFIG['roi_top_ratio'])
        mask = np.zeros_like(edges)
        roi_pts = np.array([[
            (0, h),
            (w // 2 - 50, roi_top),
            (w // 2 + 50, roi_top),
            (w, h)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_pts, 255)
        masked = cv2.bitwise_and(edges, mask)

        # Step 3: 霍夫变换检测线段
        lines = cv2.HoughLinesP(
            masked, rho=2, theta=np.pi / 180,
            threshold=80, minLineLength=60, maxLineGap=30
        )

        if lines is None:
            return self.left_line, self.right_line

        # Step 4: 按斜率分类
        left_slopes = []
        left_intercepts = []
        right_slopes = []
        right_intercepts = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # 跳过垂直线
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # 过滤角度不合理的线 (太水平或太垂直)
            angle = abs(np.degrees(np.arctan(slope)))
            if angle < 15 or angle > 75:
                continue

            # 左车道线: 斜率为负 (从左下到右上)
            # 右车道线: 斜率为正 (从左上到右下)
            if slope < -0.3:
                left_slopes.append(slope)
                left_intercepts.append(intercept)
            elif slope > 0.3:
                right_slopes.append(slope)
                right_intercepts.append(intercept)

        # Step 5: 取平均值作为车道线参数
        if left_slopes:
            self.left_line = (np.mean(left_slopes), np.mean(left_intercepts))
            self.left_history.append(self.left_line)
            if len(self.left_history) > self.history_len:
                self.left_history.pop(0)
            # 时间平滑
            avg_s = np.mean([l[0] for l in self.left_history])
            avg_i = np.mean([l[1] for l in self.left_history])
            self.left_line = (avg_s, avg_i)

        if right_slopes:
            self.right_line = (np.mean(right_slopes), np.mean(right_intercepts))
            self.right_history.append(self.right_line)
            if len(self.right_history) > self.history_len:
                self.right_history.pop(0)
            avg_s = np.mean([r[0] for r in self.right_history])
            avg_i = np.mean([r[1] for r in self.right_history])
            self.right_line = (avg_s, avg_i)

        return self.left_line, self.right_line

    def assign_lane(self, vehicles):
        """
        将车辆分配到车道

        车道编号: 0=左车道, 1=中间车道, 2=右车道, -1=未知

        参数:
            vehicles: 车辆列表

        返回:
            车道映射字典 {vehicle_id: lane_id}
        """
        lane_map = {}
        mid_x = self.frame_w // 2

        for v in vehicles:
            cx = v['center'][0]
            vid = v['id']

            if self.left_line is not None and self.right_line is not None:
                # 有两条车道线时，用线的位置判断
                # 在画面底部计算两条线的 X 坐标
                y_bottom = self.frame_h - 1
                left_x = (y_bottom - self.left_line[1]) / self.left_line[0] if self.left_line[0] != 0 else 0
                right_x = (y_bottom - self.right_line[1]) / self.right_line[0] if self.right_line[0] != 0 else self.frame_w

                lane_center = (left_x + right_x) / 2

                if cx < left_x + (lane_center - left_x) * 0.3:
                    lane_map[vid] = 0  # 左车道
                elif cx > right_x - (right_x - lane_center) * 0.3:
                    lane_map[vid] = 2  # 右车道
                else:
                    lane_map[vid] = 1  # 中间车道
            else:
                # 无线检测结果时，简单按 X 坐标三等分
                if cx < self.frame_w * 0.33:
                    lane_map[vid] = 0
                elif cx < self.frame_w * 0.66:
                    lane_map[vid] = 1
                else:
                    lane_map[vid] = 2

        return lane_map

    def draw_lanes(self, frame):
        """在帧上绘制车道线"""
        overlay = frame.copy()
        h = self.frame_h
        y_top = int(h * CONFIG['roi_top_ratio'])
        y_bottom = int(h - 1)

        if self.left_line is not None:
            s, i = self.left_line
            if s != 0:
                x_top = int((y_top - i) / s)
                x_bottom = int((y_bottom - i) / s)
                cv2.line(overlay, (int(x_bottom), int(y_bottom)),
                         (int(x_top), int(y_top)), (0, 255, 0), 2)

        if self.right_line is not None:
            s, i = self.right_line
            if s != 0:
                x_top = int((y_top - i) / s)
                x_bottom = int((y_bottom - i) / s)
                cv2.line(overlay, (int(x_bottom), int(y_bottom)),
                         (int(x_top), int(y_top)), (0, 255, 0), 2)

        return overlay
