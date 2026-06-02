# -*- coding: utf-8 -*-
"""
智能车辆检测程序 - Web 版
基于 YOLOv8 + OpenCV + Flask
功能：浏览器打开控制面板，实时切换「车速检测」「车流热力图」「车距预警」三种模式
运行后自动打开浏览器 http://localhost:5000
"""

# ============================================================
#  一、所有可调参数统一在此处配置
# ============================================================

# 视频文件列表（可添加多个视频）
VIDEO_LIST = [
    "./traffic_rideo.mp4",
    "./road_redio.mp4",
]
# 默认播放第一个视频
VIDEO_PATH = VIDEO_LIST[0]

# YOLOv8 模型路径（使用 yolov8n 轻量模型）
MODEL_PATH = "./yolov8n.pt"

# 车辆类别 ID（COCO 数据集中：2=轿车, 5=大巴, 7=货车）
VEHICLE_CLASS_IDS = [2, 5, 7]

# ----- 测速线参数（竖直线，车辆水平穿越） -----
# 左侧测速线的 X 坐标（像素）
LINE_X_LEFT = 150
# 右侧测速线的 X 坐标（像素）
LINE_X_RIGHT = 400
# 测速线颜色（蓝色 BGR）
LINE_COLOR = (255, 0, 0)
# 测速线粗细
LINE_THICKNESS = 2

# ----- 像素距离 → 实际距离换算 -----
# 两条测速线之间的实际距离（米）
REAL_DISTANCE_M = 30.0
# 两条测速线之间的像素距离（根据实际视频调整）
PIXEL_DISTANCE = abs(LINE_X_RIGHT - LINE_X_LEFT)

# ----- 热力图参数 -----
# 热力图衰减系数（0~1，越大衰减越慢，保留历史越久）
HEAT_DECAY = 0.95
# 热力图叠加透明度（0~1，越大越不透明）
HEAT_ALPHA = 0.5

# ----- 车距预警参数 -----
# 安全车距阈值（米），低于此值判定为危险车距
SAFE_DISTANCE_M = 20.0

# ----- 黑烟检测参数 -----
# 黑烟像素占比阈值（0~1），超过此值判定为有黑烟
SMOKE_THRESHOLD = 0.12
# 超标临界林格曼黑度等级（0~5），≥此等级判定尾气超标
LINGERMAN_LEVEL = 2

# ----- 检测参数 -----
# YOLO 置信度阈值
CONF_THRESHOLD = 0.4
# YOLO NMS IoU 阈值
IOU_THRESHOLD = 0.5

# ----- Web 服务参数 -----
# Flask 服务端口
FLASK_PORT = 5000
# MJPEG 推送帧率上限（控制浏览器端流畅度）
STREAM_FPS = 25

# ============================================================
#  二、自动安装依赖
# ============================================================
import subprocess
import sys

def ensure_package(pkg_name, import_name=None):
    """确保 Python 包已安装，未安装则自动安装"""
    imp = import_name or pkg_name
    try:
        __import__(imp)
    except ImportError:
        print(f"[自动安装] 正在安装 {pkg_name} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg_name, "-q"]
        )

ensure_package("ultralytics")
ensure_package("opencv-python", "cv2")
ensure_package("numpy")
ensure_package("flask")

# ============================================================
#  三、正式导入
# ============================================================
import os
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from ultralytics import YOLO

# ============================================================
#  四、车辆跟踪器（用于车速计算）
# ============================================================
class VehicleTracker:
    """
    基于位移的车辆测速器。
    跟踪车辆中心点位置，当像素位移超过阈值后计算瞬时车速。
    适用于任意行驶方向（水平/垂直/斜向）。
    """

    # 像素/米 换算系数（经验值：1.5m摄像头高度，1000等效焦距）
    PX_PER_M = 1000.0 / 150.0
    # 最少跟踪帧数（防止单帧噪声）
    MIN_TRACK_FRAMES = 10
    # 最小位移（像素），低于此不计算速度
    MIN_DISPLACEMENT = 5

    def __init__(self):
        self.speeds = {}          # {track_id: speed_kmh}  已确认车速
        self._next_id = 0
        self._prev_centroids = []
        self._max_match_dist = 30
        # 每辆车的跟踪轨迹 {tid: {'start_cx', 'start_cy', 'start_frame', 'last_cx', 'last_cy', 'frames'}}
        self._track_data = {}
        self._current_frame = 0

    def _assign_ids(self, detections):
        if not detections:
            self._prev_centroids = []
            return []
        current_pts = np.array([(d[0], d[1]) for d in detections])
        prev_pts = np.array([(p[0], p[1]) for p in self._prev_centroids]) if self._prev_centroids else np.empty((0, 2))
        assigned = []
        used_prev = set()
        used_curr = set()
        if len(prev_pts) > 0:
            dists = np.linalg.norm(current_pts[:, np.newaxis, :] - prev_pts[np.newaxis, :, :], axis=2)
            for _ in range(min(len(current_pts), len(prev_pts))):
                min_val = np.inf
                ci, pi = -1, -1
                for i in range(dists.shape[0]):
                    if i in used_curr:
                        continue
                    for j in range(dists.shape[1]):
                        if j in used_prev:
                            continue
                        if dists[i, j] < min_val:
                            min_val = dists[i, j]
                            ci, pi = i, j
                if min_val > self._max_match_dist or ci == -1:
                    break
                tid = self._prev_centroids[pi][2]
                cx, cy, cls_name = detections[ci]
                assigned.append((cx, cy, cls_name, tid))
                used_curr.add(ci)
                used_prev.add(pi)
        for i, (cx, cy, cls_name) in enumerate(detections):
            if i not in used_curr:
                assigned.append((cx, cy, cls_name, self._next_id))
                self._next_id += 1
        self._prev_centroids = [(cx, cy, tid) for cx, cy, _, tid in assigned]
        return assigned

    def update(self, tracked, fps=24.0):
        """
        根据跟踪结果更新速度。
        tracked: [(cx, cy, cls_name, track_id), ...]
        返回: {track_id: speed_kmh}  本帧新计算出的车速
        """
        self._current_frame += 1
        new_speeds = {}
        active_tids = set()

        for cx, cy, cls_name, tid in tracked:
            active_tids.add(tid)
            if tid not in self._track_data:
                self._track_data[tid] = {
                    'start_cx': cx, 'start_cy': cy,
                    'start_frame': self._current_frame,
                    'last_cx': cx, 'last_cy': cy,
                    'frames': 1
                }
            else:
                td = self._track_data[tid]
                td['last_cx'] = cx
                td['last_cy'] = cy
                td['frames'] += 1
                # 仅当该车辆尚未测速时才计算
                if tid not in self.speeds:
                    dx = cx - td['start_cx']
                    dy = cy - td['start_cy']
                    displacement = (dx * dx + dy * dy) ** 0.5
                    frame_span = self._current_frame - td['start_frame']
                    if displacement >= self.MIN_DISPLACEMENT and frame_span >= self.MIN_TRACK_FRAMES:
                        elapsed = frame_span / fps
                        if elapsed > 0:
                            dist_m = displacement / self.PX_PER_M
                            speed_ms = dist_m / elapsed
                            speed_kmh = speed_ms * 3.6
                            if 5 < speed_kmh < 250:
                                self.speeds[tid] = round(speed_kmh, 1)
                                new_speeds[tid] = self.speeds[tid]

        # 清理不再活跃的轨迹
        stale = [tid for tid in self._track_data if tid not in active_tids]
        for tid in stale:
            del self._track_data[tid]

        return new_speeds

    def get_speed(self, tid):
        return self.speeds.get(tid)

    def reset(self):
        """重置跟踪器（视频循环时调用）"""
        self._track_data.clear()
        self._prev_centroids.clear()
        self._current_frame = 0

    def get_tracked_with_speed(self, detections, fps=24.0):
        tracked = self._assign_ids(detections)
        new_speeds = self.update(tracked, fps)
        result = []
        for cx, cy, cls_name, tid in tracked:
            spd = self.get_speed(tid)
            result.append((cx, cy, cls_name, tid, spd))
        return result, new_speeds


# ============================================================
#  五、热力图生成器
# ============================================================
class HeatmapGenerator:
    """
    累积车辆位置信息，生成带衰减效果的动态热力图。
    """

    def __init__(self, frame_w, frame_h):
        self.w = frame_w
        self.h = frame_h
        self.heat_map = np.zeros((frame_h, frame_w), dtype=np.float32)

    def update(self, detections):
        """根据当前帧的检测结果更新热力图"""
        self.heat_map *= HEAT_DECAY
        for cx, cy, _ in detections:
            ix, iy = int(cx), int(cy)
            if 0 <= ix < self.w and 0 <= iy < self.h:
                r = 30
                y_lo = max(0, iy - r)
                y_hi = min(self.h, iy + r + 1)
                x_lo = max(0, ix - r)
                x_hi = min(self.w, ix + r + 1)
                yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
                gauss = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * (r / 2) ** 2))
                self.heat_map[y_lo:y_hi, x_lo:x_hi] += gauss.astype(np.float32)

    def render(self, frame):
        """将热力图渲染叠加到原始帧上"""
        hmax = self.heat_map.max()
        if hmax > 0:
            normed = (self.heat_map / hmax * 255).astype(np.uint8)
        else:
            normed = np.zeros_like(self.heat_map, dtype=np.uint8)
        colored = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 1.0, colored, HEAT_ALPHA, 0)
        return overlay


# ============================================================
#  六、核心处理引擎（四种模式复用同一检测流程）
# ============================================================
class DetectionEngine:
    """
    核心检测引擎：加载 YOLO 模型，执行推理，提供统一接口。
    速度模式和热力图模式共用同一个检测循环，避免重复计算。
    """

    def __init__(self):
        print("[初始化] 正在加载 YOLOv8n 模型...")
        self.model = YOLO(MODEL_PATH)
        self.tracker = VehicleTracker()
        self.heatmap = None
        self.mode = "speed"  # "speed" / "heatmap" / "distance" / "smoke"
        self._video_fps = 24.0
        # 黑烟统计累计数据
        self._smoke_exceed_vehicles = 0   # 超标车辆总数（跨帧累加）
        self._smoke_exceed_frames = 0     # 出现超标的总帧数
        self._smoke_level_sum = 0.0       # 所有车辆林格曼等级之和（用于算平均）
        self._smoke_vehicle_count = 0     # 已检测车辆总数（用于算平均）
        # 全局统计累计数据（供大屏看板使用）
        self._total_vehicles_all = 0      # 累计通行车辆总数
        self._speed_sum_all = 0.0         # 车速累加（用于算平均）
        self._speed_count_all = 0         # 车速计数（用于算平均）
        self._danger_vehicles_all = 0     # 累计危险车距车辆数
        self._distance_sum_all = 0.0      # 车距累加（用于算平均）
        self._distance_count_all = 0      # 车距计数（用于算平均）

    def detect_vehicles(self, frame):
        """对单帧执行 YOLO 检测，返回车辆信息列表"""
        results = self.model.predict(
            frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=VEHICLE_CLASS_IDS, verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, cls_name, (x1, y1, x2, y2)))
        return detections

    def process_frame(self, frame):
        """处理一帧：执行检测 + 追踪 + 速度计算 + 根据当前模式渲染"""
        h, w = frame.shape[:2]
        if self.heatmap is None:
            self.heatmap = HeatmapGenerator(w, h)

        detections = self.detect_vehicles(frame)
        det_for_track = [(cx, cy, cls) for cx, cy, cls, _ in detections]

        # 每帧都做跟踪和测速，保证速度数据持续更新
        fps = self._video_fps if hasattr(self, '_video_fps') and self._video_fps > 0 else 24.0
        tracked, new_speeds = self.tracker.get_tracked_with_speed(det_for_track, fps)
        for tid, spd in new_speeds.items():
            self._speed_sum_all += spd
            self._speed_count_all += 1
        self._total_vehicles_all = max(self._total_vehicles_all, self.tracker._next_id)

        if self.mode == "speed":
            return self._render_speed(frame, detections, tracked)
        elif self.mode == "distance":
            return self._render_distance(frame, detections, det_for_track)
        elif self.mode == "smoke":
            return self._render_smoke(frame, detections, det_for_track)
        else:
            return self._render_heatmap(frame, detections, det_for_track)

    def _render_speed(self, frame, detections, tracked):
        """车速检测模式渲染"""
        bbox_map = {}
        for i, (cx, cy, cls_name, _) in enumerate(detections):
            for tcx, tcy, _, tid, _ in tracked:
                if abs(cx - tcx) < 5 and abs(cy - tcy) < 5:
                    bbox_map[tid] = (cx, cy, cls_name, detections[i][3])
                    break

        overlay = frame.copy()

        # 绘制每个车辆的检测框和速度
        for cx, cy, cls_name, tid, spd in tracked:
            if tid in bbox_map:
                _, _, cname, (x1, y1, x2, y2) = bbox_map[tid]
            else:
                x1, y1 = cx - 30, cy - 30
                x2, y2 = cx + 30, cy + 30
                cname = cls_name

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{cname} #{tid}"
            if spd is not None:
                label += f" {spd}km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 255), -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        info = f"MODE: SPEED | Vehicles: {len(tracked)}"
        cv2.putText(overlay, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return overlay

    def _render_heatmap(self, frame, detections, det_for_track):
        """热力图模式渲染"""
        self.heatmap.update(det_for_track)
        overlay = self.heatmap.render(frame)
        info = f"MODE: HEATMAP | Vehicles: {len(detections)}"
        cv2.putText(overlay, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return overlay

    def _render_distance(self, frame, detections, det_for_track):
        """
        车距预警模式渲染。
        按车辆在画面中的纵向位置（Y 坐标）排序，
        计算相邻车辆之间的实际距离，低于安全阈值则红色警告。
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 像素 → 米 的换算比例（基于摄像头参数估算）
        px_to_m = 1.5 * 1000 / (h * 100) if h > 0 else 0.1

        if len(detections) < 2:
            # 车辆不足 2 辆，无法计算车距，仅画框
            for cx, cy, cls_name, (x1, y1, x2, y2) in detections:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(overlay, cls_name, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            info = f"MODE: DISTANCE | Vehicles: {len(detections)}"
            cv2.putText(overlay, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return overlay

        # 按 Y 坐标从小到大排序（画面顶部的车在前，底部的车在后）
        sorted_dets = sorted(detections, key=lambda d: d[1])

        # 记录每辆车与其前车的距离
        vehicle_dists = [None] * len(sorted_dets)
        for i in range(1, len(sorted_dets)):
            # 前车底部 Y 坐标 vs 当前车底部 Y 坐标（用 bbox 的 y2）
            _, _, _, (_, _, _, y2_prev) = sorted_dets[i - 1]
            _, _, _, (_, _, _, y2_curr) = sorted_dets[i]
            pixel_gap = abs(y2_curr - y2_prev)
            real_gap = pixel_gap * px_to_m
            vehicle_dists[i] = round(real_gap, 1)

        # 绘制每辆车的检测框和距离信息
        for i, (cx, cy, cls_name, (x1, y1, x2, y2)) in enumerate(sorted_dets):
            dist = vehicle_dists[i]

            if dist is None:
                # 第一辆车（无前车），绿色框
                color = (0, 255, 0)
                label = f"{cls_name} (first)"
            elif dist < SAFE_DISTANCE_M:
                # 危险车距，红色框
                color = (0, 0, 255)
                label = f"{cls_name} {dist}m !!DANGER"
            else:
                # 安全车距，绿色框
                color = (0, 255, 0)
                label = f"{cls_name} {dist}m OK"

            # 画检测框
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # 标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 如果危险，在框下方额外标注红色警告文字
            if dist is not None and dist < SAFE_DISTANCE_M:
                warn = "DANGER"
                cv2.putText(overlay, warn, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # 左上角模式信息 + 累计统计
        danger_count = sum(1 for d in vehicle_dists if d is not None and d < SAFE_DISTANCE_M)
        self._danger_vehicles_all += danger_count
        for d in vehicle_dists:
            if d is not None:
                self._distance_sum_all += d
                self._distance_count_all += 1
        info = f"MODE: DISTANCE | Vehicles: {len(sorted_dets)} | DANGER: {danger_count}"
        cv2.putText(overlay, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return overlay

    def _render_smoke(self, frame, detections, det_for_track):
        """
        黑烟尾气检测模式渲染。
        ① YOLO 框选车辆后截取车尾区域（bbox 底部 30%）
        ② 灰度 + 自适应阈值分割识别黑烟区域
        ③ 统计黑烟像素占比，换算林格曼黑度（0~5 级）
        ④ ≥ LINGERMAN_LEVEL 判定尾气超标，紫色粗框 + 紫圈标注
        ⑤ 正常车辆绿色框
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 林格曼黑度等级对照表：(像素占比阈值, 等级)
        # 占比 < 3% → 0级, < 8% → 1级, < 15% → 2级, < 25% → 3级, < 40% → 4级, ≥ 40% → 5级
        RINGELMANN_THRESHOLDS = [
            (0.03, 0), (0.08, 1), (0.15, 2),
            (0.25, 3), (0.40, 4), (1.01, 5)
        ]

        frame_exceed = False  # 本帧是否有超标车辆

        for cx, cy, cls_name, (x1, y1, x2, y2) in detections:
            # 确保 bbox 在画面内
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)

            # 截取车尾区域：bbox 底部 30%（假设车辆朝上行驶，车尾在底部）
            tail_h = max(1, int((y2c - y1c) * 0.3))
            tail_y1 = y2c - tail_h
            tail_roi = frame[tail_y1:y2c, x1c:x2c]

            # 黑烟检测：灰度 + 自适应阈值
            smoke_ratio = 0.0
            ringelmann_level = 0

            if tail_roi.size > 0 and tail_roi.shape[0] > 5 and tail_roi.shape[1] > 5:
                gray = cv2.cvtColor(tail_roi, cv2.COLOR_BGR2GRAY)
                # 自适应阈值：低于均值 - 15 的区域视为暗区（黑烟）
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 25, 10
                )
                # 额外用固定阈值过滤：灰度值 < 80 的深色像素
                _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
                # 取两种方法的交集，减少误检
                smoke_mask = cv2.bitwise_and(binary, dark_mask)
                smoke_pixels = cv2.countNonZero(smoke_mask)
                total_pixels = smoke_mask.shape[0] * smoke_mask.shape[1]
                smoke_ratio = smoke_pixels / total_pixels if total_pixels > 0 else 0.0

            # 换算林格曼黑度等级
            for thresh, level in RINGELMANN_THRESHOLDS:
                if smoke_ratio < thresh:
                    ringelmann_level = level
                    break

            # 更新统计
            self._smoke_level_sum += ringelmann_level
            self._smoke_vehicle_count += 1

            is_exceed = ringelmann_level >= LINGERMAN_LEVEL
            if is_exceed:
                self._smoke_exceed_vehicles += 1
                frame_exceed = True

            # ---- 渲染 ----
            if is_exceed:
                # 超标：紫色粗框
                purple = (200, 50, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), purple, 3)

                # 车尾黑烟区域画半透明紫圈
                tail_cx = (x1 + x2) // 2
                tail_cy = y2 - tail_h // 2
                radius = max(tail_h // 2, 15)
                # 创建半透明叠加层
                circle_overlay = overlay.copy()
                cv2.circle(circle_overlay, (tail_cx, tail_cy), radius, purple, -1)
                cv2.addWeighted(circle_overlay, 0.35, overlay, 0.65, 0, overlay)

                # 标签
                label = f"{cls_name} Lv{ringelmann_level} SMOKE"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), purple, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # 正常：绿色框
                green = (0, 255, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), green, 2)
                label = f"{cls_name} Lv{ringelmann_level} OK"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), green, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if frame_exceed:
            self._smoke_exceed_frames += 1

        # 左上角统计信息
        avg_level = (self._smoke_level_sum / self._smoke_vehicle_count
                     if self._smoke_vehicle_count > 0 else 0.0)
        info1 = f"MODE: SMOKE | Vehicles: {len(detections)}"
        info2 = f"Exceed: {self._smoke_exceed_vehicles} | Frames: {self._smoke_exceed_frames} | AvgLv: {avg_level:.1f}"
        cv2.putText(overlay, info1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 255), 2)
        cv2.putText(overlay, info2, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 50, 255), 1)

        return overlay

    def set_mode(self, mode):
        """切换模式"""
        self.mode = mode
        name_map = {"speed": "车速检测", "heatmap": "车流热力图",
                    "distance": "车距预警", "smoke": "黑烟检测"}
        print(f"[切换] 当前模式: {name_map.get(mode, mode)}")

    def reset(self):
        """重置跟踪器、热力图和黑烟统计（切换视频时调用）"""
        self.tracker = VehicleTracker()
        self.heatmap = None
        self._smoke_exceed_vehicles = 0
        self._smoke_exceed_frames = 0
        self._smoke_level_sum = 0.0
        self._smoke_vehicle_count = 0
        self._total_vehicles_all = 0
        self._speed_sum_all = 0.0
        self._speed_count_all = 0
        self._danger_vehicles_all = 0
        self._distance_sum_all = 0.0
        self._distance_count_all = 0

    def get_stats(self):
        """
        返回当前所有统计数据的字典，供大屏看板 API 使用。
        """
        # 平均车速：取引擎层累计的车速均值
        avg_speed = round(self._speed_sum_all / self._speed_count_all, 1) if self._speed_count_all > 0 else 0.0

        # 平均车距
        avg_distance = round(self._distance_sum_all / self._distance_count_all, 1) if self._distance_count_all > 0 else 0.0

        # 平均林格曼等级
        avg_level = round(self._smoke_level_sum / self._smoke_vehicle_count, 1) if self._smoke_vehicle_count > 0 else 0.0

        # 当前模式中文名
        mode_names = {
            "speed": "车速检测", "heatmap": "车流热力图",
            "distance": "车距预警", "smoke": "黑烟检测"
        }

        return {
            "total_vehicles": self._total_vehicles_all,
            "avg_speed": avg_speed,
            "danger_vehicles": self._danger_vehicles_all,
            "avg_distance": avg_distance,
            "smoke_exceed": self._smoke_exceed_vehicles,
            "avg_ringelmann": avg_level,
            "current_mode": mode_names.get(self.mode, self.mode),
            "mode_key": self.mode,
        }


# ============================================================
#  七、视频流生成器（Flask MJPEG 推送）
# ============================================================
class VideoStreamer:
    """
    后台线程持续读取视频帧 + YOLO 检测 + 编码 JPEG，
    通过 Flask Response 以 MJPEG 格式推送到浏览器。
    支持实时切换视频源和检测模式。
    """

    def __init__(self, engine: DetectionEngine):
        self.engine = engine
        self._current_index = 0
        self._cap = None
        self._lock = threading.Lock()
        self._frame_jpeg = None  # 最新编码好的 JPEG bytes
        self._running = True
        self._total_frames = 0   # 已处理总帧数

        # 打开默认视频
        self._open_video(VIDEO_LIST[0])

        # 启动后台采集线程
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open_video(self, path):
        """打开视频文件"""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            print(f"[错误] 无法打开视频: {path}")
            return False
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.engine._video_fps = fps
        print(f"[视频] {os.path.basename(path)} | {w}x{h} @ {fps:.1f}fps, 共 {total} 帧")
        return True

    def _capture_loop(self):
        """后台线程：持续读帧 → 检测 → 编码 JPEG"""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                # 视频播放完毕，循环：重置跟踪器以重新测速
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.engine.tracker.reset()
                continue

            # YOLO 检测 + 渲染
            result = self.engine.process_frame(frame)
            self._total_frames += 1

            # 编码为 JPEG（quality=80 平衡质量和带宽）
            _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 80])

            with self._lock:
                self._frame_jpeg = buf.tobytes()

            # 控制采集帧率
            time.sleep(1.0 / STREAM_FPS)

    def switch_video(self, index):
        """切换视频源"""
        if 0 <= index < len(VIDEO_LIST) and index != self._current_index:
            with self._lock:
                self._current_index = index
                if self._open_video(VIDEO_LIST[index]):
                    self.engine.reset()
                    self._total_frames = 0
                    print(f"[切换] 视频: {os.path.basename(VIDEO_LIST[index])}")
                    return True
                else:
                    # 切换失败，重新打开原视频
                    self._open_video(VIDEO_LIST[self._current_index])
                    return False
        return True

    def get_frame(self):
        """获取最新一帧的 JPEG bytes"""
        with self._lock:
            return self._frame_jpeg

    def get_current_index(self):
        """获取当前视频索引"""
        return self._current_index

    def get_total_frames(self):
        """获取已处理总帧数"""
        return self._total_frames

    def stop(self):
        """停止采集"""
        self._running = False
        if self._cap is not None:
            self._cap.release()


# ============================================================
#  八、Flask Web 应用
# ============================================================
app = Flask(__name__)

# 全局对象（在 main() 中初始化）
engine = None
streamer = None

# HTML 页面模板（内嵌，无需额外文件）
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能车辆检测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e; color: #eee;
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 18px 30px;
            display: flex; align-items: center; justify-content: space-between;
            box-shadow: 0 2px 12px rgba(0,0,0,0.4);
        }
        .header h1 { font-size: 22px; font-weight: 700; letter-spacing: 1px; }
        .header .status {
            font-size: 13px; color: #8ecae6;
            display: flex; gap: 18px;
        }
        .header .status span { display: inline-flex; align-items: center; gap: 5px; }
        .header .status .dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #4ade80; display: inline-block;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%,100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .main {
            display: flex; gap: 20px;
            padding: 20px; max-width: 1400px; margin: 0 auto;
        }
        .video-container {
            flex: 1; background: #16213e; border-radius: 12px;
            overflow: hidden; position: relative;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .video-container img {
            width: 100%; display: block;
        }
        .video-label {
            position: absolute; top: 10px; right: 14px;
            background: rgba(0,0,0,0.6); padding: 4px 12px;
            border-radius: 6px; font-size: 12px; color: #a0d2db;
        }
        .panel {
            width: 280px; background: #16213e;
            border-radius: 12px; padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            display: flex; flex-direction: column; gap: 14px;
        }
        .panel h2 {
            font-size: 16px; color: #e2e8f0;
            border-bottom: 1px solid #2a3a5c;
            padding-bottom: 10px; text-align: center;
        }
        .btn-group { display: flex; flex-direction: column; gap: 10px; }
        .btn-group label { font-size: 13px; color: #94a3b8; margin-bottom: 2px; }
        .btn {
            padding: 12px 16px; border: none; border-radius: 8px;
            font-size: 14px; font-weight: 600; cursor: pointer;
            transition: all 0.2s; color: #fff;
            font-family: inherit;
        }
        .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .btn:active { transform: translateY(0); }
        .btn-speed { background: linear-gradient(135deg, #1a73e8, #1557b0); }
        .btn-speed.active { box-shadow: 0 0 0 3px rgba(26,115,232,0.5); }
        .btn-heat  { background: linear-gradient(135deg, #e8501a, #b03c14); }
        .btn-heat.active  { box-shadow: 0 0 0 3px rgba(232,80,26,0.5); }
        .btn-dist  { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .btn-dist.active  { box-shadow: 0 0 0 3px rgba(245,158,11,0.5); }
        .btn-smoke { background: linear-gradient(135deg, #9333ea, #7c3aed); }
        .btn-smoke.active { box-shadow: 0 0 0 3px rgba(147,51,234,0.5); }
        .btn-vid   { background: linear-gradient(135deg, #3a7d44, #2d6334); }
        .btn-vid.active   { box-shadow: 0 0 0 3px rgba(58,125,68,0.5); }
        .btn-vids { display: flex; gap: 8px; }
        .btn-vids .btn { flex: 1; font-size: 12px; padding: 10px 8px; }
        .info-box {
            background: #0f3460; border-radius: 8px; padding: 12px;
            font-size: 12px; color: #94a3b8; line-height: 1.8;
        }
        .info-box b { color: #e2e8f0; }
        .footer {
            text-align: center; padding: 14px;
            font-size: 12px; color: #475569;
        }
        /* 快捷键提示 */
        kbd {
            background: #2a3a5c; border-radius: 3px;
            padding: 1px 6px; font-size: 11px; color: #94a3b8;
            border: 1px solid #3a4a6c;
        }
        /* ===== 大屏数据看板 ===== */
        .dashboard {
            max-width: 1400px; margin: 0 auto;
            padding: 0 20px 20px;
        }
        .dashboard h3 {
            font-size: 15px; color: #8ecae6; margin-bottom: 14px;
            padding-left: 4px; letter-spacing: 1px;
            border-left: 3px solid #8ecae6; padding-left: 10px;
        }
        .cards {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
        }
        .card {
            background: linear-gradient(145deg, #16213e, #1a1a2e);
            border-radius: 12px; padding: 18px 16px;
            border: 1px solid #2a3a5c;
            display: flex; flex-direction: column; gap: 6px;
            position: relative; overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .card .card-icon {
            position: absolute; top: 12px; right: 14px;
            font-size: 28px; opacity: 0.15;
        }
        .card .card-label {
            font-size: 12px; color: #64748b; letter-spacing: 0.5px;
        }
        .card .card-value {
            font-size: 28px; font-weight: 700;
            font-variant-numeric: tabular-nums;
        }
        .card .card-unit {
            font-size: 12px; color: #64748b; margin-top: 2px;
        }
        /* 每张卡片独立渐变色 */
        .card-blue   { border-top: 3px solid #3b82f6; }
        .card-blue   .card-value { color: #60a5fa; }
        .card-green  { border-top: 3px solid #22c55e; }
        .card-green  .card-value { color: #4ade80; }
        .card-red    { border-top: 3px solid #ef4444; }
        .card-red    .card-value { color: #f87171; }
        .card-amber  { border-top: 3px solid #f59e0b; }
        .card-amber  .card-value { color: #fbbf24; }
        .card-purple { border-top: 3px solid #a855f7; }
        .card-purple .card-value { color: #c084fc; }
        .card-pink   { border-top: 3px solid #ec4899; }
        .card-pink   .card-value { color: #f472b6; }
        .card-cyan   { border-top: 3px solid #06b6d4; }
        .card-cyan   .card-value { color: #22d3ee; }
        .card-slate  { border-top: 3px solid #64748b; }
        .card-slate  .card-value { color: #94a3b8; }
        @media (max-width: 900px) {
            .cards { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>智能车辆检测系统</h1>
        <div class="status">
            <span><i class="dot"></i> 实时运行中</span>
            <span id="modeLabel">模式：车速检测</span>
            <span id="videoLabel">视频：traffic_rideo.mp4</span>
        </div>
    </div>

    <div class="main">
        <!-- 视频流 -->
        <div class="video-container">
            <img id="stream" src="/video_feed" alt="视频流">
            <div class="video-label" id="streamInfo">LIVE</div>
        </div>

        <!-- 控制面板 -->
        <div class="panel">
            <h2>控制面板</h2>

            <div class="btn-group">
                <label>检测模式</label>
                <button class="btn btn-speed active" onclick="switchMode('speed')">车速检测</button>
                <button class="btn btn-heat" onclick="switchMode('heatmap')">车流热力图</button>
                <button class="btn btn-dist" onclick="switchMode('distance')">车距预警</button>
                <button class="btn btn-smoke" onclick="switchMode('smoke')">黑烟检测</button>
            </div>

            <div class="btn-group">
                <label>视频源</label>
                <div class="btn-vids">
                    <button class="btn btn-vid active" id="vid0" onclick="switchVideo(0)">视频1</button>
                    <button class="btn btn-vid" id="vid1" onclick="switchVideo(1)">视频2</button>
                </div>
            </div>

            <div class="info-box">
                <b>快捷键</b><br>
                <kbd>1</kbd> 车速检测 &nbsp;
                <kbd>2</kbd> 热力图 &nbsp;
                <kbd>3</kbd> 车距预警 &nbsp;
                <kbd>4</kbd> 黑烟检测<br>
                <kbd>Q</kbd> / <kbd>W</kbd> 切换视频<br><br>
                <b>视频列表</b><br>
                1. traffic_rideo.mp4<br>
                2. road_redio.mp4
            </div>
        </div>
    </div>

    <!-- ===== 实时数据看板 ===== -->
    <div class="dashboard">
        <h3>实时数据统计</h3>
        <div class="cards">
            <div class="card card-blue">
                <span class="card-icon">🚗</span>
                <span class="card-label">累计通行车辆</span>
                <span class="card-value" id="statVehicles">0</span>
                <span class="card-unit">辆</span>
            </div>
            <div class="card card-green">
                <span class="card-icon">⚡</span>
                <span class="card-label">平均车速</span>
                <span class="card-value" id="statSpeed">0</span>
                <span class="card-unit">km/h</span>
            </div>
            <div class="card card-red">
                <span class="card-icon">⚠️</span>
                <span class="card-label">累计危险车辆</span>
                <span class="card-value" id="statDanger">0</span>
                <span class="card-unit">辆</span>
            </div>
            <div class="card card-amber">
                <span class="card-icon">📏</span>
                <span class="card-label">平均车距</span>
                <span class="card-value" id="statDistance">0</span>
                <span class="card-unit">米</span>
            </div>
            <div class="card card-purple">
                <span class="card-icon">💨</span>
                <span class="card-label">黑烟超标总数</span>
                <span class="card-value" id="statSmoke">0</span>
                <span class="card-unit">辆次</span>
            </div>
            <div class="card card-pink">
                <span class="card-icon">🔬</span>
                <span class="card-label">平均林格曼等级</span>
                <span class="card-value" id="statRingelmann">0</span>
                <span class="card-unit">级 (0~5)</span>
            </div>
            <div class="card card-cyan">
                <span class="card-icon">🎯</span>
                <span class="card-label">当前运行模式</span>
                <span class="card-value" id="statMode" style="font-size:20px;">--</span>
                <span class="card-unit" id="statModeKey">--</span>
            </div>
            <div class="card card-slate">
                <span class="card-icon">📊</span>
                <span class="card-label">视频总处理帧数</span>
                <span class="card-value" id="statFrames">0</span>
                <span class="card-unit">帧</span>
            </div>
        </div>
    </div>

    <div class="footer">
        Powered by YOLOv8 + OpenCV + Flask &nbsp;|&nbsp;
        按 <kbd>Q</kbd><kbd>W</kbd> 切换视频，<kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>4</kbd> 切换模式
    </div>

    <script>
        const videoNames = ["traffic_rideo.mp4", "road_redio.mp4"];
        let currentVideo = 0;

        function switchMode(mode) {
            fetch('/api/switch_mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            }).then(r => r.json()).then(d => {
                const names = {speed: '车速检测', heatmap: '车流热力图', distance: '车距预警', smoke: '黑烟检测'};
                document.getElementById('modeLabel').textContent = '模式：' + names[mode];
                // 更新按钮样式
                document.querySelector('.btn-speed').classList.toggle('active', mode === 'speed');
                document.querySelector('.btn-heat').classList.toggle('active', mode === 'heatmap');
                document.querySelector('.btn-dist').classList.toggle('active', mode === 'distance');
                document.querySelector('.btn-smoke').classList.toggle('active', mode === 'smoke');
            });
        }

        function switchVideo(index) {
            fetch('/api/switch_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: index})
            }).then(r => r.json()).then(d => {
                if (d.ok) {
                    currentVideo = index;
                    document.getElementById('videoLabel').textContent =
                        '视频：' + videoNames[index];
                    document.getElementById('vid0').classList.toggle('active', index === 0);
                    document.getElementById('vid1').classList.toggle('active', index === 1);
                }
            });
        }

        // 键盘快捷键
        document.addEventListener('keydown', function(e) {
            if (e.key === '1') switchMode('speed');
            else if (e.key === '2') switchMode('heatmap');
            else if (e.key === '3') switchMode('distance');
            else if (e.key === '4') switchMode('smoke');
            else if (e.key.toLowerCase() === 'q') switchVideo(0);
            else if (e.key.toLowerCase() === 'w') switchVideo(1);
        });

        // ===== 实时数据看板自动刷新（每 1 秒） =====
        function refreshStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('statVehicles').textContent = d.total_vehicles;
                    document.getElementById('statSpeed').textContent = d.avg_speed;
                    document.getElementById('statDanger').textContent = d.danger_vehicles;
                    document.getElementById('statDistance').textContent = d.avg_distance;
                    document.getElementById('statSmoke').textContent = d.smoke_exceed;
                    document.getElementById('statRingelmann').textContent = d.avg_ringelmann;
                    document.getElementById('statMode').textContent = d.current_mode;
                    document.getElementById('statModeKey').textContent = d.mode_key;
                    document.getElementById('statFrames').textContent = d.total_frames;
                })
                .catch(() => {});  // 静默处理网络错误
        }
        refreshStats();                    // 页面加载后立即执行一次
        setInterval(refreshStats, 1000);   // 每 1 秒自动刷新
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页：返回控制面板 + 视频流页面"""
    return HTML_PAGE


@app.route('/video_feed')
def video_feed():
    """MJPEG 视频流端点"""
    def generate():
        while True:
            frame = streamer.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/switch_mode', methods=['POST'])
def api_switch_mode():
    """切换检测模式"""
    from flask import request
    data = request.get_json()
    mode = data.get('mode', 'speed')
    if mode in ('speed', 'heatmap', 'distance', 'smoke'):
        engine.set_mode(mode)
        return jsonify({"ok": True, "mode": mode})
    return jsonify({"ok": False, "error": "invalid mode"}), 400


@app.route('/api/switch_video', methods=['POST'])
def api_switch_video():
    """切换视频源"""
    from flask import request
    data = request.get_json()
    index = data.get('index', 0)
    success = streamer.switch_video(index)
    return jsonify({"ok": success, "index": streamer.get_current_index()})


@app.route('/api/stats')
def api_stats():
    """实时统计数据接口，供大屏看板每秒轮询"""
    stats = engine.get_stats()
    stats["total_frames"] = streamer.get_total_frames()
    return jsonify(stats)


# ============================================================
#  九、程序入口
# ============================================================
def main():
    global engine, streamer

    print("=" * 50)
    print("  智能车辆检测系统 - Web 版")
    print("  功能：车速检测 / 车流热力图 / 多视频切换")
    print(f"  浏览器打开: http://localhost:{FLASK_PORT}")
    print("=" * 50)

    # 初始化检测引擎
    engine = DetectionEngine()

    # 初始化视频流
    streamer = VideoStreamer(engine)

    # 启动 Flask（关闭 Flask 自带的日志以保持输出简洁）
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    print(f"\n[启动] 服务运行在 http://localhost:{FLASK_PORT}")
    print("[提示] 在浏览器中操作，按 Ctrl+C 退出\n")

    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)
    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    finally:
        streamer.stop()
        print("[完成] 程序已退出")


if __name__ == "__main__":
    main()
