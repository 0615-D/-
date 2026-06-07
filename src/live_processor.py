# -*- coding: utf-8 -*-
"""
实时视频处理器
功能: MJPEG 流式输出 + 实时统计 + 多模式切换 + 数据库持久化
架构: 采集线程(读帧+YOLO) + 渲染(主线程取帧时触发)
"""

import cv2
import numpy as np
import time
import threading
import uuid
import os
from PIL import ImageFont, ImageDraw, Image
from .config import CONFIG
from .vehicle_detector import VehicleDetector
from .speed_calculator import SpeedCalculator
from .distance_estimator import DistanceEstimator
from .lane_detector import LaneDetector
from .smoke_detector import SmokeDetector
from .traffic_db import TrafficDB
from .database import VehicleDB

# ==================== 中文渲染 ====================

_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
_font_cache = {}


def _get_font(size):
    if size not in _font_cache:
        for fp in _FONT_PATHS:
            if os.path.exists(fp):
                try:
                    _font_cache[size] = ImageFont.truetype(fp, size, index=0)
                    break
                except Exception:
                    continue
        if size not in _font_cache:
            _font_cache[size] = ImageFont.load_default(size)
    return _font_cache[size]


# ==================== 主处理器 ====================

class LiveProcessor:

    def __init__(self):
        self.detector = VehicleDetector()
        self.speed_calc = None
        self.dist_est = None
        self.lane_det = None
        self.smoke_det = None
        self.db = None           # VehicleDB, 由 app.py 注入
        self.traffic_db = None   # TrafficDB, 由 app.py 注入
        self.initialized = False

        self.mode = 'speed'
        self._cap = None
        self._lock = threading.Lock()
        self._frame_jpeg = None
        self._running = True
        self._switching = False
        self._total_frames = 0
        self._current_video = ''
        self._video_fps = 24.0
        self.frame_w = 0
        self.frame_h = 0

        # 任务
        self._task_id = None
        self._task_start_time = None

        # 统计
        self._total_vehicles = 0
        self._speed_sum = 0.0
        self._speed_count = 0
        self._danger_vehicles = 0
        self._distance_sum = 0.0
        self._distance_count = 0
        self._smoke_exceed = 0
        self._max_speed = 0.0
        self._max_cars = 0
        self._danger_frames = 0
        self._smoke_frames = 0
        self._detection_count = 0

        # 车道统计
        self._lane_counts = (0,)
        self._lane_names = []
        self._lane_history = []
        self._vehicle_lane_history = {}
        self._lane_boundaries = []
        self._lane_frame_count = 0
        self._flow_history = []
        self._last_violations = []

        # 帧计数
        self._frame_num = 0

        # 异步检测
        self._detect_thread = None
        self._detect_frame = None
        self._last_vehicles = []

        # 距离缓存（由检测线程更新）
        self._last_distances = {}
        self._last_leading = {}

        # 交通状态缓存
        self._traffic_cache_key = None
        self._traffic_cache_img = None

        # ===== 车辆生命周期跟踪（用于数据库写入）=====
        # {track_id: {enter_time, speeds[], max_speed, distances[], has_smoke, smoke_grade, lane, frame_count}}
        self._vehicle_stats = {}
        # 已入库的 track_id 集合，防止重复写入
        self._saved_vehicles = set()
        # 已标记为危险的车辆集合，防止重复计数
        self._danger_vehicle_ids = set()

        # 车道统计定时器
        self._last_lane_stat_time = 0

        # 启动采集线程
        threading.Thread(target=self._capture_loop, daemon=True).start()

    # ==================== 数据库注入 ====================

    def set_databases(self, db, traffic_db):
        self.db = db
        self.traffic_db = traffic_db

    # ==================== 模块初始化 ====================

    def _init_modules(self, w, h):
        self.speed_calc = SpeedCalculator(h)
        self.dist_est = DistanceEstimator(h)
        self.lane_det = LaneDetector(w, h)
        self.smoke_det = SmokeDetector(w, h)
        self.frame_w, self.frame_h = w, h
        self.initialized = True

    # ==================== 视频控制 ====================

    def open_video(self, path):
        self._switching = True
        time.sleep(0.2)

        # 保存上一个任务的记录
        self._save_task_record()

        with self._lock:
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                self._switching = False
                return False
            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._init_modules(w, h)
            self._reset_stats()
            self._total_frames = 0
            self._current_video = os.path.basename(path)
            self._task_id = str(uuid.uuid4())[:8]
            self._task_start_time = time.time()
        self._switching = False
        return True

    def _reset_stats(self):
        # 先保存已离开车辆的记录
        self._flush_vehicle_stats()
        self._total_vehicles = 0
        self._speed_sum = 0.0
        self._speed_count = 0
        self._danger_vehicles = 0
        self._distance_sum = 0.0
        self._distance_count = 0
        self._smoke_exceed = 0
        self._max_speed = 0.0
        self._max_cars = 0
        self._danger_frames = 0
        self._smoke_frames = 0
        self._detection_count = 0
        self._lane_counts = (0,)
        self._lane_names = []
        self._lane_history.clear()
        self._vehicle_lane_history.clear()
        self._flow_history.clear()
        self._last_violations.clear()
        self._last_distances.clear()
        self._last_leading.clear()
        self._last_vehicles = []
        self._frame_num = 0
        self._traffic_cache_key = None
        self._traffic_cache_img = None
        self._vehicle_stats.clear()
        self._saved_vehicles.clear()
        self._danger_vehicle_ids.clear()
        self._last_lane_stat_time = 0

    def get_frame(self):
        with self._lock:
            return self._frame_jpeg

    def set_mode(self, mode):
        if mode in ('speed', 'distance', 'smoke'):
            self.mode = mode

    # ==================== 采集线程 ====================

    def _capture_loop(self):
        frame_interval = 1.0 / 25.0
        next_frame_time = time.monotonic()

        while self._running:
            if self._switching:
                time.sleep(0.1)
                continue
            cap = self._cap
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue

            now = time.monotonic()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += frame_interval
            if time.monotonic() - next_frame_time > frame_interval * 3:
                next_frame_time = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                with self._lock:
                    if self._cap is cap:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self._total_frames += 1

            if self._detect_thread is None or not self._detect_thread.is_alive():
                self._detect_frame = frame.copy()
                self._detect_thread = threading.Thread(target=self._detect_worker, daemon=True)
                self._detect_thread.start()

            result = self._render_frame(frame, self._last_vehicles)
            _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._frame_jpeg = buf.tobytes()

    # ==================== 检测线程 ====================

    def _detect_worker(self):
        frame = self._detect_frame
        if frame is None:
            return

        vehicles = self.detector.detect(frame)
        self._last_vehicles = vehicles

        if not self.initialized:
            return

        fps = self._video_fps if self._video_fps > 0 else 25.0
        fn = self._total_frames
        speeds = self.speed_calc.update(vehicles, fn, fps)
        self._detection_count += 1

        # 始终计算车距（不仅限于 distance 模式）
        distances = {}
        if vehicles:
            for v in vehicles:
                x1, y1, x2, y2 = v['bbox']
                raw = self.dist_est.estimate(y2 - y1, y2, v['class_name'])
                distances[v['id']] = self.dist_est.estimate_smooth(v['id'], raw)
            active_ids = {v['id'] for v in vehicles}
            self.dist_est.cleanup(active_ids)
            self._last_distances = distances

            # 车道映射
            lane_map = {}
            if self.lane_det:
                lane_map = self.lane_det.assign_lane(vehicles)
            self._last_leading = self.dist_est.find_leading(vehicles, distances, lane_map)

        # 更新统计
        current_count = len(vehicles)
        if current_count > self._max_cars:
            self._max_cars = current_count

        has_smoke_this_frame = False
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        violations = []
        active_ids = {v['id'] for v in vehicles}

        for v in vehicles:
            vid = v['id']
            spd = speeds.get(vid, 0.0)
            dist = distances.get(vid, 999.0)

            # 初始化车辆统计
            if vid not in self._vehicle_stats:
                self._vehicle_stats[vid] = {
                    'enter_time': now_str,
                    'speeds': [],
                    'max_speed': 0.0,
                    'distances': [],
                    'has_smoke': False,
                    'smoke_grade': 0,
                    'lane': 0,
                    'frame_count': 0,
                    'car_type': v['class_name'],
                }
                self._total_vehicles += 1
            vs = self._vehicle_stats[vid]
            vs['frame_count'] += 1
            if spd > 0:
                vs['speeds'].append(spd)
                if spd > vs['max_speed']:
                    vs['max_speed'] = spd
            if 0 < dist < 999:
                vs['distances'].append(dist)

            # 更新最大速度
            if spd > self._max_speed:
                self._max_speed = spd

            # 更新速度统计
            if spd > 0:
                self._speed_sum += spd
                self._speed_count += 1

            # 更新距离统计
            if 0 < dist < 999:
                self._distance_sum += dist
                self._distance_count += 1

            # 超速检测
            if spd > CONFIG['speed_limit']:
                violations.append((vid, '超速', f'{spd:.0f}km/h', vs.get('lane', 0)))

            # 车距过近检测
            if 0 < dist < CONFIG['dist_danger_m']:
                violations.append((vid, '车距过近', f'{dist:.1f}m', vs.get('lane', 0)))
                if vid not in self._danger_vehicle_ids:
                    self._danger_vehicle_ids.add(vid)
                    self._danger_vehicles += 1

            # 黑烟检测（始终运行）
            if self.smoke_det:
                sr = self.smoke_det.detect(frame, v)
                if sr['has_smoke']:
                    if not vs['has_smoke']:
                        self._smoke_exceed += 1
                    vs['has_smoke'] = True
                    vs['smoke_grade'] = max(vs['smoke_grade'], sr['grade'])
                    has_smoke_this_frame = True

        if has_smoke_this_frame:
            self._smoke_frames += 1
        if violations:
            self._danger_frames += 1

        # 更新违章列表（去重，保留最近10条）
        if violations:
            for vio in violations:
                entry = (vio[0], vio[1], vio[2])
                if entry not in self._last_violations:
                    self._last_violations.append(entry)
                    # 写入 traffic_db
                    if self.traffic_db:
                        self.traffic_db.add_violation(vio[0], vio[1], now_str, vio[3])
            self._last_violations = self._last_violations[-10:]

        # 更新车道信息
        lane_map = {}
        if self.lane_det:
            lane_map = self.lane_det.assign_lane(vehicles)
        for v in vehicles:
            vid = v['id']
            if vid in self._vehicle_stats:
                self._vehicle_stats[vid]['lane'] = lane_map.get(vid, 0)

        # 清理离开画面的车辆，写入数据库
        departed = [vid for vid in self._vehicle_stats if vid not in active_ids]
        for vid in departed:
            self._save_vehicle_record(vid)
            del self._vehicle_stats[vid]

        # 定期写入车道统计（每30秒）
        if time.time() - self._last_lane_stat_time >= 30:
            self._write_lane_stats()
            self._last_lane_stat_time = time.time()

    # ==================== 数据库写入 ====================

    def _save_vehicle_record(self, vid):
        """保存单辆车的检测记录到两个数据库"""
        if vid in self._saved_vehicles:
            return
        vs = self._vehicle_stats.get(vid)
        if not vs or vs['frame_count'] < 3:
            return

        self._saved_vehicles.add(vid)
        avg_speed = round(sum(vs['speeds']) / len(vs['speeds']), 1) if vs['speeds'] else 0.0
        max_speed = round(vs['max_speed'], 1)
        min_dist = round(min(vs['distances']), 1) if vs['distances'] else 0.0
        is_over = 1 if max_speed > CONFIG['speed_limit'] else 0

        # 写入 vehicle.db
        if self.db and self._task_id:
            self.db.add_vehicle(
                task_id=self._task_id,
                track_id=vid,
                car_type=vs['car_type'],
                avg_speed=avg_speed,
                max_speed=max_speed,
                lane_num=vs['lane'],
                min_distance=min_dist,
                has_smoke=vs['has_smoke'],
                smoke_grade=vs['smoke_grade'],
                is_overspeed=is_over,
                frame_count=vs['frame_count'],
            )

        # 写入 traffic.db
        if self.traffic_db:
            self.traffic_db.add_vehicle(
                track_id=vid,
                car_type=vs['car_type'],
                avg_speed=avg_speed,
                enter_time=vs['enter_time'],
                leave_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                lane_id=vs['lane'],
            )

    def _flush_vehicle_stats(self):
        """保存所有仍在跟踪的车辆记录"""
        for vid in list(self._vehicle_stats.keys()):
            self._save_vehicle_record(vid)

    def _write_lane_stats(self):
        """写入车道流量统计到 traffic.db"""
        if not self.traffic_db:
            return
        stat_min = time.strftime("%Y-%m-%d %H:%M")
        counts = list(self._lane_counts)
        for i, cnt in enumerate(counts):
            if cnt > 0:
                # 估算该车道的平均速度
                lane_speeds = []
                for vs in self._vehicle_stats.values():
                    if vs.get('lane') == i and vs['speeds']:
                        lane_speeds.append(sum(vs['speeds']) / len(vs['speeds']))
                avg_spd = round(sum(lane_speeds) / len(lane_speeds), 1) if lane_speeds else 0.0
                # 简单按车型估算轿车/公交数量
                car_num = cnt
                bus_num = 0
                for vs in self._vehicle_stats.values():
                    if vs.get('lane') == i and vs['car_type'] == 'bus':
                        bus_num += 1
                        car_num = max(0, car_num - 1)
                self.traffic_db.add_lane_stat(stat_min, i, car_num, bus_num, avg_spd)

    def _save_task_record(self):
        """保存任务汇总记录"""
        if not self.db or not self._task_id or not self._task_start_time:
            return
        self._flush_vehicle_stats()
        elapsed = time.time() - self._task_start_time if self._task_start_time else 0
        avg_speed = round(self._speed_sum / self._speed_count, 1) if self._speed_count > 0 else 0.0
        detection_rate = round(self._detection_count / max(1, self._total_frames) * 100, 1)
        self.db.add_task(
            task_id=self._task_id,
            video_name=self._current_video,
            summary={
                'total_frames': self._total_frames,
                'elapsed_time': round(elapsed, 1),
                'avg_speed': avg_speed,
                'max_speed': round(self._max_speed, 1),
                'max_cars': self._max_cars,
                'danger_frames': self._danger_frames,
                'smoke_frames': self._smoke_frames,
                'detection_rate': detection_rate,
            },
        )

    # ==================== 渲染 ====================

    def _render_frame(self, frame, vehicles):
        self._frame_num += 1
        mode = self.mode

        self._update_lane_stats(vehicles, frame)

        if mode == 'speed':
            return self._render_speed(frame, vehicles)
        elif mode == 'distance':
            return self._render_distance(frame, vehicles)
        else:
            return self._render_smoke(frame, vehicles)

    def _render_speed(self, frame, vehicles):
        h, w = frame.shape[:2]

        for v in vehicles:
            vid = v['id']
            x1, y1, x2, y2 = v['bbox']
            speed = self.speed_calc.get_speed(vid)
            is_over = self.speed_calc.is_overspeed(speed)

            box_color = (0, 0, 255) if is_over else (0, 255, 0)
            thickness = 3 if is_over else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            label = f"{v['class_name']} #{vid} {speed:.1f}km/h"
            text_color = (0, 0, 255) if is_over else (0, 255, 0)
            bg_color = (0, 0, 160) if is_over else (0, 80, 0)
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            lx = max(2, min(x1, w - tw - 6))
            ly = y1 - th - 10
            if ly < 2:
                ly = y2 + 10
            cv2.rectangle(frame, (lx - 2, ly - 2), (lx + tw + 4, ly + th + 4), bg_color, -1)
            cv2.putText(frame, label, (lx + 1, ly + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        self._draw_traffic_level(frame)
        return frame

    def _render_distance(self, frame, vehicles):
        h, w = frame.shape[:2]
        distances = self._last_distances

        for v in vehicles:
            vid = v['id']
            x1, y1, x2, y2 = v['bbox']
            dist = distances.get(vid, -1)
            is_danger = 0 < dist < CONFIG['dist_danger_m']

            box_color = (0, 0, 255) if is_danger else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3 if is_danger else 2)

            if 0 < dist < 999:
                status = "DANGER" if is_danger else "OK"
                label = f"{v['class_name']} #{vid} {dist:.1f}m {status}"
                text_color = (0, 0, 255) if is_danger else (0, 255, 0)
                bg_color = (0, 0, 160) if is_danger else (0, 80, 0)
            else:
                label = f"{v['class_name']} #{vid} --"
                text_color = (0, 255, 0)
                bg_color = (0, 80, 0)

            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            lx = max(2, min(x1, w - tw - 6))
            ly = y1 - th - 10
            if ly < 2:
                ly = y2 + 8
            cv2.rectangle(frame, (lx - 2, ly - 2), (lx + tw + 4, ly + th + 4), bg_color, -1)
            cv2.putText(frame, label, (lx + 1, ly + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        self._draw_traffic_level(frame)
        return frame

    def _render_smoke(self, frame, vehicles):
        h, w = frame.shape[:2]
        for v in vehicles:
            vid = v['id']
            x1, y1, x2, y2 = v['bbox']

            # 从 _vehicle_stats 获取烟雾状态（由检测线程更新）
            vs = self._vehicle_stats.get(vid, {})
            has_smoke = vs.get('has_smoke', False)
            grade = vs.get('smoke_grade', 0)

            if has_smoke:
                # 红色框标记黑烟车辆
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"Black Smoke Lv{grade}"
                font_scale = 0.6
                thickness = 2
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                # 文字固定在框上方，留足间距不被截断
                lx = max(2, min(x1, w - tw - 8))
                ly = y1 - th - 14
                # 如果上方空间不足，放到框下方
                if ly < 4:
                    ly = y2 + th + 8
                # 绘制红色背景条
                cv2.rectangle(frame, (lx - 3, ly - th - 4), (lx + tw + 6, ly + 4), (0, 0, 200), -1)
                cv2.putText(frame, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            else:
                # 绿色框标记正常车辆
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        self._draw_traffic_level(frame)
        return frame

    # ==================== 交通状态 ====================

    def _draw_traffic_level(self, frame):
        count = len(self.detector.last_vehicles)
        avg = self._speed_sum / self._speed_count if self._speed_count > 0 else 60
        if count >= 10 or avg < 20:
            level, color_rgb = "拥堵", (255, 0, 0)
        elif count >= 5 or avg < 40:
            level, color_rgb = "缓行", (255, 200, 0)
        else:
            level, color_rgb = "畅通", (0, 255, 0)

        cache_key = (level, frame.shape[0], frame.shape[1])
        if self._traffic_cache_key != cache_key:
            text = f"TRAFFIC: {level}"
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = _get_font(26)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([8, 6, 12 + tw + 4, 10 + th + 4], fill=(0, 0, 0))
            draw.text((10, 8), text, font=font, fill=color_rgb)
            self._traffic_cache_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            self._traffic_cache_key = cache_key

        h, w = frame.shape[:2]
        roi_h = min(40, h)
        roi_w = min(300, w)
        frame[0:roi_h, 0:roi_w] = self._traffic_cache_img[0:roi_h, 0:roi_w]

    # ==================== 车道统计 ====================

    def _update_lane_stats(self, vehicles, frame):
        now = time.time()
        count = len(vehicles)
        self._flow_history.append((now, count))
        self._flow_history = [(t, c) for t, c in self._flow_history if now - t <= 30]

        boundaries = self._detect_lanes(frame)
        n_lanes = len(boundaries) + 1
        raw_counts = [0] * n_lanes

        for v in vehicles:
            cx = v['center'][0]
            vid = v['id']
            idx = 0
            for bx in boundaries:
                if cx >= bx:
                    idx += 1
                else:
                    break
            idx = min(idx, n_lanes - 1)

            if vid not in self._vehicle_lane_history:
                self._vehicle_lane_history[vid] = []
            self._vehicle_lane_history[vid].append(idx)
            if len(self._vehicle_lane_history[vid]) > 6:
                self._vehicle_lane_history[vid] = self._vehicle_lane_history[vid][-6:]

            history = self._vehicle_lane_history[vid]
            if len(history) >= 3 and all(l == idx for l in history[-3:]):
                raw_counts[idx] += 1

        active_ids = {v['id'] for v in vehicles}
        stale = [t for t in self._vehicle_lane_history if t not in active_ids]
        for t in stale:
            del self._vehicle_lane_history[t]

        self._lane_counts = tuple(raw_counts)
        self._lane_names = [f"车道{i + 1}" for i in range(n_lanes)]

        if not self._lane_history or now - self._lane_history[-1][0] >= 1.0:
            self._lane_history.append((now, list(self._lane_counts)))
        self._lane_history = [(t, c) for t, c in self._lane_history if now - t <= 30]

    def _detect_lanes(self, frame):
        fh, fw = frame.shape[:2]
        if self._lane_boundaries and self._lane_frame_count > 0 and self._lane_frame_count % 90 != 0:
            self._lane_frame_count += 1
            return self._lane_boundaries
        self._lane_frame_count = 1

        roi_top = int(fh * 0.30)
        roi_bot = int(fh * 0.90)
        roi = frame[roi_top:roi_bot, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([10, 60, 80]), np.array([40, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 40, 255]))
        kernel = np.ones((3, 15), np.uint8)
        yellow_dilated = cv2.dilate(yellow_mask, kernel, iterations=1)
        combined = cv2.bitwise_or(yellow_dilated, white_mask)
        edges = cv2.Canny(cv2.GaussianBlur(combined, (5, 5), 0), 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=fw // 12, maxLineGap=fw // 6)

        y_lines, w_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1_l, x2, y2_l = line[0]
                dx, dy = x2 - x1, y2_l - y1_l
                if abs(dx) < abs(dy) * 0.4:
                    continue
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                if angle < 10 or angle > 170:
                    continue
                mx = (x1 + x2) / 2
                y_min, y_max = min(y1_l, y2_l), max(y1_l, y2_l)
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min = max(0, y_min)
                y_max = min(yellow_dilated.shape[0] - 1, y_max)
                x_min = max(0, x_min)
                x_max = min(yellow_dilated.shape[1] - 1, x_max)
                if y_max > y_min and x_max > x_min:
                    yellow_ratio = np.mean(yellow_dilated[y_min:y_max + 1, x_min:x_max + 1] > 0)
                    if yellow_ratio > 0.15:
                        y_lines.append(mx)
                        continue
                w_lines.append(mx)

        center_x = None
        if y_lines:
            y_sorted = sorted(y_lines)
            clusters = [[y_sorted[0]]]
            for i in range(1, len(y_sorted)):
                if y_sorted[i] - y_sorted[i - 1] < fw * 0.08:
                    clusters[-1].append(y_sorted[i])
                else:
                    clusters.append([y_sorted[i]])
            cluster_means = [np.mean(c) for c in clusters]
            center_x = min(cluster_means, key=lambda c: abs(c - fw / 2))

        merge_gap = fw * 0.06
        if center_x is not None:
            w_left = sorted([x for x in w_lines if x < center_x - fw * 0.05])
            w_right = sorted([x for x in w_lines if x > center_x + fw * 0.05])
        else:
            w_left = sorted([x for x in w_lines if x < fw / 2])
            w_right = sorted([x for x in w_lines if x >= fw / 2])

        def merge(lst, gap):
            if not lst:
                return []
            result = [lst[0]]
            for x in lst[1:]:
                if x - result[-1] < gap:
                    result[-1] = (result[-1] + x) / 2
                else:
                    result.append(x)
            return result

        boundaries = []
        if center_x is not None:
            boundaries.append(center_x)
        boundaries += merge(w_left, merge_gap) + merge(w_right, merge_gap)
        boundaries = sorted(boundaries)
        if len(boundaries) < 1:
            boundaries = [fw / 3, fw * 2 / 3]
        self._lane_boundaries = boundaries
        return boundaries

    # ==================== 统计 API ====================

    def get_stats(self):
        now = time.time()
        avg_speed = round(self._speed_sum / self._speed_count, 1) if self._speed_count > 0 else 0.0
        avg_distance = round(self._distance_sum / self._distance_count, 1) if self._distance_count > 0 else 0.0

        count = len(self.detector.last_vehicles)
        if count >= 10 or avg_speed < 20:
            level, color = "拥堵", "#ef4444"
        elif count >= 5 or avg_speed < 40:
            level, color = "缓行", "#f59e0b"
        else:
            level, color = "畅通", "#22c55e"

        buckets = {}
        for t, c in self._flow_history:
            s = int(now - t)
            buckets[s] = buckets.get(s, 0) + c
        flow = [{"t": i, "c": buckets.get(i, 0)} for i in range(30, -1, -1)]

        counts = list(self._lane_counts)
        names = list(self._lane_names) if self._lane_names else ['车道1']
        total_lane = sum(counts) or 1
        pcts = [round(c / total_lane * 100) for c in counts]

        lane_hist = []
        for item in self._lane_history:
            t, cts = item[0], item[1]
            entry = {"t": round(now - t), "total": sum(cts)}
            for i, c in enumerate(cts):
                entry[f"l{i}"] = c
            lane_hist.append(entry)
        lane_hist.reverse()

        mode_names = {"speed": "车速检测", "distance": "车距预警", "smoke": "黑烟检测"}

        # 计算平均林格曼等级
        grades = [vs['smoke_grade'] for vs in self._vehicle_stats.values() if vs.get('has_smoke')]
        avg_ringelmann = round(sum(grades) / len(grades), 1) if grades else 0.0

        return {
            "total_vehicles": self._total_vehicles,
            "avg_speed": avg_speed,
            "danger_vehicles": self._danger_vehicles,
            "avg_distance": avg_distance,
            "smoke_exceed": self._smoke_exceed,
            "avg_ringelmann": avg_ringelmann,
            "current_mode": mode_names.get(self.mode, self.mode),
            "mode_key": self.mode,
            "total_frames": self._total_frames,
            "current_video": self._current_video,
            "traffic_level": level,
            "traffic_color": color,
            "violations": self._last_violations[-10:],
            "flow_history": flow,
            "lane_names": names,
            "lane_counts": counts,
            "lane_pcts": pcts,
            "lane_total": sum(counts),
            "lane_history": lane_hist,
        }
