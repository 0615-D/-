# -*- coding: utf-8 -*-
"""
模块: 视频处理器 (v3)
功能: 一次性预生成 5 个带标注的 MP4 视频文件
      speed.mp4 / distance.mp4 / heat.mp4 / smoke.mp4 / all.mp4
"""

import cv2
import numpy as np
import json
import time
from .config import CONFIG
from .vehicle_detector import VehicleDetector
from .speed_calculator import SpeedCalculator
from .distance_estimator import DistanceEstimator
from .lane_detector import LaneDetector
from .smoke_detector import SmokeDetector
from .heatmap import HeatmapGenerator


class VideoProcessor:
    """视频处理器 - 一次性生成 5 个图层视频"""

    def __init__(self):
        self.detector = VehicleDetector()
        self.speed_calc = None
        self.dist_est = None
        self.lane_det = None
        self.smoke_det = None
        self.heatmap = None
        self.initialized = False

    def _init_modules(self, w, h):
        self.speed_calc = SpeedCalculator(h)
        self.dist_est = DistanceEstimator(h)
        self.lane_det = LaneDetector(w, h)
        self.smoke_det = SmokeDetector(w, h)
        self.heatmap = HeatmapGenerator(w, h)
        self.frame_w, self.frame_h = w, h
        self.initialized = True

    # ==================== 绘制函数 ====================

    def _draw_speed_layer(self, frame, vehicles, speeds):
        """绘制车速图层: 参考线 + 车辆框 + ID + 速度"""
        h, w = frame.shape[:2]
        line1_y, line2_y = self.speed_calc.get_line_positions()

        # 两条蓝色测速参考线
        cv2.line(frame, (0, line1_y), (w, line1_y), (255, 100, 0), 2)
        cv2.line(frame, (0, line2_y), (w, line2_y), (255, 100, 0), 2)

        # 距离标注
        mid_y = (line1_y + line2_y) // 2
        cv2.putText(frame, f"{CONFIG['line_real_dist']}m", (w // 2 - 20, mid_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        for v in vehicles:
            vid = v['id']
            x1, y1, x2, y2 = v['bbox']
            speed = speeds.get(vid, 0)
            is_over = self.speed_calc.is_overspeed(speed) if speed > 0 else False

            # 车辆框: 超速红色，正常绿色
            color = (0, 0, 255) if is_over else (0, 200, 0)
            thickness = 3 if is_over else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # ID
            cv2.putText(frame, f"ID:{vid}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 车速
            if speed > 0:
                speed_color = (0, 0, 255) if is_over else (0, 200, 0)
                cv2.putText(frame, f"{speed:.0f}km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
                if is_over:
                    cv2.putText(frame, "OVERSPEED!", (x2 - 100, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def _draw_distance_layer(self, frame, vehicles, distances, speeds, leading):
        """绘制车距图层: 距离标注 + 危险连线"""
        for v in vehicles:
            vid = v['id']
            x1, y1, x2, y2 = v['bbox']
            cx, cy = v['center']
            dist = distances.get(vid, -1)

            # 距离文字
            if 0 < dist < 999:
                cv2.putText(frame, f"Dist:{dist:.1f}m", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 危险车距连线
            lead_id = leading.get(vid)
            if lead_id is not None:
                lead_dist = distances.get(vid, -1)
                safe = self.dist_est.compute_safe_distance(speeds.get(vid, 0))
                if 0 < lead_dist < safe:
                    # 找到前车
                    for u in vehicles:
                        if u['id'] == lead_id:
                            cv2.line(frame, (cx, cy), u['center'], (0, 0, 255), 2)
                            mx, my = (cx + u['center'][0]) // 2, (cy + u['center'][1]) // 2
                            cv2.putText(frame, "DANGER!", (mx - 30, my),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            break

    def _draw_heat_layer(self, frame):
        """绘制热力图叠加"""
        return self.heatmap.generate_overlay(frame)

    def _draw_smoke_layer(self, frame, vehicles, smoke_results):
        """绘制黑烟检测图层"""
        for v in vehicles:
            vid = v['id']
            x1, y1, x2, y2 = v['bbox']
            smoke = smoke_results.get(vid, {})
            has_smoke = smoke.get('has_smoke', False)
            grade = smoke.get('grade', 0)
            roi = smoke.get('roi')

            # 绘制尾气 ROI 区域
            if roi:
                rx1, ry1, rx2, ry2 = roi
                roi_color = (0, 0, 255) if has_smoke else (0, 165, 255)
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_color, 2)

            # 车辆框
            box_color = (0, 0, 255) if has_smoke else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # 黑烟等级标注
            if has_smoke:
                cv2.putText(frame, f"SMOKE G{grade}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ==================== 主处理 ====================

    def process_video(self, input_path, output_dir):
        """
        处理视频: 一次性生成 5 个图层视频

        参数:
            input_path: 输入视频路径
            output_dir: 输出目录 (static/res/)

        返回:
            (total_frames, elapsed_time, summary_stats, video_paths_dict)
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 60:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.initialized:
            self._init_modules(w, h)

        fourcc = cv2.VideoWriter_fourcc(*CONFIG['output_codec'])

        # 创建 5 个 VideoWriter
        video_names = ['speed', 'distance', 'heat', 'smoke', 'all']
        writers = {}
        for name in video_names:
            path = f"{output_dir}/{name}.mp4"
            writers[name] = cv2.VideoWriter(path, fourcc, fps, (w, h))

        detections = []
        all_stats = []
        start_time = time.time()
        fc = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fc += 1

            # ===== 检测 =====
            vehicles = self.detector.detect(frame)
            speeds = self.speed_calc.update(vehicles, fc, fps)
            self.lane_det.detect(frame)
            lane_map = self.lane_det.assign_lane(vehicles)

            distances = {}
            for v in vehicles:
                x1, y1, x2, y2 = v['bbox']
                distances[v['id']] = self.dist_est.estimate(y2 - y1, y2)

            leading = self.dist_est.find_leading_vehicle(vehicles, distances, lane_map)

            smoke_results = {}
            for v in vehicles:
                res = self.smoke_det.detect(v)
                if res['roi'] is not None:
                    rx1, ry1, rx2, ry2 = res['roi']
                    roi_img = frame[ry1:ry2, rx1:rx2]
                    grade, ratio, has_smoke = self.smoke_det.analyze_roi(roi_img, v['id'])
                    smoke_results[v['id']] = {'grade': grade, 'ratio': ratio, 'has_smoke': has_smoke, 'roi': res['roi']}
                else:
                    smoke_results[v['id']] = {'grade': 0, 'ratio': 0, 'has_smoke': False, 'roi': None}
            self.smoke_det.cleanup({v['id'] for v in vehicles})
            self.heatmap.update(vehicles)

            # ===== 写入 5 个视频 =====

            # 1. speed.mp4
            speed_frame = frame.copy()
            self._draw_speed_layer(speed_frame, vehicles, speeds)
            writers['speed'].write(speed_frame)

            # 2. distance.mp4
            dist_frame = frame.copy()
            self._draw_distance_layer(dist_frame, vehicles, distances, speeds, leading)
            writers['distance'].write(dist_frame)

            # 3. heat.mp4
            heat_frame = self._draw_heat_layer(frame.copy())
            writers['heat'].write(heat_frame)

            # 4. smoke.mp4
            smoke_frame = frame.copy()
            self._draw_smoke_layer(smoke_frame, vehicles, smoke_results)
            writers['smoke'].write(smoke_frame)

            # 5. all.mp4 - 全图层叠加
            all_frame = frame.copy()
            self._draw_speed_layer(all_frame, vehicles, speeds)
            self._draw_distance_layer(all_frame, vehicles, distances, speeds, leading)
            all_frame = self._draw_heat_layer(all_frame)
            self._draw_smoke_layer(all_frame, vehicles, smoke_results)
            writers['all'].write(all_frame)

            # 构建帧检测数据 (JSON)
            frame_data = []
            for v in vehicles:
                vid = v['id']
                x1, y1, x2, y2 = v['bbox']
                cx, cy = v['center']
                speed = speeds.get(vid, 0)
                dist = distances.get(vid, -1)
                smoke = smoke_results.get(vid, {})
                is_over = self.speed_calc.is_overspeed(speed) if speed > 0 else False

                lead_id = leading.get(vid)
                lead_dist = -1
                if lead_id is not None:
                    lead_dist = distances.get(vid, -1)
                    safe = self.dist_est.compute_safe_distance(speed)
                    if lead_dist > safe or lead_dist >= 999:
                        lead_id = None
                        lead_dist = -1

                entry = {
                    'id': vid,
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'cx': int(cx), 'cy': int(cy),
                    'cls': v['class_name'],
                    'conf': round(float(v['conf']), 2),
                    'speed': round(float(speed), 1) if speed > 0 else 0,
                    'overspeed': is_over,
                    'dist': round(float(dist), 1) if 0 < dist < 999 else -1,
                    'lead_id': lead_id if lead_id is not None else -1,
                    'lead_dist': round(float(lead_dist), 1) if lead_dist > 0 else -1,
                    'smoke': smoke.get('has_smoke', False),
                    'smoke_grade': smoke.get('grade', 0),
                    'smoke_roi': [int(x) for x in smoke['roi']] if smoke.get('roi') else [],
                }
                frame_data.append(entry)

            detections.append({'f': fc, 'v': frame_data})

            # 帧统计
            active_speeds = [s for s in speeds.values() if s > 0]
            all_stats.append({
                'car_count': len(vehicles),
                'avg_speed': float(np.mean(active_speeds)) if active_speeds else 0,
                'max_speed': float(max(active_speeds)) if active_speeds else 0,
                'min_dist': min(distances.values()) if distances else 999,
                'danger_count': sum(1 for v in vehicles
                                    if 0 < distances.get(v['id'], 999) <
                                    self.dist_est.compute_safe_distance(speeds.get(v['id'], 0))),
                'smoke_count': sum(1 for s in smoke_results.values() if s.get('has_smoke')),
            })

        # 释放资源
        cap.release()
        for w_obj in writers.values():
            w_obj.release()

        elapsed = time.time() - start_time

        # 保存热力图图片
        heatmap_path = f"{output_dir}/heatmap.png"
        self.heatmap.save_image(heatmap_path)

        # 保存检测数据 JSON
        l1y, l2y = self.speed_calc.get_line_positions()
        json_data = {
            'w': w, 'h': h, 'fps': fps,
            'line1_y': l1y, 'line2_y': l2y,
            'line_dist': CONFIG['line_real_dist'],
            'speed_limit': CONFIG['speed_limit'],
            'frames': detections,
        }
        json_path = f"{output_dir}/data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, separators=(',', ':'))

        summary = self._summarize(all_stats, fc, elapsed, fps)
        summary['heatmap_path'] = heatmap_path

        # 视频路径字典
        video_paths = {name: f"{name}.mp4" for name in video_names}
        video_paths['heatmap'] = 'heatmap.png'
        video_paths['json'] = 'data.json'

        return fc, elapsed, summary, video_paths

    def _summarize(self, all_stats, total_frames, elapsed, fps):
        if not all_stats:
            return {}
        confirmed_speeds = self.speed_calc.get_confirmed_speeds()
        frames_with_cars = [s for s in all_stats if s['car_count'] > 0]
        dists = [s['min_dist'] for s in frames_with_cars if s['min_dist'] < 999]
        return {
            'total_frames': total_frames,
            'elapsed_time': round(elapsed, 1),
            'fps_actual': round(total_frames / max(elapsed, 1), 1),
            'avg_speed': round(float(np.mean(confirmed_speeds)), 1) if confirmed_speeds else None,
            'max_speed': round(float(max(confirmed_speeds)), 1) if confirmed_speeds else None,
            'speed_count': len(confirmed_speeds),
            'avg_min_dist': round(float(np.mean(dists)), 1) if dists else 0,
            'max_cars': max((s['car_count'] for s in all_stats), default=0),
            'danger_frames': sum(1 for s in all_stats if s['danger_count'] > 0),
            'smoke_frames': sum(1 for s in all_stats if s['smoke_count'] > 0),
            'detection_rate': round(len(frames_with_cars) / max(total_frames, 1) * 100, 1),
        }
