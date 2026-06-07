# -*- coding: utf-8 -*-
"""
模块: 车速计算器
功能: 基于像素位移的逐帧测速，带透视校正和平滑滤波
"""

from .config import CONFIG


class SpeedCalculator:
    """
    逐帧测速器

    工作流程:
    1. 每帧记录车辆中心坐标，保留最近N帧轨迹
    2. 连续N帧位移<1像素 → 判定静止，显示0
    3. 用最后5帧位移计算速度，透视校正后EMA平滑
    """

    def __init__(self, frame_h):
        self.frame_h = frame_h
        # {vid: [(frame_num, cx, cy), ...]}
        self.tracks = {}
        # {vid: smoothed_speed_kmh}
        self.smoothed = {}

        self.track_len = CONFIG['speed_track_len']
        self.window = CONFIG['speed_smooth_window']
        self.static_frames = CONFIG['speed_static_frames']
        self.static_px = CONFIG['speed_static_px']
        self.alpha = CONFIG['speed_ema_alpha']
        self.clamp_max = CONFIG['speed_clamp_max']

        self.ppm_bottom = CONFIG['ppm_bottom']
        self.ppm_top = CONFIG['ppm_top']

    def _ppm_at_y(self, y):
        """根据Y坐标返回透视校正后的每米像素数"""
        ratio = max(0.0, min(1.0, y / self.frame_h))
        return self.ppm_top + (self.ppm_bottom - self.ppm_top) * ratio

    def update(self, vehicles, frame_num, fps):
        """
        更新车辆位置，计算速度

        返回: {vid: speed_kmh}
        """
        speeds = {}
        active_ids = set()

        for v in vehicles:
            vid = v['id']
            active_ids.add(vid)
            cx, cy = v['center']

            # 记录轨迹
            if vid not in self.tracks:
                self.tracks[vid] = []
            self.tracks[vid].append((frame_num, cx, cy))
            if len(self.tracks[vid]) > self.track_len:
                self.tracks[vid] = self.tracks[vid][-self.track_len:]

            history = self.tracks[vid]
            n = len(history)

            if n < 2:
                speeds[vid] = 0.0
                continue

            # 静止判定
            if n >= self.static_frames:
                recent = history[-self.static_frames:]
                all_static = True
                for i in range(1, len(recent)):
                    dx = recent[i][1] - recent[i - 1][1]
                    dy = recent[i][2] - recent[i - 1][2]
                    if (dx * dx + dy * dy) ** 0.5 >= self.static_px:
                        all_static = False
                        break
                if all_static:
                    self.smoothed[vid] = 0.0
                    speeds[vid] = 0.0
                    continue

            # 需要足够帧数
            if n < self.window:
                speeds[vid] = self.smoothed.get(vid, 0.0)
                continue

            window = history[-self.window:]
            f0, x0, y0 = window[0]
            f1, x1, y1 = window[-1]
            frame_gap = f1 - f0
            if frame_gap <= 0:
                speeds[vid] = self.smoothed.get(vid, 0.0)
                continue

            pixel_dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            # 透视校正：用轨迹起点的Y坐标
            ppm = self._ppm_at_y(y0)
            speed_kmh = (pixel_dist / frame_gap) * fps / ppm * 3.6

            # 异常过滤
            if speed_kmh < 0.5:
                speed_kmh = 0.0
            elif speed_kmh > self.clamp_max:
                speed_kmh = self.clamp_max

            # EMA平滑
            prev = self.smoothed.get(vid, speed_kmh)
            smoothed = self.alpha * speed_kmh + (1 - self.alpha) * prev
            self.smoothed[vid] = smoothed
            speeds[vid] = round(smoothed, 1)

        # 清理消失车辆
        stale = [vid for vid in self.tracks if vid not in active_ids]
        for vid in stale:
            del self.tracks[vid]
            self.smoothed.pop(vid, None)

        return speeds

    def get_speed(self, vid):
        return self.smoothed.get(vid, 0.0)

    def is_overspeed(self, speed):
        return speed > CONFIG['speed_limit']
