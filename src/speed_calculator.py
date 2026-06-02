# -*- coding: utf-8 -*-
"""
模块: 车速计算器
功能: 基于双参考线法测量车辆速度
原理: 在视频画面中设置两条水平参考线，跟踪车辆经过两条线的时间差，
      利用公式 速度 = 距离 / 时间 计算车速
"""

from .config import CONFIG


class SpeedCalculator:
    """
    双参考线测速器

    工作流程:
    1. 在帧画面中画出两条水平蓝色参考线
    2. 跟踪每个车辆中心点经过两条线的帧号
    3. 车辆经过两条线后，计算速度并做平滑处理
    4. 超过限速的车辆标记为红色
    """

    def __init__(self, frame_h):
        """
        参数:
            frame_h: 视频帧高度 (像素)，用于计算参考线 Y 坐标
        """
        self.frame_h = frame_h

        # 计算两条参考线的 Y 坐标 (像素)
        self.line1_y = int(frame_h * CONFIG['speed_line1_y_ratio'])
        self.line2_y = int(frame_h * CONFIG['speed_line2_y_ratio'])

        # 车辆跟踪数据: {vehicle_id: {'crossings': [...], 'speed': float, 'last_frame': int}}
        self.tracker = {}

        # 通过冷却帧数 (防止同一位置重复触发)
        self.cooldown = CONFIG['speed_cooldown']

        # 确认有效车速列表 (车辆成功穿过双线后记录)
        self.confirmed_speeds = []

    def update(self, vehicles, frame_num, fps):
        """
        更新车辆位置并计算速度

        参数:
            vehicles: 车辆检测列表 (需包含 'id' 和 'center')
            frame_num: 当前帧号
            fps: 视频帧率

        返回:
            车辆速度字典 {vehicle_id: speed_kmh}
        """
        speeds = {}

        for v in vehicles:
            vid = v['id']
            cy = v['center'][1]  # 车辆中心 Y 坐标

            # 初始化跟踪数据
            if vid not in self.tracker:
                self.tracker[vid] = {
                    'crossings': [],     # 记录穿越事件
                    'speed': 0,          # 当前平滑速度
                    'last_calc_frame': 0  # 上次计算速度的帧号
                }

            t = self.tracker[vid]
            prev_cy = t.get('prev_cy', cy)
            t['prev_cy'] = cy

            # 检测是否穿越参考线 (检查前一帧和当前帧是否跨越了线)
            for line_idx, line_y in enumerate([self.line1_y, self.line2_y]):
                # 跨越检测: 前一帧在线的一侧，当前帧在另一侧 (或在线上)
                crossed = (prev_cy - line_y) * (cy - line_y) <= 0
                # 也检测在附近的容差区域
                near = abs(cy - line_y) < 30

                if crossed or near:
                    # 检查冷却
                    recent = [c for c in t['crossings']
                              if c['line'] == line_idx
                              and frame_num - c['frame'] < self.cooldown]
                    if not recent:
                        t['crossings'].append({
                            'line': line_idx,
                            'frame': frame_num,
                            'y': cy
                        })

            # 检查是否可以计算速度 (两条线都穿越过)
            line0_crossings = [c for c in t['crossings'] if c['line'] == 0]
            line1_crossings = [c for c in t['crossings'] if c['line'] == 1]

            if line0_crossings and line1_crossings:
                # 取最近的两次穿越
                c0 = line0_crossings[-1]
                c1 = line1_crossings[-1]

                # 计算时间差 (秒)
                dt_frames = abs(c1['frame'] - c0['frame'])
                if dt_frames > 0 and fps > 0:
                    dt_sec = dt_frames / fps

                    # 速度(km/h) = 距离(m) / 时间(s) × 3.6
                    raw_speed = (CONFIG['line_real_dist'] / dt_sec) * 3.6

                    # 过滤异常值
                    if CONFIG['speed_min'] <= raw_speed <= CONFIG['speed_max']:
                        # 指数移动平均平滑
                        alpha = CONFIG['speed_smooth_alpha']
                        t['speed'] = alpha * t['speed'] + (1 - alpha) * raw_speed
                        # 记录确认有效车速
                        self.confirmed_speeds.append(t['speed'])

                    # 清除已使用的穿越记录，避免重复计算
                    t['crossings'] = []
                    t['last_calc_frame'] = frame_num

            speeds[vid] = t['speed']

        # 清理长时间未出现的车辆
        stale = [vid for vid, t in self.tracker.items()
                 if frame_num - t.get('last_calc_frame', frame_num) > fps * 5]
        for vid in stale:
            del self.tracker[vid]

        return speeds

    def get_line_positions(self):
        """返回两条参考线的 Y 坐标 (用于绘制)，确保为 Python int"""
        return int(self.line1_y), int(self.line2_y)

    def is_overspeed(self, speed):
        """判断是否超速"""
        return speed > CONFIG['speed_limit']

    def get_confirmed_speeds(self):
        """返回所有确认有效的车速列表 (车辆成功穿过双测速线)"""
        return list(self.confirmed_speeds)
