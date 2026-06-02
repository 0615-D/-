# -*- coding: utf-8 -*-
"""
模块: 车辆停留热力图
功能: 累积车辆位置，生成密度热力图并叠加在视频画面上
原理: 对每帧检测到的车辆中心位置在低分辨率矩阵中累加，
      高斯模糊平滑后应用 JET 色彩映射，半透明叠加
"""

import cv2
import numpy as np
from .config import CONFIG


class HeatmapGenerator:
    """
    车辆停留热力图生成器

    为保证性能，热力图在低分辨率下计算 (默认 1/4 原始分辨率)，
    上采样后再叠加到原始帧上
    """

    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # 热力图计算分辨率
        self.scale = CONFIG['heatmap_scale']
        self.hm_w = int(frame_w * self.scale)
        self.hm_h = int(frame_h * self.scale)

        # 累积矩阵 (float32 精度足够)
        self.accumulator = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)

        # 帧计数器
        self.frame_count = 0
        self.update_interval = CONFIG['heatmap_update_interval']

        # 缓存上一次生成的叠加层 (避免重复计算)
        self.cached_overlay = None

    def update(self, vehicles):
        """
        更新热力图数据

        参数:
            vehicles: 车辆列表 (需包含 'center' 字段)
        """
        self.frame_count += 1

        for v in vehicles:
            cx, cy = v['center']
            # 缩放到低分辨率坐标
            cx_s = int(cx * self.scale)
            cy_s = int(cy * self.scale)

            # 在中心位置累加一个高斯点
            radius = max(3, int(15 * self.scale))
            y1 = max(0, cy_s - radius)
            y2 = min(self.hm_h, cy_s + radius)
            x1 = max(0, cx_s - radius)
            x2 = min(self.hm_w, cx_s + radius)

            if y2 > y1 and x2 > x1:
                # 创建径向渐变
                yy, xx = np.mgrid[y1:y2, x1:x2]
                dist = np.sqrt((xx - cx_s) ** 2 + (yy - cy_s) ** 2)
                glow = np.clip(1.0 - dist / radius, 0, 1)
                self.accumulator[y1:y2, x1:x2] += glow.astype(np.float32)

    def generate_overlay(self, frame):
        """
        生成热力图叠加层

        参数:
            frame: 原始视频帧 (BGR)

        返回:
            叠加了热力图的帧
        """
        # 按间隔更新 (不是每帧都重新生成叠加层)
        if self.frame_count % self.update_interval == 0:
            self.cached_overlay = self._create_overlay()

        if self.cached_overlay is None:
            return frame

        # 叠加到原始帧
        alpha = CONFIG['heatmap_alpha']
        result = cv2.addWeighted(frame, 1.0, self.cached_overlay, alpha, 0)
        return result

    def _create_overlay(self):
        """生成热力图叠加图像 (BGR)"""
        if np.max(self.accumulator) == 0:
            return None

        # 归一化到 0-255
        norm = self.accumulator / max(np.max(self.accumulator), 1) * 255
        norm = norm.astype(np.uint8)

        # 高斯模糊平滑
        k = CONFIG['heatmap_blur_kernel']
        blurred = cv2.GaussianBlur(norm, (k, k), 0)

        # 上采样到原始分辨率
        blurred_full = cv2.resize(blurred, (self.frame_w, self.frame_h),
                                  interpolation=cv2.INTER_LINEAR)

        # 应用 JET 色彩映射 (蓝→绿→黄→红)
        heatmap_color = cv2.applyColorMap(blurred_full, cv2.COLORMAP_JET)

        # 创建遮罩: 只在有数据的区域显示
        mask = blurred_full > 5
        overlay = np.zeros_like(heatmap_color)
        overlay[mask] = heatmap_color[mask]

        return overlay

    def save_image(self, output_path):
        """保存独立的热力图图片"""
        overlay = self._create_overlay()
        if overlay is None:
            return False
        # 加深色背景
        bg = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        bg[:] = (15, 18, 25)
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0
        bg[mask] = overlay[mask]
        cv2.imwrite(output_path, bg)
        return True
