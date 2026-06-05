"""
高级车道线检测模块
包含: 相机标定、透视变换、多阈值融合、曲率计算、车辆偏移、帧平滑、完整性校验
"""

import cv2
import numpy as np
import os


class LaneDetector:
    """
    高级车道线检测器
    完整 7 阶段 Pipeline:
    1. 相机标定与去畸变
    2. 透视变换(鸟瞰图)
    3. 多色彩空间 + 梯度阈值融合
    4. 直方图峰值检测
    5. 滑动窗口多项式拟合
    6. 曲率半径 & 偏移量计算
    7. 可视化与逆透视变换
    """

    def __init__(self):
        # 相机标定参数
        self.mtx = None
        self.dist = None
        self.calibrated = False

        # 透视变换矩阵
        self.M = None
        self.Minv = None

        # 车道线多项式系数(上一帧)
        self.left_fit = None
        self.right_fit = None

        # 帧平滑队列
        self.recent_left_fits = []
        self.recent_right_fits = []
        self.smooth_window = 10

        # 车道线历史记录（用于稳定性判断）
        self.detected = False

        # 像素到米的转换系数
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700

    # ----------------------------------------------------------
    # Stage 1: 相机标定
    # ----------------------------------------------------------
    def calibrate_camera(self, calibration_dir, pattern_size=(9, 6)):
        """
        使用棋盘格图片进行相机标定
        calibration_dir: 包含棋盘格图片的目录
        """
        objpoints = []
        imgpoints = []

        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        if not os.path.isdir(calibration_dir):
            print(f"标定目录不存在: {calibration_dir}，跳过标定")
            return False

        images = [os.path.join(calibration_dir, f)
                  for f in os.listdir(calibration_dir)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not images:
            print("未找到标定图片，跳过标定")
            return False

        img_size = None
        found_count = 0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = (gray.shape[1], gray.shape[0])
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                found_count += 1

        if found_count == 0:
            print("未能在任何图片中检测到棋盘格角点")
            return False

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None)
        self.calibrated = True
        print(f"相机标定完成，使用 {found_count}/{len(images)} 张图片")
        return True

    def undistort(self, image):
        """去畸变"""
        if self.calibrated:
            return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return image

    # ----------------------------------------------------------
    # Stage 2: 透视变换
    # ----------------------------------------------------------
    def setup_perspective(self, image, src_points=None, dst_points=None):
        """
        设置透视变换参数
        """
        h, w = image.shape[:2]
        if src_points is None:
            src_points = np.float32([
                [int(w * 0.45), int(h * 0.63)],
                [int(w * 0.55), int(h * 0.63)],
                [int(w * 0.90), int(h * 0.95)],
                [int(w * 0.10), int(h * 0.95)]
            ])
        if dst_points is None:
            dst_points = np.float32([
                [int(w * 0.20), 0],
                [int(w * 0.80), 0],
                [int(w * 0.80), h],
                [int(w * 0.20), h]
            ])
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        return self.M

    def warp_perspective(self, image):
        """应用透视变换得到鸟瞰图"""
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def unwarp_perspective(self, image):
        """逆透视变换"""
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, self.Minv, (w, h), flags=cv2.INTER_LINEAR)

    # ----------------------------------------------------------
    # Stage 3: 多阈值融合
    # ----------------------------------------------------------
    @staticmethod
    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Sobel绝对值梯度阈值"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled = np.uint8(255 * abs_sobel / (np.max(abs_sobel) + 1e-7))
        binary = np.zeros_like(scaled)
        binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
        return binary

    @staticmethod
    def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
        """Sobel幅值阈值"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-7))
        binary = np.zeros_like(scaled)
        binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
        return binary

    @staticmethod
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        """Sobel方向阈值"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary = np.zeros_like(absgraddir)
        binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary

    @staticmethod
    def hls_s_channel(img, thresh=(170, 255)):
        """HLS 色彩空间 S 通道阈值 (对阴影鲁棒)"""
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
        s = hls[:, :, 2]
        binary = np.zeros_like(s)
        binary[(s >= thresh[0]) & (s <= thresh[1])] = 1
        return binary

    @staticmethod
    def lab_b_channel(img, thresh=(155, 200)):
        """LAB 色彩空间 B 通道阈值 (黄色线专用)"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        b = lab[:, :, 2]
        binary = np.zeros_like(b)
        binary[(b >= thresh[0]) & (b <= thresh[1])] = 1
        return binary

    @staticmethod
    def hsv_white_mask(img):
        """HSV 色彩空间白色掩码"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return (mask / 255).astype(np.uint8)

    def combined_threshold(self, image):
        """
        多阈值融合: Sobel梯度 + HLS S通道 + LAB B通道 + HSV白色
        """
        gradx = self.abs_sobel_thresh(image, 'x', 3, (20, 100))
        grady = self.abs_sobel_thresh(image, 'y', 3, (20, 100))
        mag_binary = self.mag_thresh(image, 3, (30, 100))
        dir_binary = self.dir_threshold(image, 15, (0.7, 1.3))
        s_binary = self.hls_s_channel(image, (170, 255))
        b_binary = self.lab_b_channel(image, (155, 200))
        white_binary = self.hsv_white_mask(image)

        combined = np.zeros_like(gradx)
        combined[
            ((gradx == 1) & (grady == 1)) |
            ((mag_binary == 1) & (dir_binary == 1)) |
            (s_binary == 1) |
            (b_binary == 1) |
            (white_binary == 1)
        ] = 1
        return combined

    # ----------------------------------------------------------
    # Stage 4 & 5: 直方图 + 滑动窗口
    # ----------------------------------------------------------
    def sliding_window(self, binary_warped, nwindows=9, margin=100, minpix=50):
        """滑动窗口多项式拟合"""
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = binary_warped.shape[0] // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xll) & (nonzerox < win_xlh)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xrl) & (nonzerox < win_xrh)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

        return left_fit, right_fit, left_lane_inds, right_lane_inds

    def search_around_poly(self, binary_warped, margin=80):
        """基于上一帧多项式的快速搜索"""
        if self.left_fit is None or self.right_fit is None:
            return self.sliding_window(binary_warped)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lf = self.left_fit
        rf = self.right_fit

        left_lane_inds = (
            (nonzerox > (lf[0] * nonzeroy ** 2 + lf[1] * nonzeroy + lf[2] - margin)) &
            (nonzerox < (lf[0] * nonzeroy ** 2 + lf[1] * nonzeroy + lf[2] + margin))
        )
        right_lane_inds = (
            (nonzerox > (rf[0] * nonzeroy ** 2 + rf[1] * nonzeroy + rf[2] - margin)) &
            (nonzerox < (rf[0] * nonzeroy ** 2 + rf[1] * nonzeroy + rf[2] + margin))
        )

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 100 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 100 else None

        if left_fit is None or right_fit is None:
            return self.sliding_window(binary_warped)

        return left_fit, right_fit, left_lane_inds, right_lane_inds

    # ----------------------------------------------------------
    # Stage 6: 曲率 & 偏移量
    # ----------------------------------------------------------
    def measure_curvature(self, binary_warped, left_fit, right_fit):
        """计算曲率半径（单位：米）"""
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # 转换为实际坐标
        left_fit_cr = np.polyfit(
            ploty * self.ym_per_pix,
            (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]) * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            ploty * self.ym_per_pix,
            (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]) * self.xm_per_pix, 2)

        y_eval = binary_warped.shape[0] * self.ym_per_pix

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.abs(2 * left_fit_cr[0] + 1e-7)
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.abs(2 * right_fit_cr[0] + 1e-7)

        return (left_curverad + right_curverad) / 2

    def measure_offset(self, binary_warped, left_fit, right_fit):
        """计算车辆相对车道中心的偏移量（单位：米）"""
        y_eval = binary_warped.shape[0]
        left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        lane_center = (left_x + right_x) / 2
        img_center = binary_warped.shape[1] / 2
        offset_m = (img_center - lane_center) * self.xm_per_pix
        return offset_m

    # ----------------------------------------------------------
    # 完整性校验 & 帧平滑
    # ----------------------------------------------------------
    def sanity_check(self, left_fit, right_fit, binary_warped):
        """
        检测结果完整性校验：
        - 车道宽度合理性
        - 左右曲率一致性
        - 多项式系数合理性
        """
        if left_fit is None or right_fit is None:
            return False

        h = binary_warped.shape[0]
        ploty = np.linspace(0, h - 1, h)
        left_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # 车道宽度检查
        lane_width = right_x - left_x
        mean_width = np.mean(lane_width)
        width_std = np.std(lane_width)

        if mean_width < 400 or mean_width > 1000 or width_std > 80:
            return False

        # 左右车道线应大致平行
        left_curv = ((1 + (2 * left_fit[0] * h + left_fit[1]) ** 2) ** 1.5) / \
                    np.abs(2 * left_fit[0] + 1e-7)
        right_curv = ((1 + (2 * right_fit[0] * h + right_fit[1]) ** 2) ** 1.5) / \
                     np.abs(2 * right_fit[0] + 1e-7)

        if left_curv > 100 and right_curv > 100:
            return True  # 两条直线，OK
        if max(left_curv, right_curv) / (min(left_curv, right_curv) + 1e-7) > 5:
            return False

        return True

    def smooth_fits(self, left_fit, right_fit):
        """加权移动平均平滑"""
        self.recent_left_fits.append(left_fit)
        self.recent_right_fits.append(right_fit)
        self.recent_left_fits = self.recent_left_fits[-self.smooth_window:]
        self.recent_right_fits = self.recent_right_fits[-self.smooth_window:]

        weights = np.arange(1, len(self.recent_left_fits) + 1, dtype=float)
        weights /= weights.sum()

        smoothed_left = np.average(self.recent_left_fits, axis=0, weights=weights)
        smoothed_right = np.average(self.recent_right_fits, axis=0, weights=weights)
        return smoothed_left, smoothed_right

    # ----------------------------------------------------------
    # Stage 7: 可视化
    # ----------------------------------------------------------
    def draw_lane(self, original_img, binary_warped, left_fit, right_fit, curvature, offset):
        """
        在原图上绘制检测到的车道区域（逆透视变换叠加）
        """
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # 在鸟瞰图上也绘制车道线
        cv2.polylines(color_warp, [np.int_(pts_left)], False, (255, 0, 0), 3)
        cv2.polylines(color_warp, [np.int_(pts_right)], False, (0, 0, 255), 3)

        # 逆透视变换
        newwarp = cv2.warpPerspective(color_warp, self.Minv,
                                      (original_img.shape[1], original_img.shape[0]))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

        # 文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f'Curvature: {curvature:.0f} m', (50, 50), font, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        direction = "left" if offset < 0 else "right"
        cv2.putText(result, f'Offset: {abs(offset):.2f} m {direction} of center',
                    (50, 100), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return result

    def draw_info_panel(self, image, curvature, offset):
        """
        绘制信息面板 - 显示各项检测参数
        """
        panel = image.copy()
        h, w = panel.shape[:2]

        # 半透明背景
        overlay = panel.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        cv2.putText(panel, "Lane Detection System", (20, 40), font, 0.6, (0, 255, 0), 2)
        cv2.putText(panel, f"Curvature: {curvature:.0f} m", (20, 70), font, 0.5, color, 1)
        direction = "LEFT" if offset < 0 else "RIGHT"
        cv2.putText(panel, f"Offset: {abs(offset):.2f}m {direction}", (20, 95), font, 0.5, color, 1)
        cv2.putText(panel, f"Lane Width: 3.7m (standard)", (20, 120), font, 0.5, color, 1)
        return panel

    # ----------------------------------------------------------
    # 完整处理管线
    # ----------------------------------------------------------
    def process_frame(self, frame):
        """
        处理单帧图像 - 完整 Pipeline
        """
        # Stage 1: 去畸变
        undist = self.undistort(frame)

        # Stage 2: 透视变换
        if self.M is None:
            self.setup_perspective(frame)
        warped = self.warp_perspective(undist)

        # Stage 3: 多阈值融合
        binary = self.combined_threshold(warped)

        # Stage 4 & 5: 车道线检测
        if self.detected:
            left_fit, right_fit, l_inds, r_inds = self.search_around_poly(binary)
        else:
            left_fit, right_fit, l_inds, r_inds = self.sliding_window(binary)

        # 完整性校验
        if left_fit is not None and right_fit is not None:
            if self.sanity_check(left_fit, right_fit, binary):
                self.detected = True
                self.left_fit, self.right_fit = self.smooth_fits(left_fit, right_fit)
            else:
                self.detected = False
                self.left_fit = left_fit
                self.right_fit = right_fit
        else:
            self.detected = False

        if self.left_fit is None or self.right_fit is None:
            return frame

        # Stage 6: 计算曲率和偏移
        curvature = self.measure_curvature(binary, self.left_fit, self.right_fit)
        offset = self.measure_offset(binary, self.left_fit, self.right_fit)

        # Stage 7: 可视化
        result = self.draw_lane(undist, binary, self.left_fit, self.right_fit, curvature, offset)
        return result
