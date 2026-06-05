"""
基础车道线检测模块
包含: 视频帧提取、ROI提取、直线检测、弯道检测
"""

import cv2
import numpy as np
import os


# ============================================================
# Task 1: 读取视频，逐帧提取图片并保存
# ============================================================
def extract_frames(video_path, output_dir, max_frames=None):
    """
    从视频中逐帧提取图片并保存到指定目录
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {os.path.basename(video_path)}, FPS={fps}, 总帧数={total}")

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"frame_{count:05d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    print(f"共提取 {count} 帧到 {output_dir}")
    return count


# ============================================================
# Task 2: 设置ROI感兴趣区域
# ============================================================
def extract_roi(image, vertices=None):
    """
    从图像中提取感兴趣区域(ROI)
    vertices: ROI区域的顶点坐标列表，格式为 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    if vertices is None:
        h, w = image.shape[:2]
        # 默认梯形ROI，适配大部分道路场景
        vertices = np.array([[
            (int(w * 0.05), h),            # 左下
            (int(w * 0.45), int(h * 0.6)), # 左上
            (int(w * 0.55), int(h * 0.6)), # 右上
            (int(w * 0.95), h)             # 右下
        ]], dtype=np.int32)

    mask = np.zeros_like(image)
    if len(image.shape) == 3:
        cv2.fillPoly(mask, vertices, (255, 255, 255))
    else:
        cv2.fillPoly(mask, vertices, 255)

    masked = cv2.bitwise_and(image, mask)
    return masked, mask


def auto_roi(image):
    """
    自动计算ROI区域 - 根据图像尺寸自适应
    """
    h, w = image.shape[:2]
    vertices = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.45), int(h * 0.6)),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.95), h)
    ]], dtype=np.int32)
    return extract_roi(image, vertices)


# ============================================================
# 辅助: 预处理 (灰度化 + 高斯模糊)
# ============================================================
def preprocess(image, kernel_size=5):
    """
    图像预处理：灰度转换 + 高斯模糊去噪
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return blur


# ============================================================
# Task 3: 基于边缘检测和霍夫变换的直车道线检测
# ============================================================
def detect_straight_lines(image, canny_low=50, canny_high=150,
                          hough_threshold=50, min_line_length=100,
                          max_line_gap=50, roi_vertices=None):
    """
    Canny边缘检测 + HoughLinesP 霍夫变换检测直线
    """
    h, w = image.shape[:2]
    original = image.copy()

    # 1. 预处理
    processed = preprocess(image)

    # 2. 提取ROI
    if roi_vertices is None:
        roi_vertices = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.45), int(h * 0.6)),
            (int(w * 0.55), int(h * 0.6)),
            (int(w * 0.95), h)
        ]], dtype=np.int32)
    roi, _ = extract_roi(processed, roi_vertices)

    # 3. Canny 边缘检测
    edges = cv2.Canny(roi, canny_low, canny_high)

    # 4. 霍夫变换检测直线段
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # 5. 绘制检测到的直线
    line_image = np.zeros_like(original)
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:  # 过滤水平线
                continue
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    # 拟合并绘制左右车道线
    detected_lines = []
    if left_lines:
        left_avg = _average_line(left_lines, h)
        if left_avg is not None:
            cv2.line(line_image, (left_avg[0], left_avg[1]),
                     (left_avg[2], left_avg[3]), (0, 0, 255), 3)
            detected_lines.append(left_avg)
    if right_lines:
        right_avg = _average_line(right_lines, h)
        if right_avg is not None:
            cv2.line(line_image, (right_avg[0], right_avg[1]),
                     (right_avg[2], right_avg[3]), (0, 0, 255), 3)
            detected_lines.append(right_avg)

    result = cv2.addWeighted(original, 0.8, line_image, 1.0, 0)
    return result, edges, detected_lines


def _average_line(lines, img_height):
    """
    对一组线段进行加权平均，返回一条从底部延伸到ROI顶部的代表性线段
    """
    if not lines:
        return None
    x_coords, y_coords, weights = [], [], []
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
        weights.extend([length, length])

    poly = np.polyfit(y_coords, x_coords, 1)
    y_bottom = img_height
    y_top = int(img_height * 0.6)
    x_bottom = int(np.polyval(poly, y_bottom))
    x_top = int(np.polyval(poly, y_top))
    return [x_bottom, y_bottom, x_top, y_top]


# ============================================================
# Task 4: 基于颜色阈值 + 滑动窗口的弯道车道线检测
# ============================================================

def color_threshold(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    颜色与梯度阈值化：
    - HLS色彩空间S通道（对阴影鲁棒）
    - Sobel X方向梯度（检测垂直边缘）
    """
    img = np.copy(image)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel X 梯度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-7))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # S通道阈值
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # 合并两个条件
    combined = np.zeros_like(sxbinary)
    combined[(sxbinary == 1) | (s_binary == 1)] = 255
    return combined


def color_threshold_advanced(image):
    """
    多色彩空间组合阈值：
    - HLS S通道（检测黄色/白色线，抗阴影）
    - LAB B通道（专门检测黄色线）
    - HSV 白色掩码（专门检测白色线）
    - Sobel X梯度（垂直边缘）
    - Sobel方向阈值（过滤噪声）
    """
    img = np.copy(image)

    # --- HLS S通道 ---
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 120) & (s_channel <= 255)] = 1

    # --- LAB B通道（黄色线专用） ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 155) & (b_channel <= 200)] = 1

    # --- HSV 白色掩码 ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_binary = (white_mask / 255).astype(np.uint8)

    # --- Sobel X 梯度 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-7))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    # --- Sobel Y 梯度 ---
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255 * abs_sobely / (np.max(abs_sobely) + 1e-7))
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= 20) & (scaled_sobely <= 100)] = 1

    # --- Sobel 方向阈值 ---
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= 0.7) & (absgraddir <= 1.3)] = 1

    # --- Sobel 幅值 ---
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_mag = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-7))
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag >= 30) & (scaled_mag <= 100)] = 1

    # 融合策略: 多种条件 OR 组合
    combined = np.zeros_like(gray)
    combined[
        (s_binary == 1) |
        (b_binary == 1) |
        (white_binary == 1) |
        ((sxbinary == 1) & (sybinary == 1)) |
        ((mag_binary == 1) & (dir_binary == 1))
    ] = 255
    return combined


def perspective_transform(image, src=None, dst=None):
    """
    透视变换 - 鸟瞰图视角
    将梯形道路视角转为矩形俯视图，使车道线平行
    """
    h, w = image.shape[:2]
    if src is None:
        src = np.float32([
            [int(w * 0.45), int(h * 0.63)],  # 左上
            [int(w * 0.55), int(h * 0.63)],  # 右上
            [int(w * 0.90), int(h * 0.95)],  # 右下
            [int(w * 0.10), int(h * 0.95)]   # 左下
        ])
    if dst is None:
        dst = np.float32([
            [int(w * 0.20), 0],
            [int(w * 0.80), 0],
            [int(w * 0.80), h],
            [int(w * 0.20), h]
        ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def sliding_window_detection(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    滑动窗口多项式拟合 - 车道线检测核心算法
    从底部向上追踪车道线像素，用二阶多项式拟合
    """
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
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

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


def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    """
    基于上一帧多项式的快速搜索（跳过滑动窗口，提高视频处理效率）
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (
        (nonzerox > (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
        (nonzerox < (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] + margin))
    )
    right_lane_inds = (
        (nonzerox > (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
        (nonzerox < (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] + margin))
    )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new = np.polyfit(lefty, leftx, 2) if len(leftx) > 100 else None
    right_fit_new = np.polyfit(righty, rightx, 2) if len(rightx) > 100 else None

    return left_fit_new, right_fit_new


def draw_sliding_windows(binary_warped, left_lane_inds, right_lane_inds, margin=100, nwindows=9):
    """
    可视化滑动窗口搜索过程
    """
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype(np.uint8) * 255
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = binary_warped.shape[0] // nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # 绘制窗口矩形
        cv2.rectangle(out_img, (leftx_current - margin, win_y_low),
                      (leftx_current + margin, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (rightx_current - margin, win_y_low),
                      (rightx_current + margin, win_y_high), (0, 255, 0), 2)
        # 更新窗口中心
        if len(left_lane_inds) > 0:
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= leftx_current - margin) & (nonzerox < leftx_current + margin)).nonzero()[0]
            if len(good_left) > 50:
                leftx_current = int(np.mean(nonzerox[good_left]))
        if len(right_lane_inds) > 0:
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= rightx_current - margin) & (nonzerox < rightx_current + margin)).nonzero()[0]
            if len(good_right) > 50:
                rightx_current = int(np.mean(nonzerox[good_right]))

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return out_img
