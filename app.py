"""
OpenCV 智能视觉系统 - 可视化Web界面
========================================
交互式可视化平台，集成车道线检测与手势识别

技术栈: Gradio, OpenCV, MediaPipe, NumPy, Matplotlib

功能:
  1. 车道线检测 - 图片/视频车道线检测
  2. 手势识别 - 手部关键点检测、手势分类、手指计数
  3. 虚拟画板 - 手势控制绘画
  4. Pipeline可视化 - 8阶段处理过程逐步展示
  5. 方法对比 - 多种阈值方法效果对比
"""

import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 项目路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES = os.path.join(ROOT_DIR, 'test_images')
TEST_VIDEOS = os.path.join(ROOT_DIR, 'test_videos')

from src.vehicle_speed_distance import (
    detect_speed_distance_image as _detect_sd_image,
    detect_speed_distance_video as _detect_sd_video,
    analyze_detections as _analyze_sd,
    CentroidTracker, VehicleDetector,
    process_frame as _process_sd_frame,
)

# ============================================================
# 核心检测函数
# ============================================================

def preprocess(image, kernel_size=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    return blur


def extract_roi(image, vertices=None):
    if vertices is None:
        h, w = image.shape[:2]
        vertices = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.45), int(h * 0.6)),
            (int(w * 0.55), int(h * 0.6)),
            (int(w * 0.95), h)
        ]], dtype=np.int32)
    mask = np.zeros_like(image)
    if len(image.shape) == 3:
        cv2.fillPoly(mask, vertices, (255, 255, 255))
    else:
        cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked, mask


def color_threshold_advanced(image):
    img = np.copy(image)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 120) & (s_channel <= 255)] = 1

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 155) & (b_channel <= 200)] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_binary = (white_mask / 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-7))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255 * abs_sobely / (np.max(abs_sobely) + 1e-7))
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= 20) & (scaled_sobely <= 100)] = 1

    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= 0.7) & (absgraddir <= 1.3)] = 1

    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_mag = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-7))
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag >= 30) & (scaled_mag <= 100)] = 1

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
    h, w = image.shape[:2]
    if src is None:
        src = np.float32([
            [int(w * 0.45), int(h * 0.63)],
            [int(w * 0.55), int(h * 0.63)],
            [int(w * 0.90), int(h * 0.95)],
            [int(w * 0.10), int(h * 0.95)]
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
    return warped, M, Minv, src, dst


def sliding_window_detection(binary_warped, nwindows=9, margin=100, minpix=50):
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

    window_info = []  # 记录窗口信息用于可视化

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

        window_info.append({
            'left': (win_xll, win_y_low, win_xlh, win_y_high),
            'right': (win_xrl, win_y_low, win_xrh, win_y_high),
            'left_count': len(good_left),
            'right_count': len(good_right)
        })

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

    return left_fit, right_fit, left_lane_inds, right_lane_inds, window_info


def measure_curvature(binary_warped, left_fit, right_fit):
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fit_cr = np.polyfit(
        ploty * ym_per_pix,
        (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]) * xm_per_pix, 2)
    right_fit_cr = np.polyfit(
        ploty * ym_per_pix,
        (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]) * xm_per_pix, 2)
    y_eval = binary_warped.shape[0] * ym_per_pix
    left_curv = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / \
                np.abs(2 * left_fit_cr[0] + 1e-7)
    right_curv = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / \
                 np.abs(2 * right_fit_cr[0] + 1e-7)
    return (left_curv + right_curv) / 2


def measure_offset(binary_warped, left_fit, right_fit):
    xm_per_pix = 3.7 / 700
    y_eval = binary_warped.shape[0]
    left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    img_center = binary_warped.shape[1] / 2
    return (img_center - lane_center) * xm_per_pix


def draw_lane_overlay(original_img, binary_warped, left_fit, right_fit, Minv):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, [np.int_(pts_left)], False, (255, 0, 0), 3)
    cv2.polylines(color_warp, [np.int_(pts_right)], False, (0, 0, 255), 3)

    newwarp = cv2.warpPerspective(color_warp, Minv,
                                  (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result


def draw_info_on_image(image, curvature, offset):
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 半透明背景
    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    cv2.putText(result, "Lane Detection System", (20, 40), font, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f"Curvature: {curvature:.0f} m", (20, 70), font, 0.55, (255, 255, 255), 1)
    direction = "LEFT" if offset < 0 else "RIGHT"
    cv2.putText(result, f"Offset: {abs(offset):.2f}m {direction}", (20, 95), font, 0.55, (255, 255, 255), 1)
    return result


# ============================================================
# Gradio 界面功能函数
# ============================================================

def detect_lane_image(image):
    """单张图片车道线检测"""
    if image is None:
        return None, "请上传图片"

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 完整pipeline
    binary = color_threshold_advanced(img)
    warped, M, Minv, src, dst = perspective_transform(binary)
    left_fit, right_fit, l_inds, r_inds, win_info = sliding_window_detection(warped)

    if left_fit is not None and right_fit is not None:
        curvature = measure_curvature(warped, left_fit, right_fit)
        offset = measure_offset(warped, left_fit, right_fit)
        result = draw_lane_overlay(img, warped, left_fit, right_fit, Minv)
        result = draw_info_on_image(result, curvature, offset)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        info = f"✅ 检测成功\n"
        info += f"📏 曲率半径: {curvature:.0f} m\n"
        direction = "偏左" if offset < 0 else "偏右"
        info += f"↔️ 车道偏移: {abs(offset):.2f} m {direction}\n"
        info += f"📊 左车道线像素点: {len(l_inds)}\n"
        info += f"📊 右车道线像素点: {len(r_inds)}"

        return result_rgb, info
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "❌ 未检测到车道线，请尝试其他图片"


def show_pipeline_stages(image):
    """展示8阶段Pipeline处理过程"""
    if image is None:
        return [None] * 8

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    stages = []

    # Stage 1: 原图
    stages.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Stage 2: 灰度 + 高斯模糊
    gray = preprocess(img)
    stages.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    # Stage 3: ROI提取
    roi, mask = extract_roi(img)
    stages.append(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # Stage 4: Canny边缘
    edges = cv2.Canny(gray, 50, 150)
    roi_edges, _ = extract_roi(edges)
    stages.append(cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2RGB))

    # Stage 5: 多阈值融合
    binary = color_threshold_advanced(img)
    stages.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

    # Stage 6: 透视变换
    warped, M, Minv, src, dst = perspective_transform(binary)
    stages.append(cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB))

    # Stage 7: 滑动窗口
    left_fit, right_fit, l_inds, r_inds, win_info = sliding_window_detection(warped)
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    window_vis = np.dstack((warped, warped, warped)).astype(np.uint8) * 255
    if len(l_inds) > 0:
        window_vis[nonzeroy[l_inds], nonzerox[l_inds]] = [255, 0, 0]
    if len(r_inds) > 0:
        window_vis[nonzeroy[r_inds], nonzerox[r_inds]] = [0, 0, 255]
    for wi in win_info:
        xl, yl, xh, yh = wi['left']
        cv2.rectangle(window_vis, (xl, yl), (xh, yh), (0, 255, 0), 2)
        xr, yr, xrh, yrh = wi['right']
        cv2.rectangle(window_vis, (xr, yr), (xrh, yrh), (0, 255, 0), 2)
    if left_fit is not None:
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts_l = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        pts_r = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(window_vis, pts_l, False, (255, 255, 0), 2)
        cv2.polylines(window_vis, pts_r, False, (255, 255, 0), 2)
    stages.append(window_vis)

    # Stage 8: 最终叠加
    if left_fit is not None and right_fit is not None:
        result = draw_lane_overlay(img, warped, left_fit, right_fit, Minv)
        curvature = measure_curvature(warped, left_fit, right_fit)
        offset = measure_offset(warped, left_fit, right_fit)
        result = draw_info_on_image(result, curvature, offset)
        stages.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        stages.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return stages


def show_threshold_comparison(image):
    """展示不同阈值方法对比"""
    if image is None:
        return [None] * 6

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = []

    # HLS S通道
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    s = hls[:, :, 2]
    s_bin = np.zeros_like(s)
    s_bin[(s >= 170) & (s <= 255)] = 255
    results.append(cv2.cvtColor(s_bin.astype(np.uint8), cv2.COLOR_GRAY2RGB))

    # LAB B通道
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = lab[:, :, 2]
    b_bin = np.zeros_like(b)
    b_bin[(b >= 155) & (b <= 200)] = 255
    results.append(cv2.cvtColor(b_bin.astype(np.uint8), cv2.COLOR_GRAY2RGB))

    # HSV白色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    results.append(cv2.cvtColor(white_mask, cv2.COLOR_GRAY2RGB))

    # Sobel X
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sx = np.absolute(sobelx)
    scaled_sx = np.uint8(255 * abs_sx / (np.max(abs_sx) + 1e-7))
    sx_bin = np.zeros_like(scaled_sx)
    sx_bin[(scaled_sx >= 20) & (scaled_sx <= 100)] = 255
    results.append(cv2.cvtColor(sx_bin, cv2.COLOR_GRAY2RGB))

    # Sobel方向
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sy = np.absolute(sobely)
    absgraddir = np.arctan2(abs_sy, abs_sx)
    dir_bin = np.zeros_like(absgraddir)
    dir_bin[(absgraddir >= 0.7) & (absgraddir <= 1.3)] = 255
    results.append(cv2.cvtColor(dir_bin.astype(np.uint8), cv2.COLOR_GRAY2RGB))

    # 融合结果
    combined = color_threshold_advanced(img)
    results.append(cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB))

    return results


def show_perspective_view(image):
    """展示透视变换效果"""
    if image is None:
        return None, None

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # 绘制源点四边形
    src = np.float32([
        [int(w * 0.45), int(h * 0.63)],
        [int(w * 0.55), int(h * 0.63)],
        [int(w * 0.90), int(h * 0.95)],
        [int(w * 0.10), int(h * 0.95)]
    ])
    img_show = img.copy()
    pts = src.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img_show, [pts], True, (0, 255, 0), 3)
    for i, pt in enumerate(src):
        cv2.circle(img_show, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
        cv2.putText(img_show, f'S{i}', (int(pt[0]) + 12, int(pt[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 透视变换
    binary = color_threshold_advanced(img)
    warped, _, _, _, _ = perspective_transform(binary)

    return cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB), cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)


def convert_to_browser_compatible(input_path, output_path):
    """将视频转换为浏览器兼容的H.264格式"""
    import subprocess

    # 方法1: 尝试系统ffmpeg
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 方法2: 使用imageio-ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception:
        pass

    return False


def detect_video(video):
    """视频车道线检测"""
    if video is None:
        return None, "请上传视频"

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, "无法打开视频"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建临时输出文件 - 先用OpenCV写入
    temp_path = tempfile.mktemp(suffix='.avi')
    output_path = tempfile.mktemp(suffix='.mp4')

    # 尝试多种编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        # 备选编码器
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        temp_path = tempfile.mktemp(suffix='.avi')
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    frame_count = 0
    detected_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        binary = color_threshold_advanced(frame)
        warped, M, Minv, _, _ = perspective_transform(binary)
        left_fit, right_fit, _, _, _ = sliding_window_detection(warped)

        if left_fit is not None and right_fit is not None:
            curvature = measure_curvature(warped, left_fit, right_fit)
            offset = measure_offset(warped, left_fit, right_fit)
            result = draw_lane_overlay(frame, warped, left_fit, right_fit, Minv)
            result = draw_info_on_image(result, curvature, offset)
            detected_count += 1
        else:
            result = frame

        writer.write(result)
        frame_count += 1

    cap.release()
    writer.release()

    # 转换为浏览器兼容格式
    success = convert_to_browser_compatible(temp_path, output_path)

    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if not success:
        # 如果ffmpeg不可用，直接返回avi文件
        output_path = temp_path

    info = f"✅ 视频处理完成\n"
    info += f"📊 总帧数: {frame_count}\n"
    info += f"📊 检测成功: {detected_count} 帧\n"
    info += f"📊 检测率: {detected_count/frame_count*100:.1f}%\n"
    info += f"📊 分辨率: {w}x{h}\n"
    info += f"📊 帧率: {fps:.1f} fps"

    return output_path, info


def load_example_image(example_name):
    """加载示例图片"""
    path = os.path.join(TEST_IMAGES, example_name)
    if os.path.exists(path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


# ============================================================
# 手势识别功能函数
# ============================================================

def recognize_gesture_image(image):
    """图片手势识别"""
    if image is None:
        return None, "请上传包含手部的图片"

    from src.gesture_recognition import HandDetector, process_gesture_image

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result_img, info = process_gesture_image(img)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # 构建信息文本
    if info["hands_detected"] == 0:
        return result_rgb, "未检测到手部，请上传包含手部的图片"

    text = f"检测到 {info['hands_detected']} 只手\n\n"
    for g in info["gestures"]:
        text += f"手 {g['hand']}:\n"
        text += f"  手势: {g['gesture']} - {g['description']}\n"
        text += f"  手指: {g['fingers']}\n"
        text += f"  竖起数量: {g['count']}\n\n"

    return result_rgb, text


def recognize_gesture_video(video):
    """视频手势识别"""
    if video is None:
        return None, "请上传包含手部的视频"

    from src.gesture_recognition import HandDetector, process_gesture_video_frame

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, "无法打开视频"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建临时输出文件
    temp_path = tempfile.mktemp(suffix='.avi')
    output_path = tempfile.mktemp(suffix='.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))

    detector = HandDetector(max_hands=2)
    frame_count = 0
    gesture_stats = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_img, gesture_info = process_gesture_video_frame(frame, detector)

        # 统计手势
        for g in gesture_info["gestures"]:
            gesture_name = g["gesture"]
            gesture_stats[gesture_name] = gesture_stats.get(gesture_name, 0) + 1

        writer.write(result_img)
        frame_count += 1

    cap.release()
    writer.release()

    # 转换格式
    success = convert_to_browser_compatible(temp_path, output_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if not success:
        output_path = temp_path

    # 构建统计信息
    info = f"视频处理完成\n"
    info += f"总帧数: {frame_count}\n"
    info += f"分辨率: {w}x{h}\n\n"
    info += "手势统计:\n"
    for gesture, count in sorted(gesture_stats.items(), key=lambda x: -x[1]):
        percentage = count / frame_count * 100
        info += f"  {gesture}: {count} 帧 ({percentage:.1f}%)\n"

    return output_path, info


def process_virtual_painter_frame(image):
    """虚拟画板处理"""
    if image is None:
        return None, "请上传包含手部的图片"

    from src.gesture_recognition import HandDetector, VirtualPainter

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    detector = HandDetector(max_hands=1)
    painter = VirtualPainter()
    painter.init_canvas(h, w)

    # 设置不同颜色的画笔
    painter.draw_color = (0, 0, 255)  # 红色

    img = detector.find_hands(img)
    landmarks = detector.find_position(img, 0, draw=False)

    if landmarks:
        result_img, mode = painter.process_frame(img, landmarks, detector)
        result_img = painter.overlay_canvas(result_img)

        gesture, desc = detector.recognize_gesture()
        fingers = detector.count_fingers()

        info = f"模式: {mode}\n"
        info += f"手势: {gesture} - {desc}\n"
        info += f"竖起手指: {fingers}\n"
        info += "说明:\n"
        info += "  - 食指竖起: 绘画模式\n"
        info += "  - 食指+中指: 选择模式\n"
        info += "  - 其他: 空闲模式"
    else:
        result_img = img
        info = "未检测到手部\n请上传包含手部的图片"

    return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), info


# ============================================================
# 实时摄像头处理 (全局实例避免重复初始化)
# ============================================================

# 全局检测器实例
_global_detector = None
_global_painter = None

def get_global_detector():
    """获取全局检测器实例"""
    global _global_detector
    if _global_detector is None:
        from src.gesture_recognition import HandDetector
        _global_detector = HandDetector(max_hands=2, detection_confidence=0.7, tracking_confidence=0.5)
    return _global_detector

def get_global_painter():
    """获取全局画板实例"""
    global _global_painter
    if _global_painter is None:
        from src.gesture_recognition import VirtualPainter
        _global_painter = VirtualPainter()
    return _global_painter


def process_webcam_frame(frame):
    """
    处理摄像头帧 - 实时手势识别

    参数:
        frame: RGB图像 (来自Gradio摄像头)

    返回:
        result: 处理后的RGB图像
    """
    if frame is None:
        return None

    try:
        from src.gesture_recognition import process_gesture_video_frame

        # RGB转BGR (OpenCV格式)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detector = get_global_detector()
        result_img, gesture_info = process_gesture_video_frame(img, detector)

        # 确保result_img是numpy数组
        if result_img is None or not isinstance(result_img, np.ndarray):
            return frame

        # BGR转RGB (Gradio格式)
        return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[ERROR] process_webcam_frame: {e}")
        return frame


def process_webcam_painter(frame):
    """
    处理摄像头帧 - 虚拟画板模式

    参数:
        frame: RGB图像 (来自Gradio摄像头)

    返回:
        result: 处理后的RGB图像
    """
    if frame is None:
        return None

    try:
        from src.gesture_recognition import HandDetector, VirtualPainter

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape

        detector = get_global_detector()
        painter = get_global_painter()

        # 初始化画布
        if painter.canvas is None or painter.canvas.shape[:2] != (h, w):
            painter.init_canvas(h, w)

        img = detector.find_hands(img, draw=True)
        landmarks = detector.find_position(img, 0, draw=False)

        if landmarks:
            gesture, desc = detector.recognize_gesture()
            fingers = detector.fingers_up()

            # 绘画模式提示
            if fingers and len(fingers) >= 2:
                if fingers[1] == 1 and fingers[2] == 0:
                    cv2.putText(img, "DRAWING MODE", (w//2 - 100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif fingers[1] == 1 and fingers[2] == 1:
                    cv2.putText(img, "SELECT MODE", (w//2 - 100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            img, mode = painter.process_frame(img, landmarks, detector)

        img = painter.overlay_canvas(img)

        # 确保img是numpy数组
        if img is None or not isinstance(img, np.ndarray):
            return frame

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[ERROR] process_webcam_painter: {e}")
        return frame


def clear_painter_canvas():
    """清空画布"""
    global _global_painter
    if _global_painter is not None:
        _global_painter.clear_canvas()
    return "画布已清空"


# ============================================================
# 车速车距检测处理函数
# ============================================================

def _sd_image_fn(image):
    if image is None:
        return None, "请上传图片"
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result, info = _detect_sd_image(image)
    if not info:
        return result, "未检测到运动目标（MOG2 背景减除需要连续帧，单张图片效果有限）"
    lines = [f"[{d['class']}]  距离={d['distance_m']}m" for d in info]
    return result, f"检测到 {len(info)} 个目标\n" + "\n".join(lines)


def _sd_video_fn(video):
    if video is None:
        return None, "请上传视频"
    import gradio as gr
    output_path, all_info = _detect_sd_video(
        video, progress_cb=lambda c, t: gr.Info(f"处理进度: {c}/{t}") if c % 30 == 0 else None
    )
    stats = _analyze_sd(all_info)
    txt = (
        f"处理完成\n"
        f"总帧数: {stats['total_frames']}\n"
        f"总检测数: {stats['total_detections']}\n"
    )
    if 'distance_min' in stats:
        txt += f"最近距离: {stats['distance_min']}m  最远: {stats['distance_max']}m  平均: {stats['distance_avg']}m\n"
    if 'speed_min' in stats:
        txt += f"速度范围: {stats['speed_min']}~{stats['speed_max']}km/h\n"
    return output_path, txt


# ============================================================
# 构建 Gradio 界面
# ============================================================

def build_app():
    """构建完整的Gradio应用"""

    # 自定义CSS
    custom_css = """
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.3em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .stage-label {
        font-weight: bold;
        color: #4a5568;
    }
    """

    with gr.Blocks(
        title="🚗 OpenCV 智能视觉系统",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        )
    ) as app:

        # ---- 标题 ----
        gr.HTML("""
        <div class="main-title">🚗 OpenCV 智能视觉系统</div>
        <div class="subtitle">
            集成车道线检测与手势识别的交互式可视化平台<br>
            车道线检测 | 手势识别 | 虚拟画板 | Pipeline可视化 | 多阈值对比
        </div>
        """)

        # ==================== Tab 1: 图片检测 ====================
        with gr.Tab("🖼️ 图片检测", id="image"):
            gr.Markdown("### 上传图片或选择示例，实时检测车道线")

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="📤 上传图片",
                        type="numpy",
                        height=400
                    )

                    # 示例图片选择
                    gr.Markdown("#### 📁 示例图片")
                    example_images = sorted([
                        f for f in os.listdir(TEST_IMAGES)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                    ]) if os.path.isdir(TEST_IMAGES) else []

                    with gr.Row():
                        for i, name in enumerate(example_images[:3]):
                            btn = gr.Button(name.split('.')[0], size="sm", variant="secondary")
                            btn.click(
                                fn=lambda n=name: load_example_image(n),
                                outputs=input_image
                            )
                    with gr.Row():
                        for i, name in enumerate(example_images[3:6]):
                            btn = gr.Button(name.split('.')[0], size="sm", variant="secondary")
                            btn.click(
                                fn=lambda n=name: load_example_image(n),
                                outputs=input_image
                            )

                    detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="🎯 检测结果", height=400)
                    detection_info = gr.Textbox(
                        label="📊 检测信息",
                        lines=5,
                        interactive=False
                    )

            detect_btn.click(
                fn=detect_lane_image,
                inputs=input_image,
                outputs=[output_image, detection_info]
            )

        # ==================== Tab 2: Pipeline可视化 ====================
        with gr.Tab("🔬 Pipeline 可视化", id="pipeline"):
            gr.Markdown("### 8阶段处理流程逐步展示")
            gr.Markdown("""
            | 阶段 | 处理步骤 | 说明 |
            |------|----------|------|
            | 1 | 原始图像 | 输入的道路图像 |
            | 2 | 灰度化+高斯模糊 | 预处理去噪 |
            | 3 | ROI提取 | 感兴趣区域裁剪 |
            | 4 | Canny边缘检测 | 边缘信息提取 |
            | 5 | 多阈值融合 | HLS/LAB/HSV + Sobel |
            | 6 | 透视变换 | 鸟瞰图视角 |
            | 7 | 滑动窗口检测 | 车道线像素聚类+多项式拟合 |
            | 8 | 最终叠加 | 逆透视变换回原图 |
            """)

            with gr.Row():
                pipeline_input = gr.Image(
                    label="📤 上传图片",
                    type="numpy",
                    scale=1
                )
                pipeline_btn = gr.Button("🔬 展示Pipeline", variant="primary", scale=0)

            with gr.Row():
                stage1 = gr.Image(label="1️⃣ 原始图像", show_label=True)
                stage2 = gr.Image(label="2️⃣ 灰度+模糊", show_label=True)
                stage3 = gr.Image(label="3️⃣ ROI提取", show_label=True)
                stage4 = gr.Image(label="4️⃣ Canny边缘", show_label=True)

            with gr.Row():
                stage5 = gr.Image(label="5️⃣ 多阈值融合", show_label=True)
                stage6 = gr.Image(label="6️⃣ 透视变换", show_label=True)
                stage7 = gr.Image(label="7️⃣ 滑动窗口", show_label=True)
                stage8 = gr.Image(label="8️⃣ 最终结果", show_label=True)

            pipeline_btn.click(
                fn=show_pipeline_stages,
                inputs=pipeline_input,
                outputs=[stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8]
            )

        # ==================== Tab 3: 阈值方法对比 ====================
        with gr.Tab("📊 阈值方法对比", id="threshold"):
            gr.Markdown("### 不同阈值方法效果对比")
            gr.Markdown("""
            对比6种阈值方法的检测效果:
            - **HLS S通道**: 对阴影鲁棒，检测黄色/白色线
            - **LAB B通道**: 黄色线专用检测
            - **HSV白色掩码**: 白色线专用检测
            - **Sobel X梯度**: 垂直边缘检测
            - **Sobel方向**: 边缘角度过滤
            - **融合结果**: 多种方法OR组合
            """)

            with gr.Row():
                thresh_input = gr.Image(
                    label="📤 上传图片",
                    type="numpy",
                    scale=1
                )
                thresh_btn = gr.Button("📊 对比分析", variant="primary", scale=0)

            with gr.Row():
                t1 = gr.Image(label="HLS S通道", show_label=True)
                t2 = gr.Image(label="LAB B通道", show_label=True)
                t3 = gr.Image(label="HSV白色掩码", show_label=True)

            with gr.Row():
                t4 = gr.Image(label="Sobel X梯度", show_label=True)
                t5 = gr.Image(label="Sobel方向", show_label=True)
                t6 = gr.Image(label="✅ 融合结果", show_label=True)

            thresh_btn.click(
                fn=show_threshold_comparison,
                inputs=thresh_input,
                outputs=[t1, t2, t3, t4, t5, t6]
            )

        # ==================== Tab 4: 透视变换 ====================
        with gr.Tab("🔄 透视变换", id="perspective"):
            gr.Markdown("### 透视变换 - 鸟瞰图视角")
            gr.Markdown("""
            将梯形道路视角转为矩形俯视图，使车道线变为平行线，便于后续多项式拟合。
            - **左侧**: 原图上标记的源点四边形 (绿色)
            - **右侧**: 透视变换后的鸟瞰图
            """)

            with gr.Row():
                persp_input = gr.Image(
                    label="📤 上传图片",
                    type="numpy",
                    scale=1
                )
                persp_btn = gr.Button("🔄 透视变换", variant="primary", scale=0)

            with gr.Row():
                persp_src = gr.Image(label="📍 源点标记", show_label=True)
                persp_dst = gr.Image(label="🐦 鸟瞰图", show_label=True)

            persp_btn.click(
                fn=show_perspective_view,
                inputs=persp_input,
                outputs=[persp_src, persp_dst]
            )

        # ==================== Tab 5: 视频检测 ====================
        with gr.Tab("🎬 视频检测", id="video"):
            gr.Markdown("### 上传视频，逐帧检测车道线")
            gr.Markdown("""
            支持MP4/AVI格式视频，使用高级7阶段管线逐帧处理。
            处理后的视频会显示车道线叠加、曲率半径和偏移量信息。
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="📤 上传视频",
                        height=350
                    )

                    gr.Markdown("#### 📁 示例视频")
                    with gr.Row():
                        if os.path.isdir(TEST_VIDEOS):
                            for vf in os.listdir(TEST_VIDEOS):
                                if vf.endswith('.mp4'):
                                    vpath = os.path.join(TEST_VIDEOS, vf)
                                    btn = gr.Button(vf.split('.')[0], size="sm", variant="secondary")
                                    btn.click(
                                        fn=lambda p=vpath: p,
                                        outputs=video_input
                                    )

                    video_btn = gr.Button("🎬 开始处理", variant="primary", size="lg")

                with gr.Column(scale=1):
                    video_output = gr.Video(label="🎯 处理结果", height=350)
                    video_info = gr.Textbox(
                        label="📊 处理信息",
                        lines=6,
                        interactive=False
                    )

            video_btn.click(
                fn=detect_video,
                inputs=video_input,
                outputs=[video_output, video_info]
            )

        # ==================== Tab 6: 手势识别 ====================
        with gr.Tab("🤚 手势识别", id="gesture"):
            gr.Markdown("### 基于 MediaPipe 的手部检测与手势识别")
            gr.Markdown("""
            支持功能:
            - **手部关键点检测**: 21个关键点实时追踪
            - **手势识别**: 石头/剪刀/布、数字1-5、OK、竖大拇指等
            - **手指计数**: 实时统计竖起的手指数量
            - **虚拟画板**: 使用手势控制绘画
            """)

            with gr.Tabs():
                # 子Tab1: 图片手势识别
                with gr.Tab("🖼️ 图片识别"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gesture_input = gr.Image(
                                label="📤 上传包含手部的图片",
                                type="numpy",
                                height=400
                            )
                            gesture_btn = gr.Button("🤚 开始识别", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            gesture_output = gr.Image(label="🎯 识别结果", height=400)
                            gesture_info = gr.Textbox(
                                label="📊 识别信息",
                                lines=8,
                                interactive=False
                            )

                    gesture_btn.click(
                        fn=recognize_gesture_image,
                        inputs=gesture_input,
                        outputs=[gesture_output, gesture_info]
                    )

                # 子Tab2: 视频手势识别
                with gr.Tab("🎬 视频识别"):
                    gr.Markdown("上传包含手部动作的视频，逐帧识别手势")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gesture_video_input = gr.Video(
                                label="📤 上传视频",
                                height=350
                            )
                            gesture_video_btn = gr.Button("🎬 开始识别", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            gesture_video_output = gr.Video(label="🎯 识别结果", height=350)
                            gesture_video_info = gr.Textbox(
                                label="📊 识别统计",
                                lines=8,
                                interactive=False
                            )

                    gesture_video_btn.click(
                        fn=recognize_gesture_video,
                        inputs=gesture_video_input,
                        outputs=[gesture_video_output, gesture_video_info]
                    )

                # 子Tab3: 上传图片虚拟画板
                with gr.Tab("🎨 虚拟画板"):
                    gr.Markdown("""
                    ### 手势控制画板
                    - **食指竖起**: 绘画模式
                    - **食指+中指**: 选择模式
                    - **其他手势**: 空闲模式
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            painter_input = gr.Image(
                                label="📤 上传包含手部的图片",
                                type="numpy",
                                height=400
                            )
                            painter_btn = gr.Button("🎨 开始绘画", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            painter_output = gr.Image(label="🎯 绘画结果", height=400)
                            painter_info = gr.Textbox(
                                label="📊 模式信息",
                                lines=6,
                                interactive=False
                            )

                    painter_btn.click(
                        fn=process_virtual_painter_frame,
                        inputs=painter_input,
                        outputs=[painter_output, painter_info]
                    )

                # 子Tab4: 实时虚拟画板
                with gr.Tab("🎨 实时画板"):
                    gr.Markdown("""
                    ### 实时虚拟画板
                    用手指在摄像头前绘画！

                    **操作方式:**
                    - ✊ **空闲模式**: 握拳暂停
                    - ☝️ **绘画模式**: 只伸出食指开始绘画
                    - ✌️ **选择模式**: 伸出食指+中指选择颜色/位置
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            realtime_painter_input = gr.Image(
                                label="📹 摄像头输入",
                                source="webcam",
                                type="numpy",
                                streaming=True,
                                height=450
                            )

                        with gr.Column(scale=1):
                            realtime_painter_output = gr.Image(
                                label="🎯 绘画结果",
                                height=450
                            )

                            with gr.Row():
                                clear_btn = gr.Button("🗑️ 清空画布", variant="secondary")
                                clear_output = gr.Textbox(label="状态", interactive=False, lines=1)

                    # 实时处理绑定
                    realtime_painter_input.change(
                        fn=process_webcam_painter,
                        inputs=realtime_painter_input,
                        outputs=realtime_painter_output
                    )

                    # 清空画布按钮
                    clear_btn.click(
                        fn=clear_painter_canvas,
                        outputs=clear_output
                    )

        # ==================== Tab 7: 车速车距检测 ====================
        with gr.Tab("🚗 车速车距", id="speed_distance"):
            gr.Markdown("### 基于 OpenCV 的车速与车距估算")
            gr.Markdown("""
            **检测原理:**
            - 🎯 **车辆检测**: MOG2 背景减除 → 形态学去噪 → 轮廓检测 → NMS
            - 📏 **车距估算**: 针孔相机模型 `D = (W_real × f) / W_pixel`
            - 🚀 **车速估算**: 跨帧质心跟踪 → 纵向位移变化 / 时间间隔

            **距离图例:** 🔴 近距离(<20m)  🟠 中距离(20~50m)  🟢 远距离(>50m)
            """)

            with gr.Tabs():
                # 子Tab: 图片检测
                with gr.Tab("🖼️ 图片检测"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            sd_img_input = gr.Image(label="📤 上传道路图片", type="numpy", height=400)
                            sd_img_btn = gr.Button("🚗 开始检测", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            sd_img_output = gr.Image(label="🎯 检测结果", height=400)
                            sd_img_info = gr.Textbox(label="📊 检测信息", lines=8, interactive=False)

                    sd_img_btn.click(fn=_sd_image_fn, inputs=sd_img_input, outputs=[sd_img_output, sd_img_info])

                # 子Tab: 视频检测
                with gr.Tab("🎬 视频检测"):
                    gr.Markdown("上传道路视频，逐帧检测车辆并估算车速和车距。MOG2 会自动学习背景模型。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            sd_vid_input = gr.Video(label="📤 上传视频", height=350)
                            with gr.Row():
                                for vf in os.listdir(TEST_VIDEOS) if os.path.isdir(TEST_VIDEOS) else []:
                                    if vf.endswith('.mp4'):
                                        vpath = os.path.join(TEST_VIDEOS, vf)
                                        btn = gr.Button(vf.split('.')[0], size="sm", variant="secondary")
                                        btn.click(fn=lambda p=vpath: p, outputs=sd_vid_input)
                            sd_vid_btn = gr.Button("🎬 开始检测", variant="primary", size="lg")

                        with gr.Column(scale=1):
                            sd_vid_output = gr.Video(label="🎯 检测结果", height=350)
                            sd_vid_info = gr.Textbox(label="📊 统计信息", lines=10, interactive=False)

                    sd_vid_btn.click(fn=_sd_video_fn, inputs=sd_vid_input, outputs=[sd_vid_output, sd_vid_info])

        # ==================== Tab 8: 关于 ====================
        with gr.Tab("ℹ️ 关于", id="about"):
            gr.Markdown("""
            ## 🚗 OpenCV 智能视觉系统

            ### 📋 项目简介
            基于计算机视觉技术实现的智能视觉系统，集成车道线检测与手势识别功能，
            支持图片和视频输入，提供完整的检测Pipeline可视化。

            ### 🔬 技术架构
            | 模块 | 技术 |
            |------|------|
            | 前端界面 | Gradio Web框架 |
            | 图像处理 | OpenCV (色彩空间转换、边缘检测、形态学操作) |
            | 手势识别 | MediaPipe Hands (21关键点检测) |
            | 数值计算 | NumPy (多项式拟合、矩阵运算) |
            | 数据可视化 | Matplotlib |
            | 视频转码 | FFmpeg (H.264编码) |

            ### 📊 车道线检测 Pipeline
            1. **相机标定与去畸变** - 棋盘格角点检测 + 畸变系数计算
            2. **透视变换(鸟瞰图)** - 源点/目标点映射矩阵计算
            3. **多阈值融合** - HLS/LAB/HSV色彩空间分离 + Sobel梯度算子
            4. **直方图峰值检测** - 像素列统计 + 极值定位
            5. **滑动窗口多项式拟合** - 二阶多项式最小二乘拟合
            6. **曲率半径 & 偏移量** - 曲率公式计算 + 车道中心偏移
            7. **可视化叠加** - 逆透视变换 + 图像加权融合

            ### 🤚 手势识别功能
            | 功能 | 说明 |
            |------|------|
            | 手部关键点检测 | MediaPipe 21个关键点实时追踪 |
            | 手势识别 | 石头/剪刀/布、数字1-5、OK、竖大拇指 |
| 手指计数 | 实时统计竖起的手指数量 |
            | 虚拟画板 | 食指绘画、双指选择模式 |

            ### 🎨 支持的阈值方法
            | 方法 | 色彩空间 | 用途 |
            |------|----------|------|
            | Sobel X/Y | 灰度 | 垂直/水平边缘检测 |
            | Sobel 幅值 | 灰度 | 梯度强度筛选 |
            | Sobel 方向 | 灰度 | 边缘角度过滤 |
            | HLS S通道 | HLS | 抗阴影，检测黄/白色线 |
            | LAB B通道 | LAB | 黄色线专用检测 |
            | HSV白色掩码 | HSV | 白色线专用检测 |

            ### 🚀 使用方法
            ```bash
            # 安装依赖
            pip install -r requirements.txt

            # 启动Web界面
            python app.py

            # 或使用命令行
            python run.py
            ```
            """)

    return app


# ============================================================
# 启动入口
# ============================================================

if __name__ == '__main__':
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        inbrowser=True
    )
