"""
可视化模块 - 各处理阶段的可视化展示
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端


def visualize_all_stages(image_path, output_path=None):
    """
    展示一张图片经过所有处理阶段的结果
    用于论文报告、实验演示
    """
    from src.basic_lane import (
        preprocess, extract_roi, color_threshold, color_threshold_advanced,
        perspective_transform, sliding_window_detection
    )
    from src.advanced_lane import LaneDetector

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- 各阶段处理 ---
    # 1. 原图
    stage1 = img_rgb.copy()

    # 2. 灰度 + 高斯模糊
    gray = preprocess(img)
    stage2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 3. ROI 提取
    roi, mask = extract_roi(img)
    stage3 = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # 4. Canny 边缘
    edges = cv2.Canny(gray, 50, 150)
    roi_edges, _ = extract_roi(edges)
    stage4 = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2RGB)

    # 5. 基础颜色阈值
    binary_basic = color_threshold(img)
    stage5 = cv2.cvtColor(binary_basic, cv2.COLOR_GRAY2RGB)

    # 6. 高级多阈值融合
    binary_adv = color_threshold_advanced(img)
    stage6 = cv2.cvtColor(binary_adv, cv2.COLOR_GRAY2RGB)

    # 7. 透视变换(鸟瞰图)
    warped, _, _ = perspective_transform(binary_adv)
    stage7 = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)

    # 8. 滑动窗口可视化
    left_fit, right_fit, l_inds, r_inds = sliding_window_detection(warped, nwindows=9, margin=80)
    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        stage8 = np.dstack((warped, warped, warped)).astype(np.uint8) * 255
        stage8[nonzeroy[l_inds], nonzerox[l_inds]] = [255, 0, 0]
        stage8[nonzeroy[r_inds], nonzerox[r_inds]] = [0, 0, 255]

        # 绘制拟合曲线
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(stage8, pts_left, False, (255, 255, 0), 2)
        cv2.polylines(stage8, pts_right, False, (255, 255, 0), 2)
    else:
        stage8 = np.dstack((warped, warped, warped)).astype(np.uint8) * 255

    # --- 绘制2x4网格 ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('OpenCV Lane Detection - Pipeline Stages', fontsize=16, fontweight='bold')

    stages = [stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8]
    titles = [
        '1. Original', '2. Grayscale + Blur',
        '3. ROI Extraction', '4. Canny Edge',
        '5. Basic Threshold', '6. Advanced Multi-Threshold',
        '7. Perspective Transform', '8. Sliding Window + Fit'
    ]

    for ax, stage, title in zip(axes.flat, stages, titles):
        ax.imshow(stage)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果保存到: {output_path}")
    plt.close()


def visualize_perspective_points(image_path, output_path=None):
    """
    可视化透视变换的源点和目标点
    """
    from src.basic_lane import perspective_transform

    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    src = np.float32([
        [int(w * 0.45), int(h * 0.63)],
        [int(w * 0.55), int(h * 0.63)],
        [int(w * 0.90), int(h * 0.95)],
        [int(w * 0.10), int(h * 0.95)]
    ])

    img_show = img.copy()
    # 绘制四边形
    pts = src.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img_show, [pts], True, (0, 255, 0), 2)
    # 绘制角点
    for i, pt in enumerate(src):
        cv2.circle(img_show, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
        cv2.putText(img_show, f'S{i}', (int(pt[0]) + 10, int(pt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    warped, _, _ = perspective_transform(img)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Source Points (Trapezoid)')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Warped (Bird\'s Eye View)')
    axes[1].axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"透视变换可视化保存到: {output_path}")
    plt.close()


def visualize_threshold_comparison(image_path, output_path=None):
    """
    对比不同阈值方法的效果
    """
    from src.advanced_lane import LaneDetector

    img = cv2.imread(image_path)
    if img is None:
        return

    det = LaneDetector()

    gradx = det.abs_sobel_thresh(img, 'x', 3, (20, 100))
    grady = det.abs_sobel_thresh(img, 'y', 3, (20, 100))
    mag = det.mag_thresh(img, 3, (30, 100))
    direction = det.dir_threshold(img, 15, (0.7, 1.3))
    s_channel = det.hls_s_channel(img, (170, 255))
    b_channel = det.lab_b_channel(img, (155, 200))
    white = det.hsv_white_mask(img)
    combined = det.combined_threshold(img)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Threshold Methods Comparison', fontsize=16, fontweight='bold')

    results = [gradx, grady, mag, direction, s_channel, b_channel, white, combined]
    titles = ['Sobel X', 'Sobel Y', 'Magnitude', 'Direction',
              'HLS S-channel', 'LAB B-channel', 'HSV White', 'Combined']

    for ax, result, title in zip(axes.flat, results, titles):
        ax.imshow(result, cmap='gray')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"阈值对比保存到: {output_path}")
    plt.close()


def visualize_sliding_window_process(image_path, output_path=None):
    """
    可视化滑动窗口搜索过程
    """
    from src.basic_lane import (
        color_threshold_advanced, perspective_transform, sliding_window_detection
    )

    img = cv2.imread(image_path)
    if img is None:
        return

    binary = color_threshold_advanced(img)
    warped, _, _ = perspective_transform(binary)

    left_fit, right_fit, l_inds, r_inds = sliding_window_detection(warped, nwindows=9, margin=80)

    # 创建可视化
    out_img = np.dstack((warped, warped, warped)).astype(np.uint8) * 255
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if l_inds is not None and len(l_inds) > 0:
        out_img[nonzeroy[l_inds], nonzerox[l_inds]] = [255, 0, 0]
    if r_inds is not None and len(r_inds) > 0:
        out_img[nonzeroy[r_inds], nonzerox[r_inds]] = [0, 0, 255]

    if left_fit is not None and right_fit is not None:
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # 绘制滑动窗口
        histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        nwindows = 9
        window_height = warped.shape[0] // nwindows
        margin = 80

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        lx, rx = leftx_base, rightx_base

        for win in range(nwindows):
            y_low = warped.shape[0] - (win + 1) * window_height
            y_high = warped.shape[0] - win * window_height
            cv2.rectangle(out_img, (lx - margin, y_low), (lx + margin, y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (rx - margin, y_low), (rx + margin, y_high), (0, 255, 0), 2)
            good_l = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                      (nonzerox >= lx - margin) & (nonzerox < lx + margin)).nonzero()[0]
            good_r = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                      (nonzerox >= rx - margin) & (nonzerox < rx + margin)).nonzero()[0]
            if len(good_l) > 50:
                lx = int(np.mean(nonzerox[good_l]))
            if len(good_r) > 50:
                rx = int(np.mean(nonzerox[good_r]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(out_img)
    axes[1].set_title('Sliding Window Detection (Blue=Left, Red=Right, Green=Windows)')
    axes[1].axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"滑动窗口可视化保存到: {output_path}")
    plt.close()


def draw_histogram(binary_warped, output_path=None):
    """
    绘制直方图 - 显示底部像素分布
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(binary_warped, cmap='gray')
    axes[0].set_title('Binary Warped Image')
    axes[0].axis('off')

    axes[1].plot(histogram)
    axes[1].set_title('Histogram of Bottom Half')
    axes[1].set_xlabel('Pixel Column')
    axes[1].set_ylabel('Pixel Count')

    midpoint = histogram.shape[0] // 2
    left_peak = np.argmax(histogram[:midpoint])
    right_peak = np.argmax(histogram[midpoint:]) + midpoint
    axes[1].axvline(x=left_peak, color='r', linestyle='--', label=f'Left Peak: {left_peak}')
    axes[1].axvline(x=right_peak, color='b', linestyle='--', label=f'Right Peak: {right_peak}')
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"直方图保存到: {output_path}")
    plt.close()
