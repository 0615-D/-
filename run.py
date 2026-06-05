"""
OpenCV 车道线检测系统 - 命令行主程序
========================================
基于计算机视觉技术的车道线检测系统

功能模块:
  Task 1: 视频帧提取
  Task 2: ROI感兴趣区域提取
  Task 3: 直线车道线检测 (Canny + Hough变换)
  Task 4: 弯道车道线检测 (颜色阈值 + 滑动窗口)
  Advanced: 完整7阶段管线 (标定+透视+多阈值+多项式拟合+曲率+偏移+可视化)
  Visualize: 各阶段可视化输出

使用方法:
  python run.py                    # 运行完整演示
  python run.py --task 1           # 只运行 Task 1
  python run.py --task 2           # 只运行 Task 2
  python run.py --task 3           # 只运行 Task 3
  python run.py --task 4           # 只运行 Task 4
  python run.py --task advanced    # 运行高级管线(图片)
  python run.py --task video_basic # 基础方法处理视频
  python run.py --task video_adv   # 高级方法处理视频
  python run.py --task visualize   # 生成各阶段可视化
"""

import argparse
import cv2
import numpy as np
import os
import sys
import glob

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES = os.path.join(ROOT_DIR, 'test_images')
TEST_VIDEOS = os.path.join(ROOT_DIR, 'test_videos')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')


def run_task1():
    """Task 1: 从视频中提取帧"""
    from src.basic_lane import extract_frames

    print("=" * 60)
    print("Task 1: 视频帧提取")
    print("=" * 60)

    video_path = os.path.join(TEST_VIDEOS, 'solidWhiteRight.mp4')
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    output_dir = os.path.join(OUTPUT_DIR, 'task1_frames')
    count = extract_frames(video_path, output_dir, max_frames=50)
    print(f"\n结果: 共提取 {count} 帧到 {output_dir}")

    # 显示提取结果样例
    sample_files = sorted(glob.glob(os.path.join(output_dir, '*.jpg')))
    if sample_files:
        print(f"样例文件: {sample_files[0]}")
    return count


def run_task2():
    """Task 2: ROI区域提取"""
    from src.basic_lane import auto_roi

    print("\n" + "=" * 60)
    print("Task 2: ROI感兴趣区域提取")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, 'task2_roi')
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(TEST_IMAGES, '*.jpg')))
    if not images:
        print("未找到测试图片")
        return

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        roi, mask = auto_roi(img)

        name = os.path.splitext(os.path.basename(img_path))[0]
        roi_path = os.path.join(output_dir, f'{name}_roi.jpg')
        mask_path = os.path.join(output_dir, f'{name}_mask.jpg')
        cv2.imwrite(roi_path, roi)
        cv2.imwrite(mask_path, mask)
        print(f"  {name}: ROI -> {roi_path}")

    print(f"\n结果: ROI图片保存到 {output_dir}")


def run_task3():
    """Task 3: 直线车道线检测"""
    from src.basic_lane import detect_straight_lines

    print("\n" + "=" * 60)
    print("Task 3: 直线车道线检测 (Canny + Hough)")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, 'task3_straight')
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(TEST_IMAGES, '*.jpg')))
    if not images:
        print("未找到测试图片")
        return

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        result, edges, lines = detect_straight_lines(img)

        name = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(output_dir, f'{name}_result.jpg'), result)
        cv2.imwrite(os.path.join(output_dir, f'{name}_edges.jpg'), edges)
        print(f"  {name}: 检测到 {len(lines)} 条车道线")

    print(f"\n结果: 检测结果保存到 {output_dir}")


def run_task4():
    """Task 4: 弯道车道线检测"""
    from src.basic_lane import (
        color_threshold, color_threshold_advanced, perspective_transform,
        sliding_window_detection
    )

    print("\n" + "=" * 60)
    print("Task 4: 弯道车道线检测 (颜色阈值 + 滑动窗口)")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, 'task4_curved')
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(TEST_IMAGES, '*.jpg')))
    if not images:
        print("未找到测试图片")
        return

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        name = os.path.splitext(os.path.basename(img_path))[0]

        # 基础阈值
        binary_basic = color_threshold(img)
        # 高级阈值
        binary_adv = color_threshold_advanced(img)
        # 透视变换
        warped, M, Minv = perspective_transform(binary_adv)
        # 滑动窗口
        left_fit, right_fit, l_inds, r_inds = sliding_window_detection(warped)

        cv2.imwrite(os.path.join(output_dir, f'{name}_binary_basic.jpg'), binary_basic)
        cv2.imwrite(os.path.join(output_dir, f'{name}_binary_adv.jpg'), binary_adv)
        cv2.imwrite(os.path.join(output_dir, f'{name}_warped.jpg'), warped)

        status = "检测成功" if (left_fit is not None and right_fit is not None) else "检测失败"
        print(f"  {name}: {status}")

        # 在原图上绘制结果
        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            warp_zero = np.zeros_like(warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
            overlay = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
            cv2.imwrite(os.path.join(output_dir, f'{name}_overlay.jpg'), overlay)

    print(f"\n结果: 弯道检测结果保存到 {output_dir}")


def run_advanced():
    """Advanced: 完整7阶段管线处理图片"""
    from src.advanced_lane import LaneDetector

    print("\n" + "=" * 60)
    print("Advanced: 完整7阶段高级管线")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, 'advanced')
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(TEST_IMAGES, '*.jpg')))
    if not images:
        print("未找到测试图片")
        return

    detector = LaneDetector()

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        name = os.path.splitext(os.path.basename(img_path))[0]

        # 初始化透视变换
        detector.setup_perspective(img)

        # 处理
        undist = detector.undistort(img)
        warped = detector.warp_perspective(undist)
        binary = detector.combined_threshold(warped)

        left_fit, right_fit, l_inds, r_inds = detector.sliding_window(binary)

        if left_fit is not None and right_fit is not None:
            if detector.sanity_check(left_fit, right_fit, binary):
                left_fit, right_fit = detector.smooth_fits(left_fit, right_fit)
                curvature = detector.measure_curvature(binary, left_fit, right_fit)
                offset = detector.measure_offset(binary, left_fit, right_fit)
                result = detector.draw_lane(undist, binary, left_fit, right_fit, curvature, offset)

                print(f"  {name}: 曲率={curvature:.0f}m, 偏移={offset:.2f}m")
                cv2.imwrite(os.path.join(output_dir, f'{name}_result.jpg'), result)

                # 保存中间步骤
                cv2.imwrite(os.path.join(output_dir, f'{name}_binary.jpg'),
                            binary * 255)
                cv2.imwrite(os.path.join(output_dir, f'{name}_warped.jpg'),
                            warped)
            else:
                print(f"  {name}: 完整性校验未通过")
        else:
            print(f"  {name}: 未检测到车道线")

    print(f"\n结果: 高级管线结果保存到 {output_dir}")


def run_video_basic():
    """基础方法处理视频"""
    from src.video_processor import process_video_basic

    print("\n" + "=" * 60)
    print("视频处理: 基础方法")
    print("=" * 60)

    video_path = os.path.join(TEST_VIDEOS, 'solidWhiteRight.mp4')
    output_path = os.path.join(OUTPUT_DIR, 'video_basic_output.mp4')

    if not os.path.exists(video_path):
        print(f"视频不存在: {video_path}")
        return

    process_video_basic(video_path, output_path, show=False)


def run_video_advanced():
    """高级方法处理视频"""
    from src.video_processor import process_video_advanced

    print("\n" + "=" * 60)
    print("视频处理: 高级管线")
    print("=" * 60)

    video_path = os.path.join(TEST_VIDEOS, 'solidWhiteRight.mp4')
    output_path = os.path.join(OUTPUT_DIR, 'video_advanced_output.mp4')

    if not os.path.exists(video_path):
        print(f"视频不存在: {video_path}")
        return

    process_video_advanced(video_path, output_path, show=False)


def run_visualize():
    """生成各阶段可视化图"""
    from src.visualization import (
        visualize_all_stages, visualize_perspective_points,
        visualize_threshold_comparison, visualize_sliding_window_process
    )

    print("\n" + "=" * 60)
    print("可视化: 生成各阶段处理效果图")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    # 选择一张代表性图片
    img_path = os.path.join(TEST_IMAGES, 'solidYellowCurve.jpg')
    if not os.path.exists(img_path):
        images = sorted(glob.glob(os.path.join(TEST_IMAGES, '*.jpg')))
        if images:
            img_path = images[0]
        else:
            print("未找到测试图片")
            return

    print(f"使用图片: {os.path.basename(img_path)}")

    visualize_all_stages(img_path, os.path.join(output_dir, 'pipeline_stages.jpg'))
    visualize_perspective_points(img_path, os.path.join(output_dir, 'perspective_transform.jpg'))
    visualize_threshold_comparison(img_path, os.path.join(output_dir, 'threshold_comparison.jpg'))
    visualize_sliding_window_process(img_path, os.path.join(output_dir, 'sliding_window.jpg'))

    print(f"\n结果: 可视化图片保存到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='OpenCV 车道线检测系统')
    parser.add_argument('--task', type=str, default='all',
                        choices=['1', '2', '3', '4', 'advanced',
                                 'video_basic', 'video_adv', 'visualize', 'all'],
                        help='要运行的任务 (默认: all)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  OpenCV 车道线检测系统")
    print("  基于计算机视觉技术的车道线检测")
    print("=" * 60)

    task = args.task

    if task in ('1', 'all'):
        run_task1()
    if task in ('2', 'all'):
        run_task2()
    if task in ('3', 'all'):
        run_task3()
    if task in ('4', 'all'):
        run_task4()
    if task in ('advanced', 'all'):
        run_advanced()
    if task in ('video_basic', 'all'):
        run_video_basic()
    if task in ('video_adv', 'all'):
        run_video_advanced()
    if task in ('visualize', 'all'):
        run_visualize()

    print("\n" + "=" * 60)
    print("  所有任务完成！结果保存在 output/ 目录")
    print("=" * 60)


if __name__ == '__main__':
    main()
