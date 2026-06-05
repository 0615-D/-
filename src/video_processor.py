"""
视频处理模块 - 处理完整的视频文件
"""

import cv2
import numpy as np
import os
import time


def process_video_basic(input_path, output_path=None, show=True):
    """
    使用基础方法处理视频（直线检测 + Hough变换）
    """
    from src.basic_lane import detect_straight_lines, preprocess, extract_roi

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"处理视频: {os.path.basename(input_path)} ({w}x{h}, {fps}fps, {total}帧)")
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result, edges, lines = detect_straight_lines(frame)

        if writer:
            writer.write(result)
        if show:
            cv2.imshow('Basic Lane Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"  已处理 {frame_count}/{total} 帧 ({fps_actual:.1f} fps)")

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"完成！共处理 {frame_count} 帧，耗时 {elapsed:.1f}s")


def process_video_advanced(input_path, output_path=None, show=True):
    """
    使用高级算法处理视频（完整7阶段管线）
    """
    from src.advanced_lane import LaneDetector

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    detector = LaneDetector()

    # 尝试相机标定（如果有标定图片）
    cal_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'calibration_images')
    if os.path.isdir(cal_dir):
        detector.calibrate_camera(cal_dir)

    # 设置透视变换（用第一帧初始化）
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return
    detector.setup_perspective(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"高级管线处理: {os.path.basename(input_path)} ({w}x{h}, {fps}fps, {total}帧)")
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)

        if writer:
            writer.write(result)
        if show:
            cv2.imshow('Advanced Lane Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"  已处理 {frame_count}/{total} 帧 ({fps_actual:.1f} fps)")

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"完成！共处理 {frame_count} 帧，耗时 {elapsed:.1f}s")
