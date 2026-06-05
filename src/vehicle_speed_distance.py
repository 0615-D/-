"""
车速与车距估计模块（纯 OpenCV）
原理：
  - 车辆检测：MOG2 背景减除 → 形态学 → 轮廓 → 外接矩形
  - 车距估算：针孔相机模型 D = (W_real × f) / W_pixel
  - 车速估算：跨帧质心跟踪 → 位移 / 时间
"""

import cv2
import numpy as np
from collections import defaultdict


# ============================================================
# 常量
# ============================================================
# 典型车辆物理宽度 (米)
CAR_WIDTH = 1.8
TRUCK_WIDTH = 2.5

# 距离段颜色 (BGR)
COLOR_FAR   = (0, 200, 0)    # 绿色  >50m
COLOR_MID   = (0, 165, 255)  # 橙色  20~50m
COLOR_CLOSE = (0, 0, 255)    # 红色  <20m


def distance_color(d):
    if d < 0:
        return (200, 200, 200)
    if d < 20:
        return COLOR_CLOSE
    if d < 50:
        return COLOR_MID
    return COLOR_FAR


def focal_length_from_image(img_width):
    """估算焦距（像素），640 宽时≈576"""
    return img_width * 0.9


def estimate_distance(bbox_width_px, real_width_m, focal):
    """针孔模型: D = (W_real × f) / W_pixel"""
    if bbox_width_px <= 0:
        return -1
    return (real_width_m * focal) / bbox_width_px


def classify_vehicle(w, h):
    """根据宽高比粗略分类"""
    ratio = w / max(h, 1)
    if ratio > 1.8:
        return 'truck', TRUCK_WIDTH
    return 'car', CAR_WIDTH


# ============================================================
# 质心跟踪器
# ============================================================
class CentroidTracker:
    """
    跨帧 IoU + 质心距离跟踪器。
    为每个目标维护历史质心序列，用于速度估算。
    """
    def __init__(self, max_dist=80, max_age=20):
        self.next_id = 0
        self.tracks = {}       # id → {'bbox','centers':[(cx,cy,ts),...],'age':0}
        self.max_dist = max_dist
        self.max_age = max_age

    def update(self, detections, timestamp=None):
        """
        detections: list of [x1, y1, x2, y2]
        返回: list of (track_id, bbox)
        """
        det_used = set()
        result = []

        # 匹配已有 track
        for tid, trk in list(self.tracks.items()):
            tcx = (trk['bbox'][0] + trk['bbox'][2]) / 2
            tcy = (trk['bbox'][1] + trk['bbox'][3]) / 2
            best_dist = self.max_dist
            best_j = -1
            for j, det in enumerate(detections):
                if j in det_used:
                    continue
                dcx = (det[0] + det[2]) / 2
                dcy = (det[1] + det[3]) / 2
                dist = np.hypot(dcx - tcx, dcy - tcy)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0:
                det_used.add(best_j)
                new_box = detections[best_j]
                cx = (new_box[0] + new_box[2]) / 2
                cy = (new_box[1] + new_box[3]) / 2
                trk['bbox'] = new_box
                trk['age'] = 0
                trk['centers'].append((cx, cy, timestamp))
                result.append((tid, new_box))
            else:
                trk['age'] += 1

        # 新检测 → 新 track
        for j, det in enumerate(detections):
            if j not in det_used:
                cx = (det[0] + det[2]) / 2
                cy = (det[1] + det[3]) / 2
                self.tracks[self.next_id] = {
                    'bbox': det,
                    'age': 0,
                    'centers': [(cx, cy, timestamp)],
                }
                result.append((self.next_id, det))
                self.next_id += 1

        # 删除过期
        for tid in [k for k, v in self.tracks.items() if v['age'] > self.max_age]:
            del self.tracks[tid]

        return result


# ============================================================
# 车辆检测（MOG2 背景减除 + 轮廓）
# ============================================================
class VehicleDetector:
    """
    纯 OpenCV 车辆检测：
    1. MOG2 背景减除提取运动前景
    2. 形态学开运算去噪
    3. 轮廓检测 → 外接矩形
    4. 面积 / 宽高比过滤
    """
    def __init__(self, min_area=1500, max_area_ratio=0.35):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 7))

    def detect(self, frame):
        """返回 list of [x1,y1,x2,y2]"""
        h, w = frame.shape[:2]
        frame_area = h * w

        fg_mask = self.bg_subtractor.apply(frame)
        # 去阴影 (阴影值=127)
        fg_mask[fg_mask == 127] = 0
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > frame_area * self.max_area_ratio:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 宽高比过滤
            ratio = bw / max(bh, 1)
            if ratio < 0.3 or ratio > 5.0:
                continue
            # 底部不在画面最上方 1/3（排除远处噪声）
            if y + bh < h * 0.25:
                continue
            bboxes.append([x, y, x + bw, y + bh])

        # NMS
        bboxes = _nms(bboxes, 0.4)
        return bboxes


def _nms(bboxes, iou_thresh):
    """简单 NMS"""
    if not bboxes:
        return []
    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for box in bboxes:
        suppress = False
        for kept in keep:
            iou_val = _iou(box, kept)
            if iou_val > iou_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(box)
    return keep


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


# ============================================================
# 单帧处理
# ============================================================
def process_frame(frame, detector, tracker, frame_idx=0, fps=30):
    """
    单帧处理: 检测 → 距离 → 跟踪 → 速度 → 绘图
    返回: annotated_image, list[dict]
    """
    h, w = frame.shape[:2]
    focal = focal_length_from_image(w)
    ts = frame_idx / fps if fps > 0 else frame_idx

    lt = max(1, int(min(w, h) / 400) + 1)
    fs = max(0.35, min(w, h) / 1600)
    tt = max(1, lt - 1)

    # 检测
    bboxes = detector.detect(frame)
    # 跟踪
    tracked = tracker.update(bboxes, ts)

    output = frame.copy()
    info_list = []

    for tid, bbox in tracked:
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        cls_name, real_w = classify_vehicle(box_w, box_h)

        dist = estimate_distance(box_w, real_w, focal)

        # 速度
        speed_kmh = 0.0
        trk = tracker.tracks.get(tid)
        if trk and len(trk['centers']) >= 3:
            n = min(8, len(trk['centers']))
            cx0, cy0, t0 = trk['centers'][-n]
            cx1, cy1, t1 = trk['centers'][-1]
            dt = t1 - t0
            if dt > 0 and dist > 0:
                # 纵向运动：通过框高变化估算距离变化
                # 简化：用质心 y 偏移 × 距离比例因子
                meters_per_pixel = dist / focal
                dy = cy1 - cy0
                longitudinal_m = abs(dy) * meters_per_pixel
                raw_speed = longitudinal_m / dt  # m/s
                raw_speed *= 3.6  # km/h
                # 接近(y增大=物体变大=靠近) 或 远离
                if cy1 > cy0:
                    raw_speed = -raw_speed  # 负=接近自身车辆
                speed_kmh = max(-200, min(200, raw_speed))

        # 画框
        color = distance_color(dist)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, lt)

        # 标签
        label = f"{cls_name}"
        if dist > 0:
            label += f" | {dist:.1f}m"
        if speed_kmh < -2:
            label += f" | 接近 {-speed_kmh:.0f}km/h"
        elif speed_kmh > 2:
            label += f" | 远离 {speed_kmh:.0f}km/h"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
        cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), tt, cv2.LINE_AA)

        info_list.append({
            'track_id': tid,
            'class': cls_name,
            'bbox': bbox,
            'distance_m': round(dist, 2) if dist > 0 else None,
            'speed_kmh': round(speed_kmh, 1),
            'direction': 'approaching' if speed_kmh < -2 else ('leaving' if speed_kmh > 2 else 'static'),
        })

    # 底栏
    bar_text = f"检测到 {len(tracked)} 个目标 | 焦距={focal:.0f}px | 方法=MOG2背景减除"
    cv2.rectangle(output, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(output, bar_text, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, (255, 255, 255), tt, cv2.LINE_AA)

    # 距离图例
    legend_y = 30
    for txt, clr in [("近距离 <20m (危险)", COLOR_CLOSE),
                      ("中距离 20~50m", COLOR_MID),
                      ("远距离 >50m", COLOR_FAR)]:
        cv2.rectangle(output, (w - 220, legend_y - 12), (w - 200, legend_y + 2), clr, -1)
        cv2.putText(output, txt, (w - 195, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 22

    return output, info_list


# ============================================================
# 图片处理（单帧，无跟踪/速度）
# ============================================================
def detect_speed_distance_image(image):
    """单张图片：只做距离估算（无跟踪，无法测速）"""
    h, w = image.shape[:2]
    focal = focal_length_from_image(w)
    lt = max(1, int(min(w, h) / 400) + 1)
    fs = max(0.35, min(w, h) / 1600)
    tt = max(1, lt - 1)

    detector = VehicleDetector()
    bboxes = detector.detect(image)

    output = image.copy()
    info_list = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        cls_name, real_w = classify_vehicle(box_w, box_h)
        dist = estimate_distance(box_w, real_w, focal)

        color = distance_color(dist)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, lt)

        label = f"{cls_name} | {dist:.1f}m" if dist > 0 else cls_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
        cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), tt, cv2.LINE_AA)

        info_list.append({
            'class': cls_name,
            'bbox': bbox,
            'distance_m': round(dist, 2) if dist > 0 else None,
        })

    # 图例
    legend_y = 30
    for txt, clr in [("近距离 <20m (危险)", COLOR_CLOSE),
                      ("中距离 20~50m", COLOR_MID),
                      ("远距离 >50m", COLOR_FAR)]:
        cv2.rectangle(output, (w - 220, legend_y - 12), (w - 200, legend_y + 2), clr, -1)
        cv2.putText(output, txt, (w - 195, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 22

    return output, info_list


# ============================================================
# 视频处理
# ============================================================
def detect_speed_distance_video(video_path, output_path=None, progress_cb=None):
    """
    逐帧处理视频，返回输出路径和每帧检测信息。
    """
    import tempfile, os

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), 'speed_distance_output.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    detector = VehicleDetector()
    tracker = CentroidTracker(max_dist=80, max_age=20)
    all_info = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated, info = process_frame(frame, detector, tracker, idx, fps)
        writer.write(annotated)
        all_info.append({'frame': idx, 'detections': info})
        idx += 1
        if progress_cb:
            progress_cb(idx, total)

    cap.release()
    writer.release()
    return output_path, all_info


def analyze_detections(all_info):
    """汇总统计"""
    dists, speeds = [], []
    for fd in all_info:
        for d in fd['detections']:
            if d.get('distance_m'):
                dists.append(d['distance_m'])
            if d.get('speed_kmh') is not None:
                speeds.append(d['speed_kmh'])
    stats = {
        'total_frames': len(all_info),
        'total_detections': sum(len(fd['detections']) for fd in all_info),
    }
    if dists:
        stats['distance_min'] = round(min(dists), 1)
        stats['distance_max'] = round(max(dists), 1)
        stats['distance_avg'] = round(sum(dists) / len(dists), 1)
    if speeds:
        stats['speed_min'] = round(min(speeds), 1)
        stats['speed_max'] = round(max(speeds), 1)
    return stats
