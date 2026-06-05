"""
手势识别模块 - 基于 cvzone/MediaPipe 的手部检测与手势识别
========================================
技术栈: cvzone (MediaPipe封装), OpenCV, NumPy

功能:
  1. 手部关键点检测 (21个关键点)
  2. 手势识别 (石头/剪刀/布/数字等)
  3. 手指计数
  4. 手部追踪可视化
  5. 虚拟画板 (手势控制)
"""

import cv2
import numpy as np
import math


class HandDetector:
    """
    手部检测器 - 基于 cvzone HandDetector
    """

    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        初始化手部检测器

        参数:
            static_mode: 静态图片模式 (True) 或视频流模式 (False)
            max_hands: 最大检测手数
            detection_confidence: 检测置信度阈值
            tracking_confidence: 追踪置信度阈值
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.detector = None
        self.hands = None
        self.landmark_list = []
        self.results = None
        self.tip_ids = [4, 8, 12, 16, 20]  # 指尖关键点ID
        self.mediapipe_available = False

        # 尝试初始化检测器
        self._init_detector(static_mode)

    def _init_detector(self, static_mode):
        """初始化检测器，尝试多种方式"""
        # 方式1: 使用 cvzone
        try:
            from cvzone.HandTrackingModule import HandDetector as CvzoneDetector
            self.detector = CvzoneDetector(
                staticMode=static_mode,
                maxHands=self.max_hands,
                detectionCon=self.detection_confidence,
                minTrackCon=self.tracking_confidence
            )
            self.mediapipe_available = True
            self.use_cvzone = True
            print("[INFO] 使用 cvzone HandDetector")
            return
        except Exception as e:
            print(f"[WARN] cvzone 初始化失败: {e}")

        # 方式2: 直接使用 mediapipe
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_mode,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence
            )
            self.mediapipe_available = True
            self.use_cvzone = False
            print("[INFO] 使用 MediaPipe Hands")
            return
        except Exception as e:
            print(f"[WARN] MediaPipe 初始化失败: {e}")

        # 方式3: 均不可用
        self.mediapipe_available = False
        self.use_cvzone = False
        print("[WARN] 手势识别不可用，请安装 cvzone 或 mediapipe")

    def find_hands(self, img, draw=True):
        """
        检测手部并绘制关键点

        参数:
            img: BGR图像
            draw: 是否绘制关键点和连接线

        返回:
            img: 绘制了关键点的图像
        """
        if not self.mediapipe_available:
            return img

        try:
            if self.use_cvzone:
                img, self.results = self.detector.findHands(img, draw=draw)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = self.hands.process(img_rgb)

                if self.results.multi_hand_landmarks:
                    for hand_lms in self.results.multi_hand_landmarks:
                        if draw:
                            self.mp_draw.draw_landmarks(
                                img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_styles.get_default_hand_landmarks_style(),
                                self.mp_styles.get_default_hand_connections_style()
                            )
        except Exception as e:
            print(f"[ERROR] find_hands: {e}")

        return img

    def find_position(self, img, hand_no=0, draw=True):
        """
        获取手部关键点坐标

        参数:
            img: BGR图像
            hand_no: 手的编号 (0=第一只手)
            draw: 是否绘制关键点

        返回:
            landmark_list: 关键点坐标列表 [(id, x, y), ...]
        """
        self.landmark_list = []

        if not self.mediapipe_available:
            return self.landmark_list

        try:
            if self.use_cvzone:
                if self.results is not None and isinstance(self.results, list) and len(self.results) > hand_no:
                    hand = self.results[hand_no]
                    lm_list = hand['lmList']
                    for i, lm in enumerate(lm_list):
                        self.landmark_list.append((i, lm[0], lm[1]))
                        if draw:
                            cv2.circle(img, (lm[0], lm[1]), 5, (255, 0, 255), cv2.FILLED)
            else:
                if self.results and self.results.multi_hand_landmarks:
                    if hand_no < len(self.results.multi_hand_landmarks):
                        hand = self.results.multi_hand_landmarks[hand_no]
                        h, w, c = img.shape

                        for id, lm in enumerate(hand.landmark):
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            self.landmark_list.append((id, cx, cy))

                            if draw:
                                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        except Exception as e:
            print(f"[ERROR] find_position: {e}")

        return self.landmark_list

    def fingers_up(self):
        """
        检测哪些手指竖起

        返回:
            fingers: 5个元素的列表，1=竖起，0=弯曲 [拇指,食指,中指,无名指,小指]
        """
        if len(self.landmark_list) == 0:
            return []

        fingers = []

        try:
            # 拇指 - 使用正确的拇指IP关节(landmark 2)，并根据手掌朝向判断左右手
            thumb_tip_x = self.landmark_list[self.tip_ids[0]][1]   # landmark 4
            thumb_ip_x = self.landmark_list[2][1]                  # landmark 2 (拇指IP)
            pinky_mcp_x = self.landmark_list[17][1]                # landmark 17 (小指根)
            wrist_x = self.landmark_list[0][1]                     # landmark 0 (手腕)

            # 判断手掌朝向：小指根在手腕右侧=右手掌，反之左手掌
            is_right_hand = pinky_mcp_x > wrist_x

            if is_right_hand:
                thumb_up = thumb_tip_x > thumb_ip_x
            else:
                thumb_up = thumb_tip_x < thumb_ip_x

            fingers.append(1 if thumb_up else 0)

            # 其他四指 - 比较指尖和关节的y坐标
            for id in range(1, 5):
                if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        except Exception as e:
            print(f"[ERROR] fingers_up: {e}")
            return []

        return fingers

    def count_fingers(self):
        """
        计算竖起的手指数量

        返回:
            count: 竖起的手指数量
        """
        fingers = self.fingers_up()
        return sum(fingers) if fingers else 0

    def recognize_gesture(self):
        """
        识别手势

        返回:
            gesture: 手势名称字符串
            description: 手势描述
        """
        fingers = self.fingers_up()
        if not fingers:
            return "None", "未检测到手"

        total = sum(fingers)

        # 石头 - 所有手指弯曲
        if total == 0:
            return "Rock", "石头 (拳头)"

        # 竖大拇指 - 只有拇指竖起
        elif total == 1 and fingers[0] == 1:
            return "ThumbsUp", "竖大拇指"

        # 数字1 - 食指竖起
        elif total == 1 and fingers[1] == 1:
            return "One", "数字 1"

        # 剪刀/数字2 - 食指和中指竖起
        elif total == 2 and fingers[1] == 1 and fingers[2] == 1:
            return "Scissors", "剪刀 (二)"

        # 数字3 - 食指、中指、无名指竖起
        elif total == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            return "Three", "数字 3"

        # 数字4 - 除拇指外四指竖起
        elif total == 4 and fingers[0] == 0:
            return "Four", "数字 4"

        # 布/数字5 - 所有手指竖起
        elif total == 5:
            return "Paper", "布 (张开手掌)"

        # OK手势 - 拇指和食指形成圆圈
        elif self._is_ok_gesture():
            return "OK", "OK 手势"

        # 其他
        else:
            return f"Fingers{total}", f"竖起 {total} 根手指"

    def _is_ok_gesture(self):
        """
        检测OK手势 (拇指和食指形成圆圈)
        """
        if len(self.landmark_list) < 21:
            return False

        # 拇指尖 (4) 和食指尖 (8) 的距离
        thumb_tip = self.landmark_list[4]
        index_tip = self.landmark_list[8]
        distance = math.sqrt(
            (thumb_tip[1] - index_tip[1]) ** 2 +
            (thumb_tip[2] - index_tip[2]) ** 2
        )

        # 使用手掌大小作为自适应阈值
        wrist = self.landmark_list[0]
        middle_mcp = self.landmark_list[9]
        palm_size = math.sqrt(
            (wrist[1] - middle_mcp[1]) ** 2 +
            (wrist[2] - middle_mcp[2]) ** 2
        )

        return distance < palm_size * 0.4

    def get_distance(self, p1, p2, img=None, draw=True):
        """
        计算两个关键点之间的距离

        参数:
            p1, p2: 关键点ID
            img: 图像 (可选，用于绘制)
            draw: 是否绘制距离线

        返回:
            distance: 两点间距离
            info: ((x1,y1), (x2,y2), (cx,cy))
        """
        if len(self.landmark_list) < max(p1, p2) + 1:
            return 0, None

        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if img is not None and draw:
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return distance, ((x1, y1), (x2, y2), (cx, cy))


class VirtualPainter:
    """
    虚拟画板 - 使用手势控制绘画
    """

    def __init__(self):
        self.brush_thickness = 15
        self.eraser_thickness = 50
        self.draw_color = (0, 0, 255)  # 红色
        self.canvas = None
        self.xp, self.yp = 0, 0
        self.drawing_mode = False

        # 颜色选择区域
        self.header_colors = [
            (0, 0, 255),    # 红色
            (255, 0, 0),    # 蓝色
            (0, 255, 0),    # 绿色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 0, 0),      # 橡皮擦 (黑色)
        ]

    def init_canvas(self, h, w):
        """初始化画布"""
        self.canvas = np.zeros((h, w, 3), np.uint8)

    def process_frame(self, img, landmarks, detector=None):
        """
        处理单帧图像，根据手势进行绘画

        参数:
            img: 输入图像
            landmarks: 手部关键点列表
            detector: HandDetector实例 (用于手指状态检测)

        返回:
            img: 处理后的图像
            mode: 当前模式 ("draw", "select", "idle")
        """
        if self.canvas is None:
            h, w, _ = img.shape
            self.init_canvas(h, w)

        if not landmarks or len(landmarks) < 21:
            self.xp, self.yp = 0, 0
            return img, "idle"

        # 获取食指尖和中指尖坐标
        x1, y1 = landmarks[8][1], landmarks[8][2]  # 食指尖
        x2, y2 = landmarks[12][1], landmarks[12][2]  # 中指尖

        # 检测手指状态
        if detector is None:
            detector = HandDetector()
        detector.landmark_list = landmarks
        fingers = detector.fingers_up()

        if not fingers:
            return img, "idle"

        # 选择模式 - 食指和中指都竖起
        if fingers[1] == 1 and fingers[2] == 1:
            self.xp, self.yp = 0, 0
            # 检查是否在颜色选择区域
            if y1 < 125:
                for i, color in enumerate(self.header_colors):
                    x_start = i * 100 + 50
                    x_end = (i + 1) * 100 + 50
                    if x_start < x1 < x_end:
                        if color == (0, 0, 0):
                            self.draw_color = (0, 0, 0)
                        else:
                            self.draw_color = color
            return img, "select"

        # 绘画模式 - 只有食指竖起
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 10, self.draw_color, cv2.FILLED)

            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x1, y1

            if self.draw_color == (0, 0, 0):
                cv2.line(img, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraser_thickness)
                cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.eraser_thickness)
            else:
                cv2.line(img, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)
                cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)

            self.xp, self.yp = x1, y1
            return img, "draw"

        self.xp, self.yp = 0, 0
        return img, "idle"

    def overlay_canvas(self, img):
        """将画布叠加到图像上"""
        try:
            if self.canvas is not None and isinstance(img, np.ndarray):
                img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, img_inv)
                img = cv2.bitwise_or(img, self.canvas)
        except Exception as e:
            print(f"[ERROR] overlay_canvas: {e}")
        return img

    def clear_canvas(self):
        """清空画布"""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)


def process_gesture_image(image):
    """
    处理单张图片的手势识别

    参数:
        image: BGR图像

    返回:
        result_img: 绘制了识别结果的图像
        info: 识别信息字典
    """
    detector = HandDetector(max_hands=2)
    img = image.copy()
    img = detector.find_hands(img)
    info = {"hands_detected": 0, "gestures": []}

    for hand_no in range(2):
        landmarks = detector.find_position(img, hand_no, draw=False)
        if landmarks:
            info["hands_detected"] += 1
            gesture, desc = detector.recognize_gesture()
            fingers = detector.fingers_up()
            count = detector.count_fingers()

            # 绘制信息
            y_offset = 30 + hand_no * 100
            cv2.putText(img, f"Hand {hand_no + 1}: {desc}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Fingers: {count} {fingers}", (10, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            info["gestures"].append({
                "hand": hand_no + 1,
                "gesture": gesture,
                "description": desc,
                "fingers": fingers,
                "count": count
            })

    return img, info


def process_gesture_video_frame(img, detector, painter=None):
    """
    处理视频帧的手势识别

    参数:
        img: BGR图像
        detector: HandDetector实例
        painter: VirtualPainter实例 (可选)

    返回:
        result_img: 处理后的图像
        gesture_info: 手势信息
    """
    img = detector.find_hands(img)
    gesture_info = {"mode": "idle", "gestures": []}

    for hand_no in range(2):
        landmarks = detector.find_position(img, hand_no, draw=False)
        if landmarks:
            gesture, desc = detector.recognize_gesture()
            fingers = detector.fingers_up()
            count = detector.count_fingers()

            # 绘制手部信息
            y_offset = 30 + hand_no * 80
            cv2.putText(img, f"Hand {hand_no + 1}: {desc}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Fingers: {count}", (10, y_offset + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            gesture_info["gestures"].append({
                "hand": hand_no + 1,
                "gesture": gesture,
                "description": desc,
                "count": count
            })

            # 虚拟画板模式
            if painter:
                img, mode = painter.process_frame(img, landmarks, detector)
                gesture_info["mode"] = mode

    # 叠加画布
    if painter:
        img = painter.overlay_canvas(img)

    return img, gesture_info
