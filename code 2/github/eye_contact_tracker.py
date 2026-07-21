"""
MediaPipe Face Mesh Iris & Eye Tracking Engine.
Computes Iris Tracking, Eye Aspect Ratio (EAR), Blink Detection & Rate (per minute),
3D Gaze Direction, Eye Open Percentage, Eye Stability, and Eye Contact Percentage.
Runs silently without drawing visual iris landmarks on the frame.
"""

from collections import deque
import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np


class EyeContactTracker:
    """
    Advanced Eye Contact, Iris, Blink, and Gaze tracking module using MediaPipe Mesh Landmarks.
    """

    def __init__(self) -> None:
        # MediaPipe Landmark Indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

        self.EAR_THRESHOLD = 0.20  # Threshold under which eye is considered closed (blinking)
        self.blink_counter = 0
        self.is_blinking = False
        self.blink_timestamps: deque = deque()

        self.total_frames = 0
        self.eye_contact_frames = 0
        self.gaze_history: deque = deque(maxlen=30)

        self.last_metrics: Dict[str, Any] = self._get_empty_metrics()

    def check_contact(self, frame: np.ndarray, face_box: Any = None, raw_landmarks: Any = None) -> bool:
        """
        Main compatibility method called by main.py. Performs full gaze analysis silently.
        """
        self.total_frames += 1

        if raw_landmarks is None or not hasattr(raw_landmarks, "landmark"):
            self.last_metrics = self._get_empty_metrics()
            return False

        h, w, _ = frame.shape
        landmarks = raw_landmarks.landmark

        # Extract 2D Landmark points
        left_eye_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.LEFT_EYE])
        right_eye_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.RIGHT_EYE])
        left_iris_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.LEFT_IRIS])
        right_iris_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.RIGHT_IRIS])

        # 1. Eye Aspect Ratio (EAR)
        left_ear = self._calculate_ear(left_eye_pts)
        right_ear = self._calculate_ear(right_eye_pts)
        avg_ear = float((left_ear + right_ear) / 2.0)

        # 2. Blink Detection & Rate Engine
        current_time = time.time()
        if avg_ear < self.EAR_THRESHOLD:
            if not self.is_blinking:
                self.blink_counter += 1
                self.is_blinking = True
                self.blink_timestamps.append(current_time)
        else:
            self.is_blinking = False

        # Clean old timestamps (> 60s) for real-time Blink Rate (bpm)
        while self.blink_timestamps and (current_time - self.blink_timestamps[0]) > 60.0:
            self.blink_timestamps.popleft()
        blink_rate_bpm = len(self.blink_timestamps)

        # 3. Eye Open Percentage
        eye_open_pct = float(min(100.0, max(0.0, (avg_ear / 0.35) * 100.0)))

        # 4. Iris Center and Gaze Direction
        left_iris_center = np.mean(left_iris_pts, axis=0)
        right_iris_center = np.mean(right_iris_pts, axis=0)

        left_ratio_x, left_ratio_y = self._get_iris_ratios(left_iris_center, left_eye_pts)
        right_ratio_x, right_ratio_y = self._get_iris_ratios(right_iris_center, right_eye_pts)
        avg_ratio_x = (left_ratio_x + right_ratio_x) / 2.0
        avg_ratio_y = (left_ratio_y + right_ratio_y) / 2.0

        gaze_direction, is_contact = self._determine_gaze(avg_ratio_x, avg_ratio_y, avg_ear)

        if is_contact:
            self.eye_contact_frames += 1

        contact_pct = float((self.eye_contact_frames / self.total_frames) * 100.0)

        # 5. Eye Stability Score
        self.gaze_history.append((avg_ratio_x, avg_ratio_y))
        stability_score = self._calculate_gaze_stability()

        self.last_metrics = {
            "eye_contact": is_contact,
            "iris_tracking": {
                "left_iris": left_iris_center.tolist(),
                "right_iris": right_iris_center.tolist(),
            },
            "gaze_direction": gaze_direction,
            "blink_detected": self.is_blinking,
            "blink_count": self.blink_counter,
            "blink_rate_bpm": blink_rate_bpm,
            "ear": round(avg_ear, 3),
            "eye_open_percentage": round(eye_open_pct, 1),
            "looking_left": gaze_direction == "Looking Left",
            "looking_right": gaze_direction == "Looking Right",
            "looking_up": gaze_direction == "Looking Up",
            "looking_down": gaze_direction == "Looking Down",
            "looking_center": gaze_direction == "Looking Center",
            "eye_stability": round(stability_score, 1),
            "eye_contact_percentage": round(contact_pct, 1),
        }

        # Green iris tracking dots disabled to keep the camera view completely clean
        # for pt in np.vstack([left_iris_pts, right_iris_pts]):
        #     cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

        return is_contact

    def get_metrics(self) -> Dict[str, Any]:
        return self.last_metrics

    def _calculate_ear(self, eye_pts: np.ndarray) -> float:
        v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
        v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
        horiz = np.linalg.norm(eye_pts[0] - eye_pts[3])
        if horiz == 0:
            return 0.0
        return float((v1 + v2) / (2.0 * horiz))

    def _get_iris_ratios(self, iris_center: np.ndarray, eye_pts: np.ndarray) -> Tuple[float, float]:
        min_x = np.min(eye_pts[:, 0])
        max_x = np.max(eye_pts[:, 0])
        min_y = np.min(eye_pts[:, 1])
        max_y = np.max(eye_pts[:, 1])

        width = max_x - min_x
        height = max_y - min_y

        ratio_x = 0.5 if width == 0 else (iris_center[0] - min_x) / width
        ratio_y = 0.5 if height == 0 else (iris_center[1] - min_y) / height
        return float(ratio_x), float(ratio_y)

    def _determine_gaze(self, rx: float, ry: float, ear: float) -> Tuple[str, bool]:
        if ear < self.EAR_THRESHOLD:
            return "Blinking / Closed", False

        if rx < 0.38:
            return "Looking Right", False
        elif rx > 0.62:
            return "Looking Left", False
        elif ry < 0.35:
            return "Looking Up", False
        elif ry > 0.68:
            return "Looking Down", False

        return "Looking Center", True

    def _calculate_gaze_stability(self) -> float:
        if len(self.gaze_history) < 2:
            return 100.0
        data = np.array(self.gaze_history)
        std_dev = np.std(data, axis=0)
        variance = float(np.sum(std_dev))
        return float(max(0.0, 100.0 - (variance * 200.0)))

    def _get_empty_metrics(self) -> Dict[str, Any]:
        return {
            "eye_contact": False,
            "iris_tracking": {"left_iris": [0, 0], "right_iris": [0, 0]},
            "gaze_direction": "No Face Detected",
            "blink_detected": False,
            "blink_count": self.blink_counter,
            "blink_rate_bpm": 0,
            "ear": 0.0,
            "eye_open_percentage": 0.0,
            "looking_left": False,
            "looking_right": False,
            "looking_up": False,
            "looking_down": False,
            "looking_center": False,
            "eye_stability": 0.0,
            "eye_contact_percentage": 0.0,
        }