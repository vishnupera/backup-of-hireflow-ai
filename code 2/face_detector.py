"""
Professional MediaPipe Face Detection + 468 Landmark Mesh Engine.
Computes real-time 3D head pose estimation (solvePnP), spatial metrics,
distance estimation, visibility analytics, and movement stability tracking.
Runs cleanly without drawing mesh or directional overlays onto the frame.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    Advanced Computer Vision face detector using MediaPipe Face Mesh.
    Calculates 3D head pose, spatial position, visibility, and movement stability in real time.
    """

    def __init__(self) -> None:
        # Access solutions safely through main mediapipe package
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

        # 3D Generic Facial Model Points for Head Pose Estimation (solvePnP)
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
                (0.0, -330.0, -65.0),     # Chin (Landmark 152)
                (-225.0, 170.0, -135.0),  # Left eye corner (Landmark 33)
                (225.0, 170.0, -135.0),   # Right eye corner (Landmark 263)
                (-150.0, -150.0, -125.0), # Left mouth corner (Landmark 61)
                (150.0, -150.0, -125.0),  # Right mouth corner (Landmark 291)
            ],
            dtype=np.float64,
        )

        self.pose_landmark_indices = [1, 152, 33, 263, 61, 291]
        self.pose_history: deque = deque(maxlen=30)
        self.prev_box: Optional[Tuple[int, int, int, int]] = None
        self.smooth_factor: float = 0.4
        self.last_telemetry: Dict[str, Any] = self._get_empty_telemetry()
        self.last_raw_landmarks: Any = None

    def process_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Processes frame for face detection, computes mesh & pose, and returns bounding box list.
        """
        if frame is None or frame.size == 0:
            self._reset_tracking()
            return []

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(rgb_frame)

        if not mesh_results.multi_face_landmarks:
            self._reset_tracking()
            return []

        face_landmarks = mesh_results.multi_face_landmarks[0]
        self.last_raw_landmarks = face_landmarks

        landmarks_2d: List[List[float]] = []
        landmarks_list: List[Dict[str, float]] = []
        x_coords: List[int] = []
        y_coords: List[int] = []

        for lm in face_landmarks.landmark:
            px, py = int(lm.x * w), int(lm.y * h)
            x_coords.append(px)
            y_coords.append(py)
            landmarks_2d.append([px, py])
            landmarks_list.append({"x": lm.x, "y": lm.y, "z": lm.z})

        # Bounding box calculation with exponential smoothing to reduce jitter
        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))
        box_w = x_max - x_min
        box_h = y_max - y_min

        if self.prev_box is not None:
            px, py, pw, ph = self.prev_box
            x_min = int(px * (1 - self.smooth_factor) + x_min * self.smooth_factor)
            y_min = int(py * (1 - self.smooth_factor) + y_min * self.smooth_factor)
            box_w = int(pw * (1 - self.smooth_factor) + box_w * self.smooth_factor)
            box_h = int(ph * (1 - self.smooth_factor) + box_h * self.smooth_factor)

        self.prev_box = (x_min, y_min, box_w, box_h)
        face_center = [int(x_min + box_w / 2), int(y_min + box_h / 2)]
        face_area = float(box_w * box_h)

        # Mathematical Metrics Computation
        face_position = self._calculate_face_position(face_center, w, h)
        inter_eye_dist = float(np.linalg.norm(np.array(landmarks_2d[33]) - np.array(landmarks_2d[263])))
        face_distance = self._estimate_face_distance(inter_eye_dist, w)
        visibility_pct = self._calculate_visibility(x_min, y_min, box_w, box_h, w, h, len(landmarks_2d))
        pitch, yaw, roll, nose_2d, nose_end_3d = self._calculate_head_pose(landmarks_2d, w, h)
        head_status = self._determine_head_status(pitch, yaw, roll)
        stability_score = self._calculate_stability(pitch, yaw, roll)

        self.last_telemetry = {
            "face_detected": True,
            "bounding_box": {"x": x_min, "y": y_min, "width": box_w, "height": box_h},
            "face_center": face_center,
            "face_area": face_area,
            "face_position": face_position,
            "face_distance": face_distance,
            "visibility_percentage": round(visibility_pct, 1),
            "head_pose": {"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)},
            "head_status": head_status,
            "head_stability": round(stability_score, 1),
            "landmarks": landmarks_list,
        }

        # Drawing overlays remain disabled so video frame stays clean
        return [(x_min, y_min, box_w, box_h)]

    def extract_landmarks(self, results: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Extracts the largest visible face box for legacy backward compatibility."""
        if results and len(results) > 0:
            return max(results, key=lambda b: b[2] * b[3])
        return None

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns computed spatial, distance, and pose metrics."""
        return self.last_telemetry

    def get_raw_landmarks(self) -> Any:
        """Returns raw MediaPipe face landmarks object for iris and eye tracking."""
        return self.last_raw_landmarks

    def _reset_tracking(self) -> None:
        """Resets tracking variables when face disappears."""
        self.prev_box = None
        self.pose_history.clear()
        self.last_telemetry = self._get_empty_telemetry()
        self.last_raw_landmarks = None

    def _get_empty_telemetry(self) -> Dict[str, Any]:
        """Returns baseline empty telemetry dictionary."""
        return {
            "face_detected": False,
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "face_center": [0, 0],
            "face_area": 0.0,
            "face_position": "Face Missing",
            "face_distance": "Unknown",
            "visibility_percentage": 0.0,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "head_status": "Looking Away",
            "head_stability": 0.0,
            "landmarks": [],
        }

    def _calculate_face_position(self, center: List[int], frame_w: int, frame_h: int) -> str:
        """Calculates face position relative to frame center."""
        cx, cy = center[0], center[1]
        horizontal = "Center"
        if cx < frame_w * 0.35:
            horizontal = "Left"
        elif cx > frame_w * 0.65:
            horizontal = "Right"

        vertical = ""
        if cy < frame_h * 0.35:
            vertical = "Top "
        elif cy > frame_h * 0.65:
            vertical = "Bottom "

        return "Center" if horizontal == "Center" and not vertical else f"{vertical}{horizontal}".strip()

    def _estimate_face_distance(self, inter_eye_px: float, frame_w: int) -> str:
        """Estimates candidate distance from camera based on eye width ratio."""
        ratio = inter_eye_px / float(frame_w)
        if ratio > 0.28:
            return "Too Close"
        elif ratio < 0.12:
            return "Too Far"
        return "Ideal"

    def _calculate_visibility(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int, count: int) -> float:
        """Calculates on-screen face visibility percentage."""
        if count < 468:
            return float((count / 468.0) * 50.0)
        out_x = max(0, -x) + max(0, (x + w) - frame_w)
        out_y = max(0, -y) + max(0, (y + h) - frame_h)
        truncated_area = (out_x * h) + (out_y * w)
        total_area = float(w * h)
        return 0.0 if total_area <= 0 else float(max(0.0, 1.0 - (truncated_area / total_area)) * 100.0)

    def _calculate_head_pose(self, landmarks_2d: List[List[float]], frame_w: int, frame_h: int) -> Tuple[float, float, float, Tuple[int, int], np.ndarray]:
        """Calculates Pitch, Yaw, and Roll Euler angles using OpenCV solvePnP."""
        image_points = np.array([landmarks_2d[idx] for idx in self.pose_landmark_indices], dtype=np.float64)
        focal_length = float(frame_w)
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0, (0, 0), np.zeros((3, 1))

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        nose_2d = (int(image_points[0][0]), int(image_points[0][1]))
        nose_end_point3D = np.array([[0.0, 0.0, 500.0]], dtype=np.float64)
        nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        return pitch, yaw, roll, nose_2d, p2

    def _determine_head_status(self, pitch: float, yaw: float, roll: float) -> str:
        """Determines orientation status string from pitch/yaw/roll values."""
        if pitch > 12.0:
            return "Looking Up"
        elif pitch < -12.0:
            return "Looking Down"
        elif yaw > 14.0:
            return "Looking Right"
        elif yaw < -14.0:
            return "Looking Left"
        elif roll > 12.0:
            return "Tilt Right"
        elif roll < -12.0:
            return "Tilt Left"
        return "Straight"

    def _calculate_stability(self, pitch: float, yaw: float, roll: float) -> float:
        """Calculates head movement stability score across consecutive frames."""
        self.pose_history.append((pitch, yaw, roll))
        if len(self.pose_history) < 5:
            return 95.0
        poses = np.array(self.pose_history)
        std_devs = np.std(poses, axis=0)
        total_variance = float(np.sum(std_devs))
        return float(max(10.0, min(100.0, 100.0 - (total_variance * 2.5))))