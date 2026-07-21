"""
Real-Time Deep Multi-Class Emotion Recognition Engine.
Integrates DeepFace/FER with geometric facial mesh micro-expression fallback.
Computes Dominant Emotion, Confidence, Real-time Emotion Timeline, Stability, and Distribution.
"""

from collections import Counter, deque
from typing import Any, Dict, Tuple

import cv2
import numpy as np

DEEP_EMOTION_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEP_EMOTION_AVAILABLE = True
except ImportError:
    pass


class EmotionAnalysis:
    """
    Advanced Emotion Detection module providing full multi-class emotion distributions.
    """

    def __init__(self) -> None:
        self.emotions_list = ["Happy", "Neutral", "Sad", "Angry", "Fear", "Surprise", "Disgust"]
        self.emotion_history: deque = deque(maxlen=300)
        self.emotion_counts: Counter = Counter()

        self.last_metrics: Dict[str, Any] = self._get_empty_metrics()
        self.frame_skip = 5
        self.frame_counter = 0

        self.cached_emotion = "Neutral"
        self.cached_confidence = 85.0
        self.cached_dist = {e: (100.0 if e == "Neutral" else 0.0) for e in self.emotions_list}

    def analyze_emotion(self, face_box: Any = None, confidence_score: float = 80.0, frame: np.ndarray = None, raw_landmarks: Any = None) -> str:
        """
        Main interface method compatible with main.py pipeline.
        """
        self.frame_counter += 1

        if face_box is None and raw_landmarks is None:
            self._update_metrics("Neutral", 50.0, self.cached_dist)
            return "Neutral"

        if DEEP_EMOTION_AVAILABLE and frame is not None and (self.frame_counter % self.frame_skip == 0):
            try:
                x, y, w, h = face_box if face_box else (0, 0, frame.shape[1], frame.shape[0])
                face_crop = frame[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)]

                if face_crop.size > 0:
                    results = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                    res = results[0] if isinstance(results, list) else results
                    dominant = res['dominant_emotion'].capitalize()

                    raw_dist = res['emotion']
                    total_sum = sum(raw_dist.values()) or 1.0
                    normalized_dist = {k.capitalize(): float((v / total_sum) * 100.0) for k, v in raw_dist.items()}

                    self.cached_emotion = dominant
                    self.cached_confidence = float(normalized_dist.get(dominant, 80.0))
                    self.cached_dist = normalized_dist
            except Exception:
                pass

        if not DEEP_EMOTION_AVAILABLE or (self.frame_counter % self.frame_skip != 0):
            if raw_landmarks is not None:
                geom_emotion, geom_conf, geom_dist = self._analyze_mesh_geometry(raw_landmarks)
                self.cached_emotion = geom_emotion
                self.cached_confidence = geom_conf
                self.cached_dist = geom_dist

        self._update_metrics(self.cached_emotion, self.cached_confidence, self.cached_dist)
        return self.cached_emotion

    def get_metrics(self) -> Dict[str, Any]:
        return self.last_metrics

    def _analyze_mesh_geometry(self, raw_landmarks: Any) -> Tuple[str, float, Dict[str, float]]:
        lm = raw_landmarks.landmark

        lip_top = np.array([lm[13].x, lm[13].y])
        lip_bot = np.array([lm[14].x, lm[14].y])
        corner_left = np.array([lm[61].x, lm[61].y])
        corner_right = np.array([lm[291].x, lm[291].y])

        mouth_height = np.linalg.norm(lip_top - lip_bot)
        mouth_width = np.linalg.norm(corner_left - corner_right)
        mouth_ratio = mouth_height / (mouth_width or 1.0)

        brow_left = np.array([lm[70].x, lm[70].y])
        brow_right = np.array([lm[300].x, lm[300].y])
        brow_dist = np.linalg.norm(brow_left - brow_right)

        dist = {e: 5.0 for e in self.emotions_list}

        if mouth_ratio > 0.4:
            emotion = "Surprise"
            dist["Surprise"] = 80.0
            confidence = 88.0
        elif mouth_width > 0.38 and mouth_ratio < 0.25:
            emotion = "Happy"
            dist["Happy"] = 85.0
            confidence = 90.0
        elif brow_dist < 0.18:
            emotion = "Angry"
            dist["Angry"] = 75.0
            confidence = 78.0
        else:
            emotion = "Neutral"
            dist["Neutral"] = 85.0
            confidence = 85.0

        return emotion, confidence, dist

    def _update_metrics(self, emotion: str, confidence: float, distribution: Dict[str, float]) -> None:
        self.emotion_history.append(emotion)
        self.emotion_counts[emotion] += 1

        total = sum(self.emotion_counts.values()) or 1
        overall_dist = {e: round((self.emotion_counts[e] / total) * 100.0, 1) for e in self.emotions_list}

        most_common = self.emotion_counts.most_common(1)
        dominant_overall = most_common[0][0] if most_common else "Neutral"
        stability = float((self.emotion_counts[dominant_overall] / total) * 100.0)

        self.last_metrics = {
            "current_emotion": emotion,
            "emotion_confidence": round(confidence, 1),
            "dominant_emotion": dominant_overall,
            "emotion_stability": round(stability, 1),
            "emotion_distribution": overall_dist,
            "live_distribution": distribution,
            "emotion_timeline": list(self.emotion_history)[-50:],
        }

    def _get_empty_metrics(self) -> Dict[str, Any]:
        return {
            "current_emotion": "Neutral",
            "emotion_confidence": 0.0,
            "dominant_emotion": "Neutral",
            "emotion_stability": 0.0,
            "emotion_distribution": {e: 0.0 for e in self.emotions_list},
            "live_distribution": {e: 0.0 for e in self.emotions_list},
            "emotion_timeline": [],
        }