"""
Multi-Modal Behavioral Confidence & Interview Assessment Engine.
Synthesizes Eye Contact, Attention, Head Stability, Emotion Stability, and Blink Rate
to compute Overall Confidence, Nervousness Index, and Candidate Engagement Score.
"""

from typing import Any, Dict


class ConfidenceAnalyzer:
    """
    Evaluates total executive presentation score by weighting multi-modal computer vision streams.
    """

    def __init__(self) -> None:
        self.total_frames = 0
        self.running_confidence = 0.0
        self.last_metrics: Dict[str, Any] = self._get_empty_metrics()

    def get_confidence(
        self,
        current_emotion: str = "Neutral",
        eye_contact_detected: bool = False,
        face_box: Any = None,
        frame_width: int = 640,
        face_telemetry: Dict[str, Any] = None,
        eye_metrics: Dict[str, Any] = None,
        emotion_metrics: Dict[str, Any] = None,
        attention_metrics: Dict[str, Any] = None,
    ) -> float:
        """
        Main score compiler method compatible with main.py pipeline.
        """
        self.total_frames += 1

        if face_box is None and (face_telemetry is None or not face_telemetry.get("face_detected", False)):
            self.running_confidence += 10.0
            return 10.0

        # Component Score Weights
        eye_score = eye_metrics.get("eye_contact_percentage", 50.0) if eye_metrics else (100.0 if eye_contact_detected else 0.0)
        atten_score = attention_metrics.get("average_attention_percentage", 50.0) if attention_metrics else 50.0
        head_stability = face_telemetry.get("head_stability", 80.0) if face_telemetry else 80.0
        emotion_stability = emotion_metrics.get("emotion_stability", 70.0) if emotion_metrics else 70.0

        # Calculate Nervousness Penalties
        nervousness_score = 0.0
        if current_emotion in ["Nervous", "Fear", "Sad"]:
            nervousness_score += 40.0

        if eye_metrics:
            blink_rate = eye_metrics.get("blink_rate_bpm", 15)
            if blink_rate > 25:  # High blink rate indicates nervousness
                nervousness_score += 30.0
            if not eye_metrics.get("eye_contact", False):
                nervousness_score += 20.0

        if face_telemetry and face_telemetry.get("head_status", "Straight") != "Straight":
            nervousness_score += 20.0

        nervousness_score = min(100.0, nervousness_score)

        # Weighted Overall Confidence Score
        confidence_score = (
            (eye_score * 0.30)
            + (atten_score * 0.30)
            + (head_stability * 0.20)
            + (emotion_stability * 0.20)
            - (nervousness_score * 0.25)
        )

        confidence_score = float(max(5.0, min(100.0, confidence_score)))
        self.running_confidence += confidence_score

        # Candidate Engagement Score
        engagement_score = float(max(0.0, min(100.0, (eye_score * 0.5) + (atten_score * 0.5))))

        self.last_metrics = {
            "overall_confidence_score": round(confidence_score, 1),
            "average_confidence": round(self.running_confidence / self.total_frames, 1),
            "nervousness_score": round(nervousness_score, 1),
            "engagement_score": round(engagement_score, 1),
            "eye_contact_score": round(eye_score, 1),
            "attention_score": round(atten_score, 1),
            "head_stability_score": round(head_stability, 1),
            "emotion_stability_score": round(emotion_stability, 1),
        }

        return confidence_score

    def get_metrics(self) -> Dict[str, Any]:
        return self.last_metrics

    def _get_empty_metrics(self) -> Dict[str, Any]:
        return {
            "overall_confidence_score": 0.0,
            "average_confidence": 0.0,
            "nervousness_score": 100.0,
            "engagement_score": 0.0,
            "eye_contact_score": 0.0,
            "attention_score": 0.0,
            "head_stability_score": 0.0,
            "emotion_stability_score": 0.0,
        }