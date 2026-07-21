"""
Multi-Signal Attention & Focus Telemetry Engine.
Integrates Eye Contact, Head Pose, Face Centering, Visibility, and Distance
to calculate real-time Attention Percentage, Focus Time, Looking Away Time, and Distraction Events.
"""

from typing import Any, Dict, List


class AttentionTracker:
    """
    Evaluates candidate focus by synthesizing spatial posture, eye contact, and head orientation signals.
    """

    def __init__(self) -> None:
        self.total_frames = 0
        self.running_attention_score = 0.0

        self.focus_seconds = 0.0
        self.distraction_seconds = 0.0
        self.distraction_events = 0
        self.in_distraction = False

        self.attention_timeline: List[float] = []
        self.last_metrics: Dict[str, Any] = self._get_empty_metrics()

    def calculate_attention(
        self,
        face_box: Any = None,
        eye_contact_detected: bool = False,
        frame_width: int = 640,
        face_telemetry: Dict[str, Any] = None,
        eye_metrics: Dict[str, Any] = None,
        frame_duration: float = 0.033,
    ) -> float:
        """
        Main calculation interface method compatible with main.py pipeline.
        """
        self.total_frames += 1
        score = 100.0

        if face_box is None or (face_telemetry and not face_telemetry.get("face_detected", False)):
            score = 0.0
            is_distracted = True
        else:
            is_distracted = False

            if not eye_contact_detected:
                score -= 30.0
                is_distracted = True

            if face_telemetry:
                head_status = face_telemetry.get("head_status", "Straight")
                if head_status != "Straight":
                    score -= 25.0
                    is_distracted = True

                pos = face_telemetry.get("face_position", "Center")
                if pos != "Center":
                    score -= 15.0

                dist = face_telemetry.get("face_distance", "Ideal")
                if dist != "Ideal":
                    score -= 10.0

                vis = face_telemetry.get("visibility_percentage", 100.0)
                if vis < 80.0:
                    score -= 15.0

        score = max(0.0, min(100.0, score))
        self.running_attention_score += score
        self.attention_timeline.append(score)

        if is_distracted:
            self.distraction_seconds += frame_duration
            if not self.in_distraction:
                self.distraction_events += 1
                self.in_distraction = True
        else:
            self.focus_seconds += frame_duration
            self.in_distraction = False

        avg_attention = float(self.running_attention_score / self.total_frames)

        self.last_metrics = {
            "attention_score": round(score, 1),
            "average_attention_percentage": round(avg_attention, 1),
            "focus_time_seconds": round(self.focus_seconds, 2),
            "looking_away_time_seconds": round(self.distraction_seconds, 2),
            "distraction_events": self.distraction_events,
            "is_currently_distracted": is_distracted,
            "attention_timeline": self.attention_timeline[-100:],
        }

        return score

    def get_metrics(self) -> Dict[str, Any]:
        return self.last_metrics

    def _get_empty_metrics(self) -> Dict[str, Any]:
        return {
            "attention_score": 0.0,
            "average_attention_percentage": 0.0,
            "focus_time_seconds": 0.0,
            "looking_away_time_seconds": 0.0,
            "distraction_events": 0,
            "is_currently_distracted": True,
            "attention_timeline": [],
        }