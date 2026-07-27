import json
import os
import sys
import time
from typing import Any, Dict
import warnings

# Suppress C++ initialization logs
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')
import cv2

# Import local telemetry modules
from attention_tracker import AttentionTracker
from confidence_analyzer import ConfidenceAnalyzer
from emotion_detector import EmotionAnalysis
from eye_contact_tracker import EyeContactTracker
from face_detector import FaceDetector
from ollama_feedback import generate_llm_feedback


def get_performance_level(score: float) -> str:
    """
    Deterministically maps a numerical score (0-100) to a fixed qualitative label.
    """
    if score >= 85.0:
        return "Excellent"
    elif score >= 70.0:
        return "Good"
    elif score >= 55.0:
        return "Average"
    elif score >= 40.0:
        return "Needs Improvement"
    else:
        return "Not Ready"


def calculate_weighted_overall_score(
    confidence: float,
    attention: float,
    eye_contact: float,
    emotion_stability: float,
    head_stability: float,
    blink_rate: float,
    visibility: float,
    distractions: int
) -> int:
    """
    Computes a realistic overall score using an 8-signal weighted model.
    """
    if 10 <= blink_rate <= 22:
        blink_score = 100.0
    elif 22 < blink_rate <= 32:
        blink_score = 70.0
    else:
        blink_score = 40.0

    distraction_score = max(0.0, 100.0 - (distractions * 15.0))

    weighted_score = (
        (confidence * 0.25)
        + (attention * 0.20)
        + (eye_contact * 0.20)
        + (emotion_stability * 0.10)
        + (head_stability * 0.10)
        + (visibility * 0.05)
        + (blink_score * 0.05)
        + (distraction_score * 0.05)
    )

    return int(max(0, min(100, round(weighted_score))))


def main() -> None:
    face_det = FaceDetector()
    eye_track = EyeContactTracker()
    emotion_det = EmotionAnalysis()
    atten_track = AttentionTracker()
    conf_analyze = ConfidenceAnalyzer()

    # Hardware Camera Initialization with DirectShow for Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("\n[Hardware Error]: Unable to access webcam on index 0 or 1.")
            sys.exit(1)

    total_frames: int = 0
    start_time: float = time.time()
    last_frame_time: float = time.time()

    # Pre-initialize dictionary metrics to fix Pylance warnings
    face_telemetry: Dict[str, Any] = face_det.get_telemetry()
    eye_metrics: Dict[str, Any] = eye_track.get_metrics()
    emotion_metrics: Dict[str, Any] = emotion_det.get_metrics()
    attention_metrics: Dict[str, Any] = atten_track.get_metrics()
    confidence_metrics: Dict[str, Any] = conf_analyze.get_metrics()

    print("\n" + "=" * 70)
    print("   HIREFLOW MOCK AI INTERVIEW ENGINE - CAMERA ACTIVE")
    print("   Press 'q' inside the video window to stop and generate AI feedback.")
    print("=" * 70 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame.size == 0:
                print("\n[Video Error]: Frame capture failed or empty frame received.")
                break

            total_frames += 1
            current_time: float = time.time()
            frame_duration: float = current_time - last_frame_time
            last_frame_time = current_time

            # 1. Computer Vision Processing Pipeline
            faces = face_det.process_frame(frame)
            face_box = face_det.extract_landmarks(faces)
            face_telemetry = face_det.get_telemetry()
            raw_landmarks = face_det.get_raw_landmarks()

            eye_contact_present: bool = eye_track.check_contact(frame, face_box, raw_landmarks)
            eye_metrics = eye_track.get_metrics()

            current_emotion: str = emotion_det.analyze_emotion(face_box, 80.0, frame, raw_landmarks)
            emotion_metrics = emotion_det.get_metrics()

            _ = atten_track.calculate_attention(
                face_box, eye_contact_present, frame.shape[1], face_telemetry, eye_metrics, frame_duration
            )
            attention_metrics = atten_track.get_metrics()

            _ = conf_analyze.get_confidence(
                current_emotion=current_emotion,
                eye_contact_detected=eye_contact_present,
                face_box=face_box,
                frame_width=frame.shape[1],
                face_telemetry=face_telemetry,
                eye_metrics=eye_metrics,
                emotion_metrics=emotion_metrics,
                attention_metrics=attention_metrics,
            )
            confidence_metrics = conf_analyze.get_metrics()

            cv2.imshow("HireFlow AI Interview Workspace", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO]: Session stopped by user.")

    cap.release()
    cv2.destroyAllWindows()

    total_duration: float = time.time() - start_time

    # Compute Multi-Signal Weighted Overall Score
    overall_score: int = calculate_weighted_overall_score(
        confidence=float(confidence_metrics.get("average_confidence", 0.0)),
        attention=float(attention_metrics.get("average_attention_percentage", 0.0)),
        eye_contact=float(eye_metrics.get("eye_contact_percentage", 0.0)),
        emotion_stability=float(emotion_metrics.get("emotion_stability", 0.0)),
        head_stability=float(face_telemetry.get("head_stability", 0.0)),
        blink_rate=float(eye_metrics.get("blink_rate_bpm", 15.0)),
        visibility=float(face_telemetry.get("visibility_percentage", 100.0)),
        distractions=int(attention_metrics.get("distraction_events", 0))
    )

    # Clean face metrics (strip 468 raw landmark points)
    clean_face_metrics = face_telemetry.copy()
    clean_face_metrics.pop("landmarks", None)

    # Compute Python Qualitative Classifications Deterministically
    confidence_score = float(confidence_metrics.get("average_confidence", 0.0))
    attention_score = float(attention_metrics.get("average_attention_percentage", 0.0))
    eye_contact_score = float(eye_metrics.get("eye_contact_percentage", 0.0))
    emotion_stability_score = float(emotion_metrics.get("emotion_stability", 0.0))
    head_stability_score = float(face_telemetry.get("head_stability", 0.0))

    classified_levels: Dict[str, Any] = {
        "overall_performance_level": get_performance_level(overall_score),
        "interview_readiness_level": get_performance_level(overall_score),
        "confidence_level": get_performance_level(confidence_score),
        "attention_level": get_performance_level(attention_score),
        "eye_contact_level": get_performance_level(eye_contact_score),
        "emotion_stability_level": get_performance_level(emotion_stability_score),
        "head_stability_level": get_performance_level(head_stability_score),
    }

    # Package Complete Telemetry + Python Pre-Calculated Classifications
    final_session_report: Dict[str, Any] = {
        "session_id": f"interview_{int(start_time)}",
        "total_duration_seconds": round(total_duration, 2),
        "total_frames_processed": total_frames,
        "overall_interview_score": overall_score,
        "performance_classifications": classified_levels,
        "confidence_metrics": confidence_metrics,
        "attention_metrics": attention_metrics,
        "eye_metrics": eye_metrics,
        "emotion_metrics": emotion_metrics,
        "face_metrics": clean_face_metrics,
    }

    # Debug: Print telemetry to terminal before triggering Ollama
    print("\n================ TELEMETRY SENT TO OLLAMA ================\n")
    print(json.dumps(final_session_report, indent=2))
    print("\n==========================================================\n")

    # Generate Output Feedback via Ollama (Streaming output)
    print("\n[Ollama]: Analyzing telemetry & generating structured HireFlow report...\n")
    _ = generate_llm_feedback(final_session_report)


if __name__ == "__main__":
    main()