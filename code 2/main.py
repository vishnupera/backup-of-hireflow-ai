"""
HireFlow Master AI Interview Controller.
Coordinates Face Mesh, Iris Tracking, Emotion Recognition, Attention Telemetry,
and Behavioral Confidence to run with a clean live camera feed and zero Pylance type issues.
"""

import os
import sys
import time
from typing import Any, Dict, List
import warnings

# Hide C++ log noise and redirect stderr during library startup
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class NullWriter:
    def write(self, text: str) -> None:
        pass

    def flush(self) -> None:
        pass


# Temporarily redirect stderr to null while importing heavy C++ bindings
_stderr_backup = sys.stderr
sys.stderr = NullWriter()  # type: ignore

warnings.filterwarnings('ignore')

import cv2

# Restore stderr after MediaPipe/OpenCV bindings finish loading
sys.stderr = _stderr_backup

# Import local telemetry modules
from attention_tracker import AttentionTracker
from confidence_analyzer import ConfidenceAnalyzer
from emotion_detector import EmotionAnalysis
from eye_contact_tracker import EyeContactTracker
from face_detector import FaceDetector


def generate_direct_feedback(report: Dict[str, Any]) -> List[str]:
    """
    Generates dynamic, human-sounding feedback directly in Python 
    based on real-time MediaPipe and OpenCV metrics.
    """
    feedback: List[str] = []

    eye_pct: float = float(report["eye_metrics"]["eye_contact_percentage"])
    bpm: float = float(report["eye_metrics"]["blink_rate_bpm"])
    distractions: int = int(report["attention_metrics"]["distraction_events"])
    head_stab: float = float(report["face_metrics"]["head_stability"])
    dominant_emo: str = str(report["emotion_metrics"]["dominant_emotion"])

    # 1. Overall Performance Evaluation
    if report["overall_interview_score"] >= 80:
        feedback.append(
            "🌟 OVERALL PERFORMANCE\n"
            "You performed very well in this interview. You maintained good eye contact, a steady posture, "
            "and showed confident body language. Keep practicing to maintain this level of performance."
        )
    elif report["overall_interview_score"] >= 60:
        feedback.append(
            "👍 OVERALL PERFORMANCE\n"
            "Your interview performance was good, but there were a few areas that need improvement. "
            "Try to maintain better eye contact, reduce unnecessary head movement, and stay focused on the camera."
        )
    else:
        feedback.append(
            "⚠️ OVERALL PERFORMANCE\n"
            "Your interview performance needs improvement. We noticed frequent distractions, reduced eye contact, "
            "or unstable posture. Practicing mock interviews will help you become more confident."
        )

    # 2. Eye Contact & Blink Analysis
    if eye_pct < 60:
        if bpm > 22:
            feedback.append(
                f"👀 EYE CONTACT\n"
                f"You maintained eye contact for only {eye_pct:.1f}% of the interview, and your blink rate "
                f"was {bpm:.1f} blinks per minute.\n\n"
                "What this means:\n"
                "• You looked away from the camera frequently.\n"
                "• Frequent blinking may indicate nervousness.\n\n"
                "How to improve:\n"
                "• Look directly at the camera while answering.\n"
                "• Take a deep breath before speaking.\n"
                "• Avoid looking at notes or other screens."
            )
        else:
            feedback.append(
                f"👀 EYE CONTACT\n"
                f"You maintained eye contact for {eye_pct:.1f}% of the interview.\n\n"
                "What this means:\n"
                "• You looked away from the interviewer several times.\n\n"
                "How to improve:\n"
                "• Keep your eyes focused on the camera.\n"
                "• Practice speaking without reading notes."
            )
    else:
        feedback.append(
            f"✅ EYE CONTACT\n"
            f"You maintained eye contact for {eye_pct:.1f}% of the interview.\n\n"
            "Excellent! Your eye contact remained natural and consistent, helping you appear confident and engaged."
        )

    # 3. Head Position & Posture
    event_str: str = "1 distraction event" if distractions == 1 else f"{distractions} distraction events"

    if distractions > 2 or head_stab < 60:
        feedback.append(
            f"📐 HEAD POSITION & POSTURE\n"
            f"We detected {event_str}, and your head stability score was {head_stab:.1f}%.\n\n"
            "What this means:\n"
            "• Your head moved frequently.\n"
            "• You may have been looking away from the interviewer.\n\n"
            "How to improve:\n"
            "• Sit comfortably in front of the camera.\n"
            "• Keep your head steady while speaking.\n"
            "• Position your notes behind the camera if needed."
        )
    else:
        feedback.append(
            f"✅ HEAD POSITION & POSTURE\n"
            f"Your head stability score was {head_stab:.1f}%.\n\n"
            "Great job! Your head remained steady and well aligned with the camera throughout the interview."
        )

    # 4. Facial Expression
    if dominant_emo in ["Nervous", "Fear", "Sad", "Angry"]:
        feedback.append(
            f"🎭 FACIAL EXPRESSION\n"
            f"Your most common facial expression was '{dominant_emo}'.\n\n"
            "What this means:\n"
            "• You appeared slightly tense or uncomfortable during parts of the interview.\n\n"
            "How to improve:\n"
            "• Relax your facial muscles.\n"
            "• Smile naturally when greeting the interviewer.\n"
            "• Pause and breathe before answering difficult questions."
        )
    else:
        feedback.append(
            f"✅ FACIAL EXPRESSION\n"
            f"Your dominant facial expression was '{dominant_emo}'.\n\n"
            "Your facial expressions appeared calm, natural, and professional throughout the interview."
        )

    return feedback


def print_report_summary(report: Dict[str, Any]) -> None:
    """Prints final post-interview report table directly to the terminal."""
    feedback_points: List[str] = generate_direct_feedback(report)

    print("\n\n" + "#" * 70)
    print("                 HIREFLOW MOCK AI INTERVIEW REPORT SUMMARY            ")
    print("#" * 70)
    print(f"  ► OVERALL INTERVIEW SCORE       : {report['overall_interview_score']} / 100")
    print("-" * 70)
    print(f"  • Total Interview Duration     : {report['total_duration_seconds']} seconds")
    print(f"  • Overall Confidence Score     : {report['confidence_metrics']['average_confidence']}%")
    print(f"  • Average Attention Score      : {report['attention_metrics']['average_attention_percentage']}%")
    print(f"  • Eye Contact Continuity       : {report['eye_metrics']['eye_contact_percentage']}%")
    print(f"  • Dominant Emotion Expression  : {report['emotion_metrics']['dominant_emotion']}")
    print(f"  • Total Distraction Events     : {report['attention_metrics']['distraction_events']}")
    print(f"  • Final Blink Rate             : {report['eye_metrics']['blink_rate_bpm']} BPM")
    print("-" * 70)
    print("  AUTOMATED AI EVALUATION & FEEDBACK:")
    print("-" * 70)

    for point in feedback_points:
        print(f"  {point}\n")

    print("#" * 70 + "\n")


def main() -> None:
    # Initialize Core Engines
    face_det = FaceDetector()
    eye_track = EyeContactTracker()
    emotion_det = EmotionAnalysis()
    atten_track = AttentionTracker()
    conf_analyze = ConfidenceAnalyzer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[Hardware Error]: Unable to access webcam.")
        sys.exit(1)

    total_frames: int = 0
    start_time: float = time.time()
    last_frame_time: float = time.time()

    # Pre-initialize telemetry dicts to prevent "possibly unbound" Pylance warnings
    face_telemetry: Dict[str, Any] = face_det.get_telemetry()
    eye_metrics: Dict[str, Any] = eye_track.get_metrics()
    emotion_metrics: Dict[str, Any] = emotion_det.get_metrics()
    attention_metrics: Dict[str, Any] = atten_track.get_metrics()
    confidence_metrics: Dict[str, Any] = conf_analyze.get_metrics()

    print("\n" + "=" * 70)
    print("   HIREFLOW MOCK AI INTERVIEW ENGINE - CAMERA ACTIVE")
    print("   Press 'q' inside the video window to finish and view evaluation.")
    print("=" * 70 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            current_time: float = time.time()
            frame_duration: float = current_time - last_frame_time
            last_frame_time = current_time

            # 1. Face Detector Engine
            faces = face_det.process_frame(frame)
            face_box = face_det.extract_landmarks(faces)
            face_telemetry = face_det.get_telemetry()
            raw_landmarks = face_det.get_raw_landmarks()

            # 2. Eye & Iris Tracker Engine
            eye_contact_present: bool = eye_track.check_contact(frame, face_box, raw_landmarks)
            eye_metrics = eye_track.get_metrics()

            # 3. Real Emotion Engine
            current_emotion: str = emotion_det.analyze_emotion(face_box, 80.0, frame, raw_landmarks)
            emotion_metrics = emotion_det.get_metrics()

            # 4. Multi-Signal Attention Engine
            _ = atten_track.calculate_attention(
                face_box, eye_contact_present, frame.shape[1], face_telemetry, eye_metrics, frame_duration
            )
            attention_metrics = atten_track.get_metrics()

            # 5. Multi-Modal Confidence Engine
            _ = conf_analyze.get_confidence(
                current_emotion, eye_contact_present, face_box, frame.shape[1],
                face_telemetry, eye_metrics, emotion_metrics, attention_metrics
            )
            confidence_metrics = conf_analyze.get_metrics()

            # Display clean live camera feed without HUD/Mesh overlays
            cv2.imshow("HireFlow AI Interview Workspace", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO]: Interview session terminated by user.")

    cap.release()
    cv2.destroyAllWindows()

    total_duration: float = time.time() - start_time

    # Compile Final Session Metrics Report Dictionary In-Memory
    final_session_report: Dict[str, Any] = {
        "session_id": f"interview_{int(start_time)}",
        "total_duration_seconds": round(total_duration, 2),
        "total_frames_processed": total_frames,
        "overall_interview_score": int(
            (
                float(confidence_metrics.get("average_confidence", 0.0))
                + float(attention_metrics.get("average_attention_percentage", 0.0))
                + float(eye_metrics.get("eye_contact_percentage", 0.0))
            )
            / 3.0
        ),
        "face_metrics": face_telemetry,
        "eye_metrics": eye_metrics,
        "emotion_metrics": emotion_metrics,
        "attention_metrics": attention_metrics,
        "confidence_metrics": confidence_metrics,
    }

    # Print Post-Interview Assessment
    print_report_summary(final_session_report)


if __name__ == "__main__":
    main()