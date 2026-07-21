import cv2
import sys
import time
import os
from typing import Any, Dict

# Ensure the system path knows where to find your backend local custom trackers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from face_detector import FaceDetector
    from eye_contact_tracker import EyeContactTracker
    from emotion_detector import EmotionAnalysis
    from attention_tracker import AttentionTracker
    from confidence_analyzer import ConfidenceAnalyzer
except ImportError as e:
    print(f"\n[Import Error]: Could not find supporting tracking scripts. {e}")
    print(
        "Ensure face_detector.py, eye_contact_tracker.py, "
        "attention_tracker.py, confidence_analyzer.py, "
        "and emotion_detector.py are all in the same folder."
    )
    sys.exit(1)


def main() -> None:
    # Cast custom classes to Any so Pylance stops complaining about untyped modules
    face_det: Any = FaceDetector()
    eye_track: Any = EyeContactTracker()
    emotion_det: Any = EmotionAnalysis()
    atten_track: Any = AttentionTracker()
    conf_analyze: Any = ConfidenceAnalyzer()

    # Open webcam
    cap: cv2.VideoCapture = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "\n[Hardware Error]: Could not open webcam workspace. "
            "Verify your camera connection."
        )
        sys.exit(1)

    # ---------------------------------------------------------
    # INTERVIEW METRICS
    # ---------------------------------------------------------
    total_frames: int = 0
    eye_contact_frames: int = 0

    running_attention: float = 0.0
    running_confidence: float = 0.0

    current_emotion: str = "Neutral"

    # Precise real-time tracking clocks in seconds
    total_looking_at_camera_time: float = 0.0
    total_turning_away_time: float = 0.0

    # Solves the initialization lag spike: set to None initially
    last_frame_time: Any = None 

    # Strict three-emotion summary map
    emotion_counts: Dict[str, int] = {
        "Confident": 0,
        "Neutral": 0,
        "Nervous": 0,
    }

    print("\n" + "=" * 60)
    print("   HIREFLOW MOCK AI INTERVIEW ENGINE v2.0 - ACTIVE")
    print("   [Live Telemetry Hidden for Candidate Focus]")
    print("   Press 'q' inside the video window to finish the interview.")
    print("=" * 60 + "\n")

    # ---------------------------------------------------------
    # MAIN INTERVIEW LOOP
    # ---------------------------------------------------------
    while True:
        ret, frame = cap.read()

        if not ret:
            print("\n[Camera Error]: Failed to read frame from webcam.")
            break

        total_frames += 1
        _, w, _ = frame.shape

        # -----------------------------------------------------
        # REAL-TIME FRAME DURATION (BUG FIX ACCURACY)
        # -----------------------------------------------------
        current_time: float = time.time()
        
        if last_frame_time is None:
            # First frame baseline set after successful hardware acquisition
            frame_duration: float = 0.0
        else:
            frame_duration = current_time - last_frame_time
            
        last_frame_time = current_time

        # -----------------------------------------------------
        # FACE DETECTION
        # -----------------------------------------------------
        faces: Any = face_det.process_frame(frame)
        face_box: Any = face_det.extract_landmarks(faces)
        face_detected: bool = face_box is not None

        eye_contact_present: bool = False
        is_looking_away_this_frame: bool = False

        if face_detected:
            # Unpack dimensions cleanly
            x, y, fw, fh = face_box

            # Draw clean bounding box around detected face
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (245, 158, 11), 2)

            # -------------------------------------------------
            # EYE CONTACT DETECTION
            # -------------------------------------------------
            eye_contact_present = bool(eye_track.check_contact(frame, face_box))

            if eye_contact_present:
                eye_contact_frames += 1

            # -------------------------------------------------
            # FACE POSITION ANALYSIS
            # -------------------------------------------------
            face_center_x: float = x + (fw / 2)
            frame_center_x: float = w / 2

            # User is considered looking away when:
            # 1. Face moves significantly away from screen center
            # 2. Direct eye contact is lost
            if abs(face_center_x - frame_center_x) > (w * 0.15) or not eye_contact_present:
                is_looking_away_this_frame = True
        else:
            is_looking_away_this_frame = True

        # -----------------------------------------------------
        # ACCUMULATE EXACT REAL-TIME DURATIONS
        # -----------------------------------------------------
        if is_looking_away_this_frame:
            total_turning_away_time += frame_duration
        else:
            total_looking_at_camera_time += frame_duration

        # -----------------------------------------------------
        # BACKGROUND AI CALCULATIONS
        # -----------------------------------------------------
        if face_detected:
            frame_attention: Any = atten_track.calculate_attention(face_box, eye_contact_present, w)
            running_attention += float(frame_attention)

            frame_conf: Any = conf_analyze.get_confidence(current_emotion, eye_contact_present, face_box, w)
            current_emotion = str(emotion_det.analyze_emotion(face_box, frame_conf))

            final_conf: Any = conf_analyze.get_confidence(current_emotion, eye_contact_present, face_box, w)
            running_confidence += float(final_conf)
        else:
            running_attention += 10.0
            running_confidence += 20.0

        # -----------------------------------------------------
        # EMOTION COUNTER
        # -----------------------------------------------------
        if current_emotion in emotion_counts:
            emotion_counts[current_emotion] += 1

        # -----------------------------------------------------
        # SILENT LIVE TELEMETRY
        # -----------------------------------------------------
        sys.stdout.write(f"\r[Telemetry Processing Frame: {total_frames} | Live Dynamic Analysis Active]")
        sys.stdout.flush()

        # Show webcam workspace
        cv2.imshow("HireFlow Live Interview Workspace", frame)

        # Press Q to finish interview
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # POST-INTERVIEW CALCULATIONS
    # ---------------------------------------------------------
    if total_frames > 0:
        avg_attention: int = int(running_attention / total_frames)
        avg_confidence: int = int(running_confidence / total_frames)
        eye_contact_final: int = int((eye_contact_frames / total_frames) * 100)
        dominant_emotion: str = str(
            max(emotion_counts, key=lambda emotion: emotion_counts[emotion])
        )
    else:
        avg_attention = 0
        avg_confidence = 0
        eye_contact_final = 0
        dominant_emotion = "Neutral"

    # Calculate final overall interview score
    overall_score: int = int((avg_attention + avg_confidence + eye_contact_final) / 3)

    # ---------------------------------------------------------
    # FINAL PERFORMANCE ASSESSMENT SCORECARD
    # ---------------------------------------------------------
    print("\n\n" + "#" * 70)
    print("                 HIREFLOW MOCK AI INTERVIEW REPORT SUMMARY")
    print("#" * 70)
    print(f"  ► OVERALL INTERVIEW SCORE        : {overall_score} / 100")
    print("-" * 70)
    print(f"  • Confidence Performance Score  : {avg_confidence}%")
    print(f"  • Attention Focus Level Score   : {avg_attention}%")
    print(f"  • Eye Contact Gaze Continuity   : {eye_contact_final}%")
    print(f"  • Total Time Focused on Camera  : {total_looking_at_camera_time:.2f} seconds")
    print(f"  • Total Time Turned Side/Away   : {total_turning_away_time:.2f} seconds")
    print(f"  • Dominant Behavioral Expression: {dominant_emotion}")
    print("-" * 70)
    print("  AI EVALUATION FEEDBACK & SUGGESTIONS:")

    # ---------------------------------------------------------
    # AUTOMATIC FEEDBACK
    # ---------------------------------------------------------
    if total_turning_away_time > total_looking_at_camera_time:
        print(
            "  ⚠️ ALERT: Your turned/side gaze duration was higher than your direct camera focus.\n"
            "            Try to position your laptop screen directly in front of your posture profile."
        )
    elif eye_contact_final < 60:
        print(
            "  ⚠️ ALERT: High blink density or temporary gaze drift patterns logged.\n"
            "            Try to stabilize your eye contact focus with the lens when answering."
        )
    else:
        print(
            "  ✅ EXCELLENT PRESENTATION: Exceptional metric footprint signature profile.\n"
            "                             Professional alignment and calm posture maintained successfully."
        )

    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()