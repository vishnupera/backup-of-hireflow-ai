class ConfidenceAnalyzer:
    """Compiles multi-modal telemetry streams into an accurate presentation metric."""
    def __init__(self):
        pass

    def get_confidence(self, current_emotion, eye_contact_detected, face_box, frame_width):
        if face_box is None:
            return 20
            
        x, y, w, h = face_box
        face_center_x = x + (w / 2)
        frame_center_x = frame_width / 2
        
        confidence = 90
        
        if current_emotion == "Nervous":
            confidence -= 20
        if not eye_contact_detected:
            confidence -= 15
        if abs(face_center_x - frame_center_x) > (frame_width * 0.15):
            confidence -= 10
            
        return max(5, min(100, confidence))