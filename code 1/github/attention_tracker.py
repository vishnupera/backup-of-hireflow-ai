class AttentionTracker:
    """Monitors physical face centering and gaze metrics to calculate focus telemetry."""
    def __init__(self):
        pass

    def calculate_attention(self, face_box, eye_contact_detected, frame_width):
        if face_box is None:
            return 10
        
        x, y, w, h = face_box
        face_center_x = x + (w / 2)
        frame_center_x = frame_width / 2
        
        score = 100
        
        if abs(face_center_x - frame_center_x) > (frame_width * 0.15):
            score -= 25
        if not eye_contact_detected:
            score -= 25
            
        return max(10, score)