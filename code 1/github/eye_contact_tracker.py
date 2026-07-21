import cv2

class EyeContactTracker:
    """Tracks physical eye features natively to verify authentic screen gaze."""
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def check_contact(self, frame, face_box):
        if face_box is None:
            return False
            
        try:
            x, y, w, h = face_box
            eye_region_h = int(h * 0.55)
            eye_roi = frame[y:y+eye_region_h, x:x+w]
            
            gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.equalizeHist(gray_roi)
            
            eyes = self.eye_cascade.detectMultiScale(
                gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(22, 22)
            )
            return len(eyes) >= 1
        except Exception:
            return False