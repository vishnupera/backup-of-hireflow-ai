import cv2

class FaceDetector:
    """Handles high-accuracy native OpenCV face detection using adaptive preprocessing."""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def process_frame(self, frame):
        if frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(120, 120)
        )
        return faces

    def extract_landmarks(self, results):
        if len(results) > 0:
            return max(results, key=lambda b: b[2] * b[3])
        return None