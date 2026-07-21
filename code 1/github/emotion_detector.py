class EmotionAnalysis:
    """Maps confidence scores strictly to Confident, Neutral, and Nervous states."""
    def __init__(self):
        pass

    def analyze_emotion(self, face_box, confidence_score):
        if face_box is None:
            return "Nervous"
            
        if confidence_score >= 80:
            return "Confident"
        elif confidence_score >= 55:
            return "Neutral"
        else:
            return "Nervous"