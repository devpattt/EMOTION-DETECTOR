import cv2
from deepface import DeepFace

def detect_emotion():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Webcam is working. Press 'q' to quit the application.")
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot grab frame.")
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(analysis, list) and len(analysis) > 0:
                emotion = analysis[0]['dominant_emotion'] 
            else:
                emotion = "No Face Detected"

            cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(frame, "Emotion Detection Failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Analysis Error: {e}")

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion()
