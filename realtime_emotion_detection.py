import cv2
import numpy as np
from tensorflow.keras.models import load_model
# Load pre-trained model
model = load_model('models/emotion_recognition_model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
# Load face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start webcam
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))
        preds = model.predict(roi)[0]
        label = emotion_labels[np.argmax(preds)]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




