import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image

def predict_emotion(image_path):
    model = load_model('models/emotion_recognition_model.h5')
    emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    img = preprocess_image(image_path)
    preds = model.predict(img)
    label = emotion_labels[np.argmax(preds)]
    print(f"Predicted Emotion: {label}")
    return label
