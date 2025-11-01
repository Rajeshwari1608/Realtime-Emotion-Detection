import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = load_model('models/emotion_recognition_model.h5')

# Test data
test_dir = 'dataset/test'
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(test_dir, target_size=(48,48),
                                            color_mode='grayscale', batch_size=64,
                                            class_mode='categorical', shuffle=False)

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")


