import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# This class is an adaptation of CompDetCNN from the smart_ui repository
# for testing purposes within the 'logos' project.

class TestClassifier:
    def __init__(self, model_name="cnn-generalized.h5"):
        # We construct the absolute path to the model file.
        # It's assumed this test script is run from the project root.
        model_path = os.path.abspath(os.path.join('models', model_name))
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please ensure it's downloaded.")

        self.model = load_model(model_path)
        
        # Based on the smart_ui project, the generalized model has 13 classes.
        # We define them here for interpreting the prediction output.
        self.class_map = [
            'Button', 'CheckBox', 'Chronometer', 'EditText', 'Image', 
            'ImageButton', 'NumberPicker', 'RadioButton', 'RatingBar', 
            'SeekBar', 'Spinner', 'Switch', 'TextView'
        ]
        self.image_shape = (64, 64, 3) # Expected input shape for the model

    def preprocess_img(self, image):
        """
        Resizes and normalizes the image to be compatible with the CNN model.
        """
        # Ensure image is in BGR format if it has an alpha channel
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255.0).astype('float32')
        x = np.array([x])
        return x

    def predict(self, imgs, compos):
        """
        Predicts the class for a list of component images.
        
        Args:
            imgs (list): A list of cropped component images (as numpy arrays).
            compos (list): A list of corresponding component objects to be updated.
        """
        if self.model is None:
            print("*** No model loaded, prediction skipped. ***")
            return
            
        for i, img in enumerate(imgs):
            if img is None or img.size == 0:
                compos[i].category = 'Unknown'
                continue
            
            preprocessed_img = self.preprocess_img(img)
            prediction = self.model.predict(preprocessed_img)
            predicted_class_index = np.argmax(prediction)
            
            if predicted_class_index < len(self.class_map):
                compos[i].category = self.class_map[predicted_class_index]
            else:
                compos[i].category = 'Unknown'
