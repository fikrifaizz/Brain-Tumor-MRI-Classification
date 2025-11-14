import tensorflow as tf
from config import MODEL_PATH

class ModelLoader:
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self):
        if self._model is None:
            print(f"Loading model from {MODEL_PATH}...")
            self._model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
        return self._model
    
    @property
    def model(self):
        if self._model is None:
            return self.load()
        return self._model

# Global instance
model_loader = ModelLoader()