from pathlib import Path
import os

# Paths - support both local and Docker
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = os.getenv('MODEL_PATH', BASE_DIR / 'models' / 'resnet50_stage2.keras')

# Model settings
IMG_SIZE = (224, 224)
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# API settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

print(f"Config loaded. Model path: {MODEL_PATH}")