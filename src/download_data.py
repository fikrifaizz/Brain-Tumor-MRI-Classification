import kaggle
import os

kaggle.api.authenticate()

os.makedirs('data/raw', exist_ok=True)

kaggle.api.dataset_download_files('masoudnickparvar/brain-tumor-mri-dataset', path='data/raw', unzip=True)