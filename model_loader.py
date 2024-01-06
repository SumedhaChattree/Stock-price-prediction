# model_loader.py

from keras.models import load_model

def load_keras_model(model_path='StockPP_TataMotors.keras'):
    return load_model(model_path)
