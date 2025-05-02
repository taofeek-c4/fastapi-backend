from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import os
import uvicorn
import logging
import gdown
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cattle_disease_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

FILE_ID = "1-i0qm-Bj9UrSMnp73XgMIDDGi-Vggtl1"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    try:
        logger.info("Model file not found locally. Downloading from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError("Model download failed.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed. Ensure the .h5 file exists and is valid.")

CLASS_LABELS = ["Foot and Mouth Disease", "Healthy", "Lumpy Skin Disease", "Mastitis"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f" Error preprocessing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

@app.post("/predict/")
async def anpredict(file: UploadFile = File(...)):
    try:
        logger.info(f" Received file: {file.filename} (type: {file.content_type})")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        logger.info(f" Saving image to: {image_path}")
        file.file.seek(0)
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.info(" Image saved successfully.")

        logger.info(" Preprocessing image...")
        processed_image = preprocess_image(image_path)
        logger.info(f" Image preprocessed. Shape: {processed_image.shape}")

        logger.info(" Starting prediction...")
        predictions = model.predict(processed_image)
        logger.info(f"Raw prediction: {predictions}")

        predicted_index = np.argmax(predictions[0])
        predicted_label = CLASS_LABELS[predicted_index]
        confidence = round(float(np.max(predictions[0])), 2)

        logger.info(f"Final Prediction: {predicted_label} (Confidence: {confidence})")

        os.remove(image_path)

        return JSONResponse(content={
            "status": "success",
            "message": "Prediction successful",
            "data": {
                "predicted_disease": predicted_label,
                "confidence": confidence
            }
        })

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)