import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerOptions

nest_asyncio.apply()  # Required for Colab or notebooks

# Arabic label mapping
label_to_arabic = {
    "Alef": "ا", "Beh": "ب", "Teh": "ت", "Theh": "ث", "Jeem": "ج", "Hah": "ح", "Khah": "خ",
    "Dal": "د", "Thal": "ذ", "Reh": "ر", "Zain": "ز", "Seen": "س", "Sheen": "ش", "Sad": "ص",
    "Dad": "ض", "Tah": "ط", "Zah": "ظ", "Ain": "ع", "Ghain": "غ", "Feh": "ف", "Qaf": "ق",
    "Kaf": "ك", "Lam": "ل", "Meem": "م", "Noon": "ن", "Heh": "ه", "Waw": "و", "Yeh": "ي",
    "Teh_Marbuta": "ة", "Laa": "لا", "Al": "ال"
}

recognizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    # ✅ Use relative path to model for deployment
    model_path = os.path.join(os.path.dirname(__file__), "letters_model.task")
    recognizer = GestureRecognizer.create_from_model_path(model_path)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = recognizer.recognize(mp_image)

    top_label = result.gestures[0][0].category_name if result.gestures else None
    arabic_char = label_to_arabic.get(top_label, "غير معروف" if top_label else "لا يوجد حركة")

    return {"prediction": arabic_char}
