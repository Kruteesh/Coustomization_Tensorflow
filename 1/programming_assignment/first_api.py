import PIL.ImImagePlugin
from fastapi import FastAPI, File, UploadFile
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse 


#### PACKAGE IMPORTS ####

# Run this cell first to import all required packages. Do not make any imports elsewhere in the notebook

import tensorflow as tf

import PIL.Image
import PIL.ImageOps
from tensorflow.keras.models import load_model
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
import seaborn as sns

first_app = FastAPI()

first_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )


model_path = "C:/Users/krute/OneDrive/Desktop/Tensorflow2/Files_2/Files/home/jovyan/work/1/programming_assignment/models/MobileNetV2.h5"
pretrained_model = load_model(model_path)

@first_app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert("L")
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((160,160), PIL.Image.ANTIALIAS)
    img_array = np.array(pil_image).reshape(1,-1)
    prediction = pretrained_model.predict(img_array)
    return {"prediction": int(prediction)}

@first_app.get("/")
async def read_index():
    return FileResponse('index.html')
