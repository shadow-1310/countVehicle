import os
import torch
import cv2
import pandas as pd
import easyocr 
import streamlit as st
from countVehicle.pipeline.stage2_model_prediction import ModelPredictionPipeline

st.sidebar.title("License Plate Tracking")

st.header("Original Video")
# st.video("artifacts/output.avi")
vid = st.file_uploader("Upload your video here....")
button = st.button("upload")
# output = cv2.VideoCapture("artifacts/output.mp4")
if button:
    path = os.path.join("artifacts", vid.name)
    with open(path,"wb") as f:
         f.write(vid.getbuffer())
    m = ModelPredictionPipeline(path)
    m.main()

    video_file = open('artifacts/output.webm', 'rb') #enter the filename with filepath

    video_bytes = video_file.read() #reading the file

    st.video(video_bytes)
