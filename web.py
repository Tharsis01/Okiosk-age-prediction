import webbrowser
import detect

from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from keras.models import load_model
import cv2
import numpy as np
from yoloface import face_analysis

# Load model
model_age = load_model('E:/Okiosk-age-prediction-main/model/model_age.hdf5')

a = []
frame = cv2.VideoCapture(0)
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def create():
    a = detect.detect_video()
  
    return jsonify({"age": a})


if __name__=="__main__":
    app.run(debug=True)