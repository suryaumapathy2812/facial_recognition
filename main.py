from flask import Flask, request, jsonify
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import utils
import joblib


detector = MTCNN()
embedder = FaceNet()

# CONSTANTS
THRESHOLD = 0.8
RESIZE = (160, 160)
EMBEDDINGS_PATH = "./model/face_embeddings.npy"
LABELS_PATH = "./model/face_labels.npy"


app = Flask(__name__)


@app.route("/embedding", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Face Recognition API"}), 200


@app.route("/embedding/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    faces, rgb_image = utils.detect_faces(request.files["image"], detector)

    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    recognized_faces = []
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    for detection in faces:
        x, y, w, h = detection["box"]

        face = rgb_image[y : y + h, x : x + w]
        face_resized = cv2.resize(face, RESIZE)
        face_embedding = embedder.embeddings([face_resized])[0]

        distances = np.linalg.norm(embeddings - face_embedding, axis=1)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < THRESHOLD:
            recognized_faces.append(labels[best_match_index])

        else:
            recognized_faces.append("Unknown")

    return jsonify({"faces": recognized_faces}), 200


@app.route("/model", methods=["GET"])
def model():
    return jsonify({"message": "Welcome to Face Recognition API"}), 200


@app.route("/model/detect", methods=["POST"])
def detect_faces():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    faces, rgb_image = utils.detect_faces(request.files["image"], detector)

    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    recognized_faces = []

    classifier = joblib.load("model/face_classifier.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")

    for i, detection in enumerate(faces):
        x, y, w, h = detection["box"]
        face = rgb_image[y : y + h, x : x + w]

        face_resized = cv2.resize(face, RESIZE)
        face_embedding = embedder.embeddings([face_resized])[0]

        predicted_label_encoded = classifier.predict([face_embedding])
        predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

        recognized_faces.append(predicted_label[0])

    return jsonify({"faces": recognized_faces}), 200


if __name__ == "__main__":
    app.run(debug=True)
