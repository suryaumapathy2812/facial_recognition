import numpy as np
import cv2
from mtcnn import MTCNN


def detect_faces(file, detector):
  image = np.fromstring(file.read(), np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  faces = detector.detect_faces(rgb_image)
  
  if len(faces) == 0:
    return None
  
  return faces, rgb_image;