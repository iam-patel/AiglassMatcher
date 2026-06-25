import cv2
import numpy as np
from PIL import Image
import mediapipe as mp


# Load cascades
_face_cascade      = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade       = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
_left_eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
_right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

_mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
_mp_face_mesh      = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)


def detect_face(image_bgr):
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) > 0:
        return max(faces, key=lambda f: f[2] * f[3])
    return _detect_face_mediapipe(image_bgr)


def _detect_face_mediapipe(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = _mp_face_detection.process(rgb)
    if not results.detections:
        return None
    det = results.detections[0]
    bbox = det.location_data.relative_bounding_box
    ih, iw = image_bgr.shape[:2]
    x = int(bbox.xmin * iw)
    y = int(bbox.ymin * ih)
    w = int(bbox.width * iw)
    h = int(bbox.height * ih)
    if w <= 0 or h <= 0:
        return None
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.2)
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(iw - x, w + pad_w * 2)
    h = min(ih - y, h + pad_h * 2)
    return (x, y, w, h)


def detect_eyes(image_bgr, face_rect):
    x, y, w, h = face_rect
    roi_gray = cv2.cvtColor(image_bgr[y:y + int(h * 0.55), x:x + w], cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.equalizeHist(roi_gray)
    eyes = _eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    result = [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
    if len(result) < 2:
        result = []
        for cascade in [_left_eye_cascade, _right_eye_cascade]:
            detected = cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            if len(detected) > 0:
                ex, ey, ew, eh = detected[0]
                result.append((x + ex, y + ey, ew, eh))
    return result[:2]


def detect_eyes_mediapipe(frame, face_rect=None):
    if face_rect is not None:
        result = detect_eyes(frame, face_rect)
        if len(result) >= 2:
            return result

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return []

    landmarks = results.multi_face_landmarks[0].landmark
    ih, iw = frame.shape[:2]
    left_eye_idxs = [33, 133, 160, 159, 158, 157, 173, 153, 154, 155, 246]
    right_eye_idxs = [263, 362, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

    def _box(indices):
        xs = [landmarks[i].x * iw for i in indices]
        ys = [landmarks[i].y * ih for i in indices]
        x1 = int(max(0, min(xs)))
        y1 = int(max(0, min(ys)))
        x2 = int(min(iw, max(xs)))
        y2 = int(min(ih, max(ys)))
        if x2 <= x1 or y2 <= y1:
            return None
        pad = 4
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(iw, x2 + pad)
        y2 = min(ih, y2 + pad)
        return (x1, y1, x2 - x1, y2 - y1)

    left_box = _box(left_eye_idxs)
    right_box = _box(right_eye_idxs)
    if left_box and right_box:
        return [left_box, right_box]
    return []
