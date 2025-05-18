import cv2

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_eyes(gray_face_roi):
    return eye_cascade.detectMultiScale(gray_face_roi)

def draw_eyes(color_face_roi, eyes):
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(color_face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
