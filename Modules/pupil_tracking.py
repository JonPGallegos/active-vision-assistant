import cv2

def track_pupil(gray_eye_roi, color_eye_roi):
    # Threshold to isolate the pupil (dark region)
    _, thresh = cv2.threshold(gray_eye_roi, 70, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        (x_c, y_c, w_c, h_c) = cv2.boundingRect(contour)
        cv2.rectangle(color_eye_roi, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 0, 255), 2)
