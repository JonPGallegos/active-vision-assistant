############## Enhanced Eye Detection with Pupil Tracking ##############
import cv2

# Constants
DEBUG_MODE = True               # Toggle to enable/disable visual debugging
SCALE_FACTOR = 0.5              # Resize frame for performance
MIN_PUPIL_AREA = 100            # Minimum contour area to qualify as a pupil

# Load the face and eye Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Define regions of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # Use adaptive thresholding for robust pupil detection
            thresh = cv2.adaptiveThreshold(
                eye_gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                11, 3
            )

            # Find contours in the thresholded eye image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Draw the largest valid contour (assumed to be pupil)
            for cnt in contours:
                if cv2.contourArea(cnt) > MIN_PUPIL_AREA:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    if DEBUG_MODE:
                        cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                    break  # Only draw one contour (the best match)

    if DEBUG_MODE:
        cv2.imshow('Pupil Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
