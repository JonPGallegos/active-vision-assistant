import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face mesh
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Right iris center (landmark 474)
            right_iris = face_landmarks.landmark[474]
            ix, iy = int(right_iris.x * w), int(right_iris.y * h)
            cv2.circle(frame, (ix, iy), 5, (255, 0, 255), -1)

            # Optional: draw left iris too (landmark 469)
            left_iris = face_landmarks.landmark[469]
            lx, ly = int(left_iris.x * w), int(left_iris.y * h)
            cv2.circle(frame, (lx, ly), 5, (0, 255, 255), -1)

            break  # Only use first face

    cv2.imshow("Eye & Gaze Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
