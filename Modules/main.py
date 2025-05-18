import cv2
from video_capture import get_video_capture, release_and_destroy
from face_detection import detect_faces, draw_faces
from eye_detection import detect_eyes, draw_eyes
from pupil_tracking import track_pupil

def main():
    cap = get_video_capture()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray)
        draw_faces(frame, faces)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = detect_eyes(roi_gray)
            draw_eyes(roi_color, eyes)

            for (ex, ey, ew, eh) in eyes:
                eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_color = roi_color[ey:ey+eh, ex:ex+ew]
                track_pupil(eye_gray, eye_color)

        cv2.imshow('Face, Eye, and Pupil Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_and_destroy(cap)

if __name__ == '__main__':
    main()
