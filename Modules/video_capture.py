import cv2

def get_video_capture():
    return cv2.VideoCapture(0)

def release_and_destroy(cap):
    cap.release()
    cv2.destroyAllWindows()
