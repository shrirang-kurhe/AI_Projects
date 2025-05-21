import gradio as gr
import cv2
import numpy as np

# Load Haar cascades
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_faces_eyes(image):
    """
    image: a NumPy array in RGB format (from Gradio)
    returns: annotated RGB image
    """
    # Convert to BGR for OpenCV
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame, "Face", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )

        # Within each face, detect eyes
        roi_gray  = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )
            cv2.putText(
                roi_color, "Eye", (ex, ey - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
            )

    # Convert back to RGB for Gradio display
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Build and launch the Gradio interface
demo = gr.Interface(
    fn=detect_faces_eyes,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(label="Detected Faces & Eyes"),
    title="Face & Eye Detection"
)

if __name__ == "__main__":
    demo.launch()
