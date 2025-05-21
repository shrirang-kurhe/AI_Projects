import random
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
import tempfile

# 1️⃣ Load the pretrained model
model = YOLO("weights/yolo11n.pt", "v11")

# 2️⃣ Get COCO class names directly from the model
class_list = [model.names[i] for i in sorted(model.names)]

# 3️⃣ Generate a random BGR color for each class
detection_colors = [
    (b, g, r)
    for (r, g, b) in (
        tuple(random.randint(0,255) for _ in range(3)) for _ in class_list
    )
]

def detect_image(image):
    """Run YOLO on a single PIL image and return an annotated RGB image."""
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(source=[frame], conf=0.45)
    annotated = frame.copy()
    for box in results[0].boxes:
        clsID = int(box.cls.numpy()[0])
        conf  = float(box.conf.numpy()[0])
        x1, y1, x2, y2 = box.xyxy.numpy()[0].astype(int)
        color = detection_colors[clsID]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated,
            f"{class_list[clsID]} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def detect_video(video_path):
    """Run YOLO on a video file, write to temp .mp4, and return path for download."""
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out    = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=[frame], conf=0.45)
        for box in results[0].boxes:
            clsID = int(box.cls.numpy()[0])
            conf  = float(box.conf.numpy()[0])
            x1, y1, x2, y2 = box.xyxy.numpy()[0].astype(int)
            color = detection_colors[clsID]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"{class_list[clsID]} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        out.write(frame)

    cap.release()
    out.release()
    return temp_out


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# YOLOv11 Object Detection")

        with gr.Tab("Image Detection"):
            img_in  = gr.Image(type="pil", label="Upload Image")
            img_out = gr.Image(label="Detected Image")
            btn_i   = gr.Button("Detect")
            btn_i.click(detect_image, inputs=img_in, outputs=img_out)

        with gr.Tab("Video Detection"):
            vid_in  = gr.Video(label="Upload Video")
            vid_out = gr.File(label="Download Detected Video (.mp4)")
            btn_v   = gr.Button("Detect")
            btn_v.click(detect_video, inputs=vid_in, outputs=vid_out)

    demo.launch()

if __name__ == "__main__":
    main()
