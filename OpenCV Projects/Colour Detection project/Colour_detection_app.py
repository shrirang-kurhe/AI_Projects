import cv2
import numpy as np
import gradio as gr

def detect_color_except_white(input_img):
    # Convert to HSV
    hsv_frame = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    
    # Define HSV range to exclude white
    low = np.array([0, 42, 0]) 
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    
    # Apply the mask
    result = cv2.bitwise_and(input_img, input_img, mask=mask)
    
    # Convert BGR to RGB for display
    input_img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return input_img_rgb, result_rgb

# Gradio interface
gr.Interface(
    fn=detect_color_except_white,
    inputs=gr.Image(type="numpy", label="Upload or Capture Image"),
    outputs=[
        gr.Image(label="Original Image"),
        gr.Image(label="Detected Colors (excluding White)")
    ],
    title="Color Detection (Excluding White)",
    description="Upload or capture an image. This app will highlight all colors except white."
).launch()
