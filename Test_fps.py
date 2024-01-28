from Inference import Inference
import cv2
import torch
from Param import DEVICE
import argparse
parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--source', type=str, default='0', help='Path to save the trained model')
args = parser.parse_args()

model = torch.load("model/Weight").to(DEVICE)

font_color = {"Normal" : (0, 255, 0),
              "Abnormal": (0,0,255)}

font = cv2.FONT_HERSHEY_SIMPLEX
org_label = (50, 50)
org_fps = (50, 100)
font_scale = 1
thickness = 2

op = args.source
if op == "0":
    op = 0
cap = cv2.VideoCapture(op)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()
    
import time
# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    label = Inference(model,frame)
    frame = cv2.putText(frame, label, org_label, font, font_scale, font_color[label], thickness, cv2.LINE_AA)
    # Increment frame count
    frame_count += 1
    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps_calculated = frame_count / elapsed_time
    fps_text = f'FPS: {fps_calculated:.2f}'
    frame = cv2.putText(frame, fps_text, org_fps, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    cv2.imshow('Camera Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# import time
# start_time = time.time() # Multiply by 1000 to get milliseconds

# frame = cv2.imread("image_test.jpg")
# label = Inference(model,frame)
# print(label)

# #time.sleep(0.1)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time:.10f}seconds")
# frame = cv2.putText(frame, label, org, font, font_scale, font_color[label], thickness, cv2.LINE_AA)
# cv2.imshow('Camera Stream', frame)
# cv2.waitKey(0)

cv2.destroyAllWindows()


