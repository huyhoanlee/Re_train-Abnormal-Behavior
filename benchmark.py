from Inference import Inference
import cv2
import torch
from Param import DEVICE
import cv2
import torch
import os
# Load the model
model = torch.load("model/Weight").to(DEVICE)
count = 0
# Specify the image path directly
directory_path_abnormal = 'TEST_DATA/Abnormal'
directory_path_normal = 'TEST_DATA/Normal'
#'Data/SplitData/test/Abnormal'
abnormal_count= 0
normal_count = 0 
FRAME_COUNT = 0
TRUE_NEGATIVES = 0
TRUE_POSITIVES = 0
FALSE_POSITIVES = 0
FALSE_NEGATIVES = 0
TRUE_PREDICTIONS = 0

for filename in os.listdir(directory_path_abnormal):
    image_path = os.path.join(directory_path_abnormal, filename)

    #frame = cv2.imread(image_path)
    label = Inference(model, image_path)
    if label == "Abnormal":
        TRUE_POSITIVES += 1
        TRUE_PREDICTIONS += 1
        abnormal_count += 1
    else:
        FALSE_POSITIVES += 1

    count += 1

    print(f"Inference result for {image_path}: {label}")

for filename in os.listdir(directory_path_normal):
    image_path = os.path.join(directory_path_normal, filename)

    #frame = cv2.imread(image_path)
    label = Inference(model, image_path)

    if label == "Normal":
        TRUE_NEGATIVES += 1
        TRUE_PREDICTIONS += 1
        normal_count += 1
    else:
        FALSE_NEGATIVES += 1
    count += 1
    
    print(f"Inference result for {image_path}: {label}")



print(f"Number images: {count}")
print(f"Number true prediction: {TRUE_PREDICTIONS}")
print(f"Total number of true 'Abnormal' images: {abnormal_count}")
print(f"Total number of true 'Normal' images: {normal_count}")

PRECISION = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_POSITIVES)
RECALL = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES)
F1_SCORE = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
ACCURACY = (TRUE_POSITIVES + TRUE_NEGATIVES) / (TRUE_POSITIVES + TRUE_NEGATIVES + FALSE_POSITIVES + FALSE_NEGATIVES)


print("Precision: {:.2%}".format(PRECISION))
print("Recall: {:.2%}".format(RECALL))
print("F1 Score: {:.2%}".format(F1_SCORE))
print("Accuracy: {:.2%}".format(ACCURACY))

