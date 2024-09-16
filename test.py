#### TRANSFORMERS
import torchvision
import os
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torchvision.transforms as T

import torch

import gc
gc.collect()
torch.cuda.empty_cache()

from transformers import DetrImageProcessor
import pytorch_lightning as pl
from transformers import DetrForObjectDetection


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os


# implement NMS (Non maximal supression on the results)
def nms(results, iou_threshold=0.5):
    # get the boxes and scores
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]
    # get the indices of the boxes
    indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    # get the filtered boxes, scores and labels
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    filtered_labels = labels[indices]
    return {"boxes": filtered_boxes, "scores": filtered_scores, "labels": filtered_labels}


# accepting inputs of an image folder and trained model, runs the model on all images in the folder, saving predictions in text files.
def test_model_transformer(image_dir, model_path, output_dir):
    # Load the model
    model = DetrForObjectDetection.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

    # process the images to get the pixel values 
    img_folder = image_dir
    for img in os.listdir(img_folder):
        img_name = img.split(".")[0]
        # load the image
        image = Image.open(os.path.join(img_folder, img))
        # preprocess the image
        encoding = processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].to(device)
        # run the model
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
        # postprocess the results
        results = processor.post_process_object_detection(outputs, target_sizes=[(image.size[1], image.size[0])], threshold=0)
        results = results[0]
        # perform nms
        results = nms(results, iou_threshold=0)
        # save the results in a text file named image_name_preds.txt
        output_file_path = os.path.join(output_dir, f"{img_name}_preds.txt")
        with open(output_file_path, "w") as f:
            # f.write(f"Results for image {img}:\n")
            # Each line in the text file represents a bounding box in YOLO format (center x, center y, width, height, confidence score)
            for score, label, box in zip(results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist()):
                box = [round(i, 2) for i in box]
                # store the bounding box in YOLO format (center x, center y, width, height, score) which are currently stores as (top_left_x, top_left_y, bottom_right_x, bottom_right_y) 
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]
                # YOLO format of bounding box: Each line the annotation file has 5 values, for example: ”0 0.25 0.3 0.5 0.4”. Here, ”0” is the object class label, and the numbers represent the normalized coordinates of the bounding box: (x,y,w,h). These coordinates are normalized to the dimensions of the image, with (x,y) representing the center of the bounding box, and (w,h) representing its width and height.
                center_x_normalized = center_x / image.size[0]
                center_y_normalized = center_y / image.size[1]
                width_normalized = width / image.size[0]
                height_normalized = height / image.size[1]
                f.write(f"{center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized} {score}\n")
                # f.write(f"{center_x} {center_y} {width} {height} {score}\n")
    print("Predictions saved in text files in the output directory")




def test_model_yolo(image_dir, model_path, output_dir):
    model = YOLO(model_path)
    img_list = os.listdir(image_dir)

    for img in img_list:
        img_path = os.path.join(image_dir, img)
        # results = model(img_path)
        # results = model.predict(img_path, conf = 0, iou = 0,imgsz = 800, device =0)
        results = model.predict(img_path, conf = 0, iou = 0)
        for result in results:
            boxes = result.boxes
            labels = result.names
            probs = result.probs

            # read the boxes as Tensor(x,y,w,h)
            scores = boxes.conf.to("cpu").numpy()
            (image_height, image_width) = result.orig_shape
            boxes = boxes.xywh.to("cpu").numpy()
            

            # save the results in a text file named image_name_preds.txt
            # remove .png  from the image name
            img = img.split(".")[0]
            output_file_path = os.path.join(output_dir, f"{img}_preds.txt")
            with open(output_file_path, "w") as f:
                for i in range(len(boxes)):
                    # boxes are in x,y,w,h format, calculate the center_x, center_y, width, height normalized by the image size in YOLO format
                    center_x = boxes[i][0] + boxes[i][2] / 2
                    center_y = boxes[i][1] + boxes[i][3] / 2
                    width = boxes[i][2]
                    height = boxes[i][3]
                    center_x_normalized = center_x / image_width
                    center_y_normalized = center_y / image_height
                    width_normalized = width / image_width
                    height_normalized = height/ image_height
                    f.write(f"{center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized} {scores[i]}\n")



# test_model_transformer("data/sample_test/test/images", "eshan292/custom_detr_only0s", "output")
# test_model_yolo("data/sample_test/test/images", "/home/jain/yolo/runs/detect/train2/weights/best.pt", "output")

import sys
# main run
if __name__ == "__main__":
    # read the arguments
    #  first argument is the 'yolo' or 'detr'
    # second argument is the image folder
    # third argument is the model path
    # fourth argument is the output directory
    task = sys.argv[1]
    image_dir = sys.argv[2]
    model_path = sys.argv[3]
    output_dir = sys.argv[4]

    print(task)
    if task == "yolo":
        test_model_yolo(image_dir, model_path, output_dir)
    elif task == "transformer":
        test_model_transformer(image_dir, model_path, output_dir)
    else:
        print("Invalid task. Please provide either 'yolo' or 'transformer' as the first argument")
        sys.exit(1)


# Sample Command
# python3.10 test.py yolo '/Users/eshan/Main/OneDrive - IIT Delhi 2/Eshan/IITD/Sem-8/COL780-Computer Vision/COL780_A4/data_repo/sample_test/test/images' yolo_best.pt ../output 
