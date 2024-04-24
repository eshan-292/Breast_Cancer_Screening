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
def test_model(image_dir, model_path, output_dir):
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
        results = processor.post_process_object_detection(outputs, target_sizes=[(image.size[1], image.size[0])], threshold=0.2)
        results = results[0]
        # perform nms
        results = nms(results, iou_threshold=0.01)
        # save the results in a text file named image_name_preds.txt
        output_file_path = os.path.join(output_dir, f"{img_name}_preds.txt")
        with open(output_file_path, "a") as f:
            # f.write(f"Results for image {img}:\n")
            # Each line in the text file represents a bounding box in YOLO format (center x, center y, width, height, confidence score)
            for score, label, box in zip(results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist()):
                box = [round(i, 2) for i in box]
                # store the bounding box in YOLO format (center x, center y, width, height, confidence score) which are currently stores as (top_left_x, top_left_y, bottom_right_x, bottom_right_y) 
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]
                f.write(f"{center_x} {center_y} {width} {height} {score}\n")
    print("Predictions saved in text files in the output directory")
    
