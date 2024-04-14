import json
import cv2
import numpy as np
import os
import glob

# Function to read COCO annotations and resize images
def preprocess_coco_annotations(annotations_file, image_dir, target_size):
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    
    image_data = []
    for img in images:
        image_id = img['id']
        file_name = img['file_name']
        image_path = os.path.join(image_dir, file_name)
        
        # Read and resize image
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Extract annotations for this image
        image_annotations = [anno for anno in annotations if anno['image_id'] == image_id]
        
        # Convert bounding box annotations to Faster R-CNN format
        bbox_annotations = []
        for anno in image_annotations:
            bbox = anno['bbox']
            bbox_annotations.append([anno['category_id'], bbox])
        
        image_data.append({'image': image, 'image_id': image_id, 'annotations': bbox_annotations})
    
    return image_data

# Function to read YOLO annotations and resize images
def preprocess_yolo_annotations(image_dir, label_dir, target_size):
    image_paths = glob.glob(os.path.join(image_dir, '*'))
    label_paths = glob.glob(os.path.join(label_dir, '*'))
    
    image_data = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        
        # Get corresponding label file
        label_file = os.path.join(label_dir, os.path.basename(img_path).replace('.png', '.txt'))
        # check if label file exists
        if not os.path.exists(label_file):
            continue

        # Read bounding box annotations from label file and convert to Deformable DETR format
        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                x1 = int((x_center - width / 2) * target_size[1])
                y1 = int((y_center - height / 2) * target_size[0])
                x2 = int((x_center + width / 2) * target_size[1])
                y2 = int((y_center + height / 2) * target_size[0])
                annotations.append([class_label, [x1, y1, x2, y2]])
        
        image_data.append({'image': image, 'image_name': os.path.basename(img_path), 'annotations': annotations})
    
    return image_data

# Example usage for COCO dataset
annotations_file = 'data_repo/mammo_1k/coco_1k/annotations/instances_train2017.json'
image_dir = 'data_repo/mammo_1k/yolo_1k/train/images'
target_size = (416, 416)  # Example target size for resizing
coco_data = preprocess_coco_annotations(annotations_file, image_dir, target_size)

# Example usage for YOLO dataset
image_dir = 'data_repo/mammo_1k/yolo_1k/train/images'
label_dir = 'data_repo/mammo_1k/yolo_1k/train/labels'
yolo_data = preprocess_yolo_annotations(image_dir, label_dir, target_size)


print(coco_data)
print(yolo_data)