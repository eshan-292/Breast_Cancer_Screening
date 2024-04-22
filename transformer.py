import torchvision
import os
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torchvision.transforms as T

import torch

import gc
gc.collect()
torch.cuda.empty_cache()

b_size = 2

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


from transformers import DetrImageProcessor
from transformers import DetrFeatureExtractor
from transformers import AutoImageProcessor

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
# processor = DetrImageProcessor()
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

# Initialize the COCO dataset
img_folder = 'data/mammo_1k/coco_1k/train2017'
ann_file = 'data/mammo_1k/coco_1k/annotations/instances_train2017.json'
coco_dataset = CocoDetection(img_folder, ann_file, feature_extractor)

img_folder_val = 'data/mammo_1k/coco_1k/val2017'
ann_file_val = 'data/mammo_1k/coco_1k/annotations/instances_val2017.json'
val_coco_dataset = CocoDetection(img_folder_val, ann_file_val, feature_extractor)


print("Number of images in COCO dataset:", len(coco_dataset))

print(coco_dataset)

img_ids = coco_dataset.coco.getImgIds()
# pick a random image id
img_id = img_ids[8]
print("Image ID:", img_id)
image = coco_dataset.coco.loadImgs(img_id)[0]
print("Image file name:", image['file_name'])
image = Image.open(os.path.join(img_folder, image['file_name']))

annotations = coco_dataset.coco.imgToAnns[img_id]
print(annotations)
draw = ImageDraw.Draw(image, "RGBA")
cats = coco_dataset.coco.cats
# 0 corresponds to the tumor present and 1 corresponds to the tumor absent
id2label = {0: "tumor present"}
for anno in annotations:
    bbox = anno['bbox']
    label = id2label[anno['category_id']]
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    draw.text((x, y), label, fill="red")
# save the image
image.save("image.png")



# img_id = img_ids[8]
# print("Image ID:", img_id)
# image = coco_dataset.coco.loadImgs(img_id)[0]
# print("Image file name:", image['file_name'])
# image = Image.open(os.path.join(img_folder, image['file_name']))

# annotations = coco_dataset.coco.imgToAnns[img_id]
# print(annotations)




import torch

# LOADING THE DATALOADER
# take a subset of the dataset
# coco_dataset = torch.utils.data.Subset(coco_dataset, range(50))
# val_coco_dataset = torch.utils.data.Subset(val_coco_dataset, range(50))


from torch.utils.data import DataLoader

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  # encoding = processor.pad(pixel_values, return_tensors="pt")
  encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch


# Initialize the data loader
data_loader = DataLoader(coco_dataset, batch_size=b_size, collate_fn=collate_fn)
val_dataloader = DataLoader(val_coco_dataset, batch_size=b_size, collate_fn=collate_fn)

batch = next(iter(data_loader))
print(batch.keys())
# print the batch
print(batch)
# print(batch['pixel_values'].shape)
# print(batch['pixel_mask'].shape)

# pixel_values , target = coco_dataset[0]
# print(pixel_values.shape)
# print(target)


# # write the data loader into a file
# with open("data_loader.txt", "w") as f:
#     # iterate over the data loader
#     for batch in data_loader:
#         # write the batch to the file
#         f.write(str(batch))
#         f.write("\n")
# i=0
# for batch in data_loader:
#     i+=1
#     # for each batch print the labels
#     print("Labels for Batch:", i, ":", batch['labels'])
   


import pytorch_lightning as pl
from torchmetrics import Accuracy

from transformers import DetrForObjectDetection

from transformers import DetrConfig

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
        #  self.config = DetrConfig(use_pretrained_backbone=False)
        
        #  self.model = DetrForObjectDetection(config=self.config)
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, batch_size = b_size)
        # print the training loss
        print("Training loss:", loss)

        # acc = Accuracy(task="binary")
        # acc(outputs.logits, outputs.labels)
        # # print the first 10 predictions
        # print("Predictions:", outputs.logits)
        # labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        # print("Labels:", labels)
        
        
        # print("Accuracy:", acc)
        # # log the accuracy
        # self.log("training_accuracy", acc, batch_size = b_size)


        for k,v in loss_dict.items():
          self.log("train_" + k, v.item(), batch_size=b_size)
        #   print the training loss_dict
        # print("Training loss_dict:", loss_dict)
         

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, batch_size = b_size)
        print("Validation loss:", loss)
        # # log the accuracy 
        # acc = Accuracy(task="binary")
        # acc(outputs.logits, outputs.labels)
        # print("Accuracy:", acc)
        # self.log("validation_accuracy", acc, batch_size = b_size)


        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item(), batch_size=b_size)
        # print("Validation loss_dict:", loss_dict)
        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return data_loader

     def val_dataloader(self):
        return val_dataloader
     


# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/


import time
from pytorch_lightning import Trainer
import tensorflow as tf

start = time.time()

# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# # instantiate a distribution strategy
# tf.tpu.experimental.initialize_tpu_system(tpu)
# tpu_strategy = tf.distribute.TPUStrategy(tpu)

# # instantiating the model in the strategy scope creates the model on the TPU
# with tpu_strategy.scope():



model = Detr(lr=1e-4, lr_backbone=1e-4, weight_decay=1e-2)

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
outputs.logits.shape



# trainer = Trainer(gradient_clip_val=0.1, log_every_n_steps=50, max_epochs=50)
# trainer.fit(model)
        
end = time.time()

print("Time taken for training :", end - start)



# Load the model from ckpt file
model = Detr.load_from_checkpoint("lightning_logs/version_7/checkpoints/epoch=49-step=28000.ckpt", lr = 1e-1, lr_backbone = 1e-1, weight_decay = 1e-2)
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm", num_labels=len(id2label), ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Visualise the results


# Eval the model on an image from the validation set
val_img, val_target = coco_dataset[230]
# print image details
print("Image Id:", val_target["image_id"])
val_img = val_img.unsqueeze(0).to(device)
val_target = val_target.to(device)
print("Shape of pixel_values:", val_img.shape)


with torch.no_grad():
    outputs = model(val_img, pixel_mask=None)
print(outputs.keys())

import matplotlib.pyplot as plt

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    # save the image
    plt.savefig("output.png")

image_id = val_target["image_id"].item()
print("Image Id: ", image_id)
# image = val_coco_dataset.coco.loadImgs(image_id)[0]
image = coco_dataset.coco.loadImgs(image_id)[0]
# image = Image.open(os.path.join(img_folder_val, image['file_name']))
image = Image.open(os.path.join(img_folder, image['file_name']))

# postprocess the results
width, height = image.size
print("Image size:", width, height)

# Converts the raw output of DetrForObjectDetection into final bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
# post_process_object_detection
# ( outputsthreshold: float = 0.5target_sizes: Union = None ) → List[Dict]
# Parameters
# outputs (DetrObjectDetectionOutput) — Raw outputs of the model.
# threshold (float, optional) — Score threshold to keep object detection predictions.
# target_sizes (torch.Tensor or List[Tuple[int, int]], optional) — Tensor of shape (batch_size, 2) or list of tuples (Tuple[int, int]) containing the target size (height, width) of each image in the batch. If unset, predictions will not be resized.
# Returns
# List[Dict]
# target sizes is a list of tuples containing the target size (height, width) of each image in the batch
# print number of logits in output
print("Number of logits in output:", len(outputs["logits"]))
target_sizes = [(height, width) for _ in range(val_img.shape[0])]
post_processed_output = processor.post_process_object_detection(outputs, target_sizes=[(height, width)], threshold=0)

results = post_processed_output[0]

for score, label, box in zip(results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist()):
    box = [round(i, 2) for i in box]
    # print( f"Detected {model.config.id2label[label]} with confidence {score} at location {box}")
    print( f"Detected label {label} with confidence {score} at location {box}")
    

print(results)
# plot_results(image, results["scores"], results["labels"], results["boxes"])










# EVALUATE

# filter the coco_val dataset to take only the first 10 images
# val_coco_dataset = torch.utils.data.Subset(val_coco_dataset, range(10))


# def convert_to_xywh(boxes):
#     xmin, ymin, xmax, ymax = boxes.unbind(1)
#     return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

# def prepare_for_coco_detection(predictions):
#     coco_results = []
#     for original_id, prediction in predictions.items():
#         if len(prediction) == 0:
#             continue

#         boxes = prediction["boxes"]
#         boxes = convert_to_xywh(boxes).tolist()
#         scores = prediction["scores"].tolist()
#         labels = prediction["labels"].tolist()

#         coco_results.extend(
#             [
#                 {
#                     "image_id": original_id,
#                     "category_id": labels[k],
#                     "bbox": box,
#                     "score": scores[k],
#                 }
#                 for k, box in enumerate(boxes)
#             ]
#         )
#     return coco_results
     
    
# from coco_eval import CocoEvaluator
# from tqdm import tqdm
# import numpy as np


# evaluator = CocoEvaluator(coco_gt=val_coco_dataset.coco, iou_types=["bbox"])

# print("Evaluating the model on the validation set")
# for idx, batch in enumerate(tqdm(val_dataloader)):
#     pixel_values = batch["pixel_values"].to(device)
#     pixel_mask = batch["pixel_mask"].to(device)
#     labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
#     with torch.no_grad():
#         outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
#     # turn into a list of dictionaries (one item for each example in the batch)
#     orig_target_sizes = torch.stack([t["orig_size"] for t in batch["labels"]], dim=0)
#     results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0) 

#     predictions = {target["image_id"].item(): prediction for target, prediction in zip(labels, results)}
#     predictions = prepare_for_coco_detection(predictions)
#     evaluator.update(predictions)

# # gather the results
# evaluator.synchronize_between_processes()
# evaluator.accumulate()
# evaluator.summarize()

