import torchvision
import os
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import torchvision.transforms as T

import torch

# import gc
# gc.collect()
# torch.cuda.empty_cache()


##### Transformer DETR #########

def transfomer():



  b_size = 2
  test_id = 0
  deformable = False
  dino= False


  # def preprocess_coco(coco_dataset):


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

          # # check if class labels are empty
          # if target['class_labels'].shape[0] == 0:
          #     print("Class labels are empty")
          #     # replace the class labels with a tensor of zeros
          #     target['boxes'] = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
          #     target['class_labels'] = torch.tensor([1], dtype=torch.int64)

          return pixel_values, target


  from transformers import DetrImageProcessor
  from transformers import DetrFeatureExtractor
  from transformers import AutoImageProcessor
  from transformers import AutoModel

  from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

  from transformers import DeformableDetrForObjectDetection

  # standard PyTorch mean-std input image normalization
  transform = T.Compose([
      T.Resize(800),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  # processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

  # processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
  # processor = DetrImageProcessor( size = {'shortest_edge': 400, 'longest_edge': 660})
  # processor = DetrImageProcessor()
  processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
  # feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")


  ### DEFORMABLE DETR
  if dino==True:
      model_id = "IDEA-Research/grounding-dino-base"

      processor = AutoProcessor.from_pretrained(model_id)
      model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

      print("Loading Dino")
  elif deformable==True:
      # repo_name = "SenseTime/deformable-detr" 

      # # the Auto API automatically loads the appropriate class for us, based on the checkpoint
      # # in this case a DeformableDetrImageProcessor
      # processor = AutoImageProcessor.from_pretrained(repo_name)
      # model = DeformableDetrForObjectDetection.from_pretrained(repo_name)
      print("Loading Deformable Detr")
      processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr-with-box-refine-two-stage")
      model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr-with-box-refine-two-stage")



  # Initialize the COCO dataset
  img_folder = 'data/mammo_1k/coco_1k/train2017'
  ann_file = 'data/mammo_1k/coco_1k/annotations/instances_train2017.json'
  coco_dataset = CocoDetection(img_folder, ann_file, processor)

  img_folder_val = 'data/mammo_1k/coco_1k/val2017'
  ann_file_val = 'data/mammo_1k/coco_1k/annotations/instances_val2017.json'
  val_coco_dataset = CocoDetection(img_folder_val, ann_file_val, processor)


  # print("Number of images in COCO dataset:", len(coco_dataset))

  # print("COCO Dataset: ", type(coco_dataset))

  img_ids = val_coco_dataset.coco.getImgIds()
  # pick a random image id
  img_id = img_ids[test_id]
  # print("Image ID:", img_id)
  image = val_coco_dataset.coco.loadImgs(img_id)[0]
  # image = coco_dataset[0]
  # print("Image file name:", image['file_name'])
  image = Image.open(os.path.join(img_folder_val, image['file_name']))

  annotations = val_coco_dataset.coco.imgToAnns[img_id]
  # print(annotations)
  draw = ImageDraw.Draw(image, "RGBA")
  cats = val_coco_dataset.coco.cats
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


  # print("DATA 0:", coco_dataset[0])




  # img_id = img_ids[8]
  # print("Image ID:", img_id)
  # image = coco_dataset.coco.loadImgs(img_id)[0]
  # print("Image file name:", image['file_name'])
  # image = Image.open(os.path.join(img_folder, image['file_name']))

  # annotations = coco_dataset.coco.imgToAnns[img_id]
  # print(annotations)




  import torch
  # torch.set_float32_matmul_precision('high')

  # LOADING THE DATALOADER
  # take a subset of the dataset

  # # # modify coco_dataset[0] so that its class label is 1
  # # # coco_dataset[0][1]['class_labels'] = torch.tensor([1])
  # new_dict = coco_dataset[0][1]
  # new_dict['class_labels'] = torch.tensor([1], dtype=torch.int64)
  # # add a tensor of zeros of type torch.float32 to the boxes
  # new_dict['boxes'] = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
  # new_target = (coco_dataset[0][0], new_dict)
  # # coco_dataset= [new_target]


  # # print("DATA 0 After:", coco_dataset[0])

  # # add the data element with image_id:4 to the dataset
  # coco_dataset = [new_target, coco_dataset[4]]
  # print("DATA After:", coco_dataset)

  # # coco_dataset = torch.utils.data.Subset(coco_dataset, range(1))
  # val_coco_dataset = torch.utils.data.Subset(val_coco_dataset, range(1))



  # print("DATA 0 After:", coco_dataset[0])
    

  from torch.utils.data import DataLoader

  def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    # encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


  # Initialize the data loader
  data_loader = DataLoader(coco_dataset, batch_size=b_size, collate_fn=collate_fn)
  val_dataloader = DataLoader(val_coco_dataset, batch_size=b_size, collate_fn=collate_fn)


  # batch = next(iter(data_loader))



  import pytorch_lightning as pl
#   from torchmetrics import Accuracy

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

          #  self.model = DetrForObjectDetection.from_pretrained("eshan292/custom-deter")

          self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                              revision="no_timm",
                                                              num_labels=1,
                                                              ignore_mismatched_sizes=True).to(self.device)
          
          if dino==True:
              
              self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
              print("Training Dino")
          elif deformable==True:
              # self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr", num_labels=1, ignore_mismatched_sizes=True)
              self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr-with-box-refine-two-stage").to(self.device)
              print("Training Deformable Detr")
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
          # self.log("training_loss", loss, batch_size = b_size)
          self.log("training_loss", loss, prog_bar=True, on_epoch=True)
          # print the training loss
        

          for k,v in loss_dict.items():
            # self.log("train_" + k, v.item(), batch_size=b_size)
            self.log("train_" + k, v.item())
          #   print the training loss_dict
          # print("Training loss_dict:", loss_dict)
          

          return loss

      def validation_step(self, batch, batch_idx):
          loss, loss_dict = self.common_step(batch, batch_idx)
          # self.log("validation_loss", loss, batch_size = b_size)
          self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
          # print("Validation loss:", loss)
          

          for k,v in loss_dict.items():
            # self.log("validation_" + k, v.item(), batch_size=b_size)
            self.log("validation_" + k, v.item())
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
          global data_loader
          return data_loader

      def val_dataloader(self):
          global val_dataloader
          return val_dataloader
      




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


  # Start tensorboard.
  # %load_ext tensorboard
  # %tensorboard --logdir lightning_logs/



#### TRAINING STARTS #####

  import time
  from pytorch_lightning import Trainer
  import tensorflow as tf

  start = time.time()



  model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-3)
  model.train() 

  # outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
  # outputs.logits.shape


  # # trainer = Trainer(gradient_clip_val=0.1, log_every_n_steps=50, max_epochs=30, accelerator="gpu", devices=1, accumulate_grad_batches=8) 
  # trainer = Trainer(gradient_clip_val=0.1, log_every_n_steps=30, max_epochs=50, accelerator="gpu", devices=[1], accumulate_grad_batches=8) 
  # trainer.fit(model, train_dataloaders=data_loader, val_dataloaders=val_dataloader)


  # # # model.model.push_to_hub("eshan292/custom_detr_only0s", private = True)


  end = time.time()

  print("Time taken for training :", end - start)



##### TRAINING ENDS #####






  # Load the model from ckpt file
  # model = Detr.load_from_checkpoint("lightning_logs/version_21/checkpoints/epoch=99-step=14000.ckpt", lr = 1e-4, lr_backbone = 1e-4, weight_decay = 1e-4)
  # model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm", num_labels=len(id2label), ignore_mismatched_sizes=True)
  # model = DetrForObjectDetection.from_pretrained("eshan292/custom_detr_only0s")
  model = DetrForObjectDetection.from_pretrained("/Users/eshan/Main/OneDrive - IIT Delhi 2/Eshan/IITD/Sem-8/COL780-Computer Vision/COL780_A4/custom_detr_only0s")
  

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()




  # Visualise the results


  # Eval the model on an image from the validation set
  # coco_dataset = CocoDetection(img_folder, ann_file, processor)
  img_folder_val = 'data/mammo_1k/coco_1k/val2017'
  ann_file_val = 'data/mammo_1k/coco_1k/annotations/instances_val2017.json'
  coco_dataset = CocoDetection(img_folder_val, ann_file_val, processor)

  val_img, val_target = coco_dataset[test_id]

  # print image details
  print("Image Id:", val_target["image_id"])
  val_img = val_img.unsqueeze(0).to(device)
  val_target = val_target.to(device)



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
          # text = f'{model.config.id2label[label]}: {score:0.2f}'
          text = f'{label}: {score:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
      plt.axis('off')
      plt.show()
      # save the image
      plt.savefig("output.png")

  image_id = val_target["image_id"].item()
  # print("Image Id: ", image_id)
  # image = val_coco_dataset.coco.loadImgs(image_id)[0]
  image = coco_dataset.coco.loadImgs(image_id)[0]
  image = Image.open(os.path.join(img_folder_val, image['file_name']))
  # image = Image.open(os.path.join(img_folder, image['file_name']))

  # postprocess the results
  width, height = image.size
  # print("Image size:", width, height)

  # print("Number of logits in output:", len(outputs["logits"]))
  target_sizes = [(height, width) for _ in range(val_img.shape[0])]
  post_processed_output = processor.post_process_object_detection(outputs, target_sizes=[(height, width)], threshold=0.1)

  results = post_processed_output[0]

  for score, label, box in zip(results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist()):
      box = [round(i, 2) for i in box]
      # print( f"Detected {model.config.id2label[label]} with confidence {score} at location {box}")
      print( f"Detected label {label} with confidence {score} at location {box}")
      
  # perform nms
  results = nms(results, iou_threshold=0)

  # print(results)
  plot_results(image, results["scores"], results["labels"], results["boxes"])




  # # EVALUATE

  # # filter the coco_val dataset to take only the first 10 images
  # # val_coco_dataset = torch.utils.data.Subset(val_coco_dataset, range(10))


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
  #     results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0.2) 

  #     predictions = {target["image_id"].item(): prediction for target, prediction in zip(labels, results)}
  #     predictions = prepare_for_coco_detection(predictions)
  #     evaluator.update(predictions)

  # # gather the results
  # evaluator.synchronize_between_processes()
  # evaluator.accumulate()
  # evaluator.summarize()

# transfomer()



from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def yolo():
    

    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("model.pt")  # load a pretrained model (recommended for training)

    # dict_classes = model.model.names


    # Use the model
    model.train(data="yolo_data.yaml", epochs=300, dropout=0.2, plots=True, single_cls=True, imgsz = 800, device =0,lr0=0.001, box=10, translate= 0.2, degrees = 30, scale=0.6, flipud = 0.1, fliplr =0.1, erasing=0.4)  # train the model

    # # metrics = model.val()  # evaluate model performance on the validation set

    # # path = model.export(format="onnx")  # export the model to ONNX format




    # results = model(["/home/jain/data/mammo_1k/coco_1k/val2017/Calc-Training_P_01250_LEFT_CC.png"], conf = 0.005, iou = 0, device = 0,show=True, save=True)  # predict on an image

    # # Process results list
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     result.show()  # display to screen
    #     result.save(filename='result.jpg')  # save to disk





    # # # Customize validation settings
    # # validation_results = model.val(data='data.yaml',
    # #                                imgsz=640,
    # #                                batch=1,
    # #                                conf=0.25,
    # #                                iou=0.6,
    # #                                device='0')






    