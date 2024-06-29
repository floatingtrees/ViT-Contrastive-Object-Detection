from collections import defaultdict
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision


class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license
    def get_imgIds(self):
        return list(self.im_dict.keys())
    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        
    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]
    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]
    


class BoxLoader(Dataset):
    def __init__(self, anns_file, imgs_dir):
        self.coco = COCOParser(anns_file, imgs_dir)
        self.imgs_dir = imgs_dir
        self.preproc = v2.Compose([
            v2.PILToTensor(),
            torchvision.transforms.Resize(size=(1024, 1024), antialias=True),
            ])

    def __len__(self):
        return len(self.coco.get_imgIds())
    
    def __getitem__(self, idx):
        img_ids = self.coco.get_imgIds()
        selected_img_ids = img_ids[idx]

        #ann_ids = self.coco.get_annIds(selected_img_ids)

        image = Image.open(f"{self.imgs_dir}/{str(selected_img_ids).zfill(12)}.jpg")
        ann_ids = self.coco.get_annIds(selected_img_ids)
        annotations = self.coco.load_anns(ann_ids)
        boxes = torch.zeros((len(annotations), 4))
        encodings = torch.zeros((len(annotations), 512))
        exit()
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']
            x, y, w, h = [int(b) for b in bbox]
            boxes[i, 0] = x
            boxes[i, 0] = y
            boxes[i, 0] = x + w
            boxes[i, 0] = y + h
            class_id = ann["category_id"]
            class_name = self.coco.load_cats(class_id)[0]["name"]
        return self.preproc(image), boxes



    
coco_annotations_file="../../coco2017/annotations/instances_train2017.json"
coco_images_dir="../../coco2017/train2017"
dataset = BoxLoader(coco_annotations_file, coco_images_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for x, y in dataloader:
    print(x.shape, y)
    exit()














exit()
import matplotlib.pyplot as plt
from PIL import Image
# define a list of colors for drawing bounding boxes
color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10
num_imgs_to_disp = 1
total_images = len(coco.get_imgIds()) # total number of images
sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]
img_ids = coco.get_imgIds()
selected_img_ids = [img_ids[i] for i in sel_im_idxs]
ann_ids = coco.get_annIds(selected_img_ids)
im_licenses = coco.get_imgLicenses(selected_img_ids)

for i, im in enumerate(selected_img_ids):
    image = Image.open(f"{coco_images_dir}/{str(im).zfill(12)}.jpg")
    draw = ImageDraw.Draw(image)
    ann_ids = coco.get_annIds(im)
    annotations = coco.load_anns(ann_ids)
    for ann in annotations:
        bbox = ann['bbox']
        print(len(bbox))
        print(bbox)
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = coco.load_cats(class_id)[0]["name"]
        license = coco.get_imgLicenses(im)[0]["name"]
        color_ = color_list[class_id]

        box_coords = (x, y, w+ x, h + y)
        draw.rectangle(box_coords, outline=color_, width=2)
        _, _, text_width, text_height = draw.textbbox((0, 0), text = class_name)
        if y - text_height >= 0:
            text_position = (x + 1, y - text_height - 2)
        else:
            text_position = (x + 1, y)
        
        background_coords = (text_position[0], text_position[1], text_position[0] + text_width + 2, text_position[1] + text_height)
        draw.rectangle(background_coords, fill="white", outline = "blue")
        draw.text(text_position, class_name, fill = "red")
image.save("image.jpg")