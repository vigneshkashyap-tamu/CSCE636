import json
import os
import random

from torch.utils.data import Dataset
import torchvision

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index, enable_transform=True):    
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])

        image = Image.open(image_path).convert('RGB')   

        if enable_transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']], index
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        # HACK if is coco dataset, append COCO_val2014_ to image path
        self.is_coco = "coco" in ann_file

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []

            if type(ann['caption']) == list:  # for coco and flickr datasets
                for i, caption in enumerate(ann['caption']):
                    self.text.append(pre_caption(caption, self.max_words))
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1

            elif type(ann['caption']) == str: # for sbu dataset
                self.text.append(pre_caption(ann['caption'], self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

            else:
                assert 0
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index, enable_transform=True):    
        if self.is_coco:
            path = self.ann[index]['image'].split("/")[0] + "/COCO_val2014_" + self.ann[index]['image'].split("/")[-1]
        else:
            path = self.ann[index]['image']
        image_path = os.path.join(self.image_root, path)        
        image = Image.open(image_path).convert('RGB')    
        
        if enable_transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)

        return image, index
      
