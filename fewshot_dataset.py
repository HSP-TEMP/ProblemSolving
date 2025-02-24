import glob
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import OwlViTProcessor, OwlViTForObjectDetection, AdamW
import torch


class FewShotDetectionDataset(Dataset):
    """ Dataset class for OWL-ViT fine-tuning """
    def __init__(self,
                 dataset_folder: str,
                 processor: OwlViTProcessor) -> None:
        """ 
        Args:
            dataset_folder (str): Folder that contains image and corresponding annotation json files
            processor: OWL-ViT preprocessing object
        Returns:
            None
        """
        self.dataset_folder = dataset_folder
        self.image_paths = sorted(glob.glob(os.path.join(dataset_folder, '*.bmp')))
        self.json_paths  = sorted(glob.glob(os.path.join(dataset_folder, '*.json')))
        self.processor = processor
        assert len(self.image_paths) == len(self.json_paths), 'Image and annotation not matching!'

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        annotation_path = self.json_paths[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        # Single object in each image
        label = 'stabbed defect on metal bearing surface' # `STABBED`, data['shapes'][0]['label']
        bbox  = data['shapes'][0]['bbox']
        roi   = data['rois'][0] # list[int]
        
        roi_x = roi[0] # roi['x']
        roi_y = roi[1] # roi['y']
        roi_width  = roi[2] # roi['width']
        roi_height = roi[3] # roi['height']
        cropped_image = image.crop((roi_x, roi_y, roi_x + roi_width, roi_y + roi_height))

        if isinstance(bbox, dict):
            x_min = bbox['x']
            y_min = bbox['y']
            x_max = bbox['x'] + bbox['width']
            y_max = bbox['y'] + bbox['height']
            normalized_bbox = [
                x_min / width,
                y_min / height,
                x_max / width,
                y_max / height
            ]
            orig_bbox = [x_min, y_min, x_max, y_max]
            
            new_x_min = (bbox['x'] - roi_x) / roi_width
            new_y_min = (bbox['y'] - roi_y) / roi_height
            new_x_max = (bbox['x'] + bbox['width'] - roi_x) / roi_width
            new_y_max = (bbox['y'] + bbox['height'] - roi_y) / roi_height
            cropped_normalized_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]

        # inputs = self.processor(text=[label], images=image, return_tensors='pt')
        inputs = self.processor(text=[label], images=cropped_image, return_tensors='pt')
        pixel_values   = inputs['pixel_values'].squeeze(0) # Remove batch dim
        input_ids = inputs['input_ids'].squeeze(0)
        if input_ids is not None:
            input_ids = input_ids.squeeze(0)
        
        attention_mask = inputs['attention_mask'].squeeze(0)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)

        # Single category images => class index is 0
        target = {
            'labels': torch.tensor([0], dtype=torch.long),
            'boxes':  torch.tensor([cropped_normalized_bbox], dtype=torch.float)
        }
        return {
            'pixel_values': pixel_values,     # Image tensor for model (RGB, 3-channels)
            'input_ids': input_ids,           # Tokenized integer sequence by text (Optional)
            'attention_mask': attention_mask, # Mask for recognizing padding (Optional)
            'target': target                  # ground truth, 'labels=0' and 'normalized boxes'
        }


def collate_fn(batch: dict):
    """ Define collate function for data batching. """
    pixel_values   = torch.stack([item['pixel_values'] for item in batch])
    input_ids      = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    targets = [item['target'] for item in batch]
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'targets': targets
    }
