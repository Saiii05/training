import torch
import torch.utils.data
import torchvision
from torchvision.datasets import VOCDetection
from PIL import Image
import xml.etree.ElementTree as ET
import os

# Assuming transforms.py is in the same directory (src)
from transforms import get_transform

PASCAL_VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

class PascalVOCDataset(torch.utils.data.Dataset):
    """
    Pascal VOC Dataset for Object Detection.
    """
    def __init__(self, root_dir, year="2012", image_set="train", transform=None, download=True):
        """
        Args:
            root_dir (str): Root directory for the VOCdevkit data.
                            e.g., './object_detector/data/' which will contain 'VOCdevkit'.
            year (str): Dataset year (e.g., "2012").
            image_set (str): "train", "val", or "trainval".
            transform (callable, optional): Optional transform to be applied on a sample.
            download (bool): If true, downloads the dataset from the internet and
                             puts it in root_dir/VOCdevkit. If dataset is already downloaded,
                             it is not downloaded again.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_set = image_set
        self.year = year

        # Ensure the root directory for VOCDetection is where VOCdevkit will be/is located
        # VOCDetection expects root to be the parent of VOCdevkit
        self.voc_data = VOCDetection(
            root=self.root_dir,  # e.g. ./object_detector/data
            year=self.year,
            image_set=self.image_set,
            download=download,
            # transform=None, # We apply transforms manually to image and target
            # target_transform=None # We parse target manually
        )

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(PASCAL_VOC_CLASSES)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(PASCAL_VOC_CLASSES)}


    def __len__(self):
        return len(self.voc_data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data sample.

        Returns:
            tuple: (image, target) where image is the transformed Pytorch tensor
                   and target is a dictionary containing 'boxes' and 'labels'.
        """
        # img is a PIL Image, target_ann is the annotation dictionary
        img, target_ann = self.voc_data[idx]
        original_w, original_h = img.size

        # Parse annotations
        # The target_ann['annotation']['object'] can be a list of objects or a single dict
        objects = target_ann['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects] # Make it a list if it's a single object

        boxes = []
        labels = []

        for obj in objects:
            class_name = obj['name']
            # Skip difficult or truncated objects if needed, or handle them
            # For now, we include all objects
            # if obj['difficult'] == '1' or obj['truncated'] == '1':
            #     continue
            
            # VOC bndbox is [xmin, ymin, xmax, ymax]
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[class_name])

        if not boxes: # Handle cases with no objects after filtering (if any)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes_tensor, 'labels': labels_tensor}

        # Apply image transform
        if self.transform:
            # Store original image size for bounding box scaling
            # The transform pipeline usually includes Resize. We need to know the new size.
            # A simple way is to assume the resize transform is T.Resize and get its size.
            # However, transforms can be more complex.
            # A robust way: transform the image, then get its new size.
            
            transformed_img = self.transform(img) # This applies all transforms including Resize
            
            # Get the size of the image *after* transformation
            # transformed_img is a tensor of shape [C, H', W']
            new_h, new_w = transformed_img.shape[1], transformed_img.shape[2]

            # Scale bounding boxes
            if boxes_tensor.shape[0] > 0:
                # Calculate scaling factors
                scale_x = new_w / original_w
                scale_y = new_h / original_h

                # Apply scaling to bounding box coordinates
                scaled_boxes = boxes_tensor.clone()
                scaled_boxes[:, 0] = boxes_tensor[:, 0] * scale_x # xmin
                scaled_boxes[:, 1] = boxes_tensor[:, 1] * scale_y # ymin
                scaled_boxes[:, 2] = boxes_tensor[:, 2] * scale_x # xmax
                scaled_boxes[:, 3] = boxes_tensor[:, 3] * scale_y # ymax
                
                # Ensure boxes are within image bounds after scaling (clamping)
                scaled_boxes[:, 0::2] = torch.clamp(scaled_boxes[:, 0::2], 0, new_w -1) # x coords
                scaled_boxes[:, 1::2] = torch.clamp(scaled_boxes[:, 1::2], 0, new_h -1) # y coords

                target['boxes'] = scaled_boxes
            
            img = transformed_img
        else: # If no transform, ensure image is a tensor for collate_fn
            # This case might be less common if get_transform always provides ToTensor
            img = torchvision.transforms.functional.to_tensor(img)


        return img, target


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.

    Args:
        batch (list): A list of (image, target) tuples.

    Returns:
        tuple: (images, targets) where images is a stacked tensor and
               targets is a list of target dictionaries.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


if __name__ == '__main__':
    print("--- PascalVOCDataset Demo ---")
    
    # Define root directory for dataset
    # The script is in object_detector/src, data is in object_detector/data
    # So, ../data relative to this script's location.
    # VOCDetection will create VOCdevkit inside this root_dir.
    dataset_root_dir = os.path.join(os.path.dirname(__file__), '..', 'data') 
    os.makedirs(dataset_root_dir, exist_ok=True)
    print(f"Dataset root directory: {dataset_root_dir}")

    # 1. Instantiate training dataset
    print("\n--- Loading Training Data (will download if not present) ---")
    # For the demo, use a fixed resize_size that get_transform will apply
    resize_demo_size = (300, 300)
    train_transform = get_transform(train=True, resize_size=resize_demo_size)
    
    # It's good practice to use a small subset for initial testing if download is slow
    # or if the dataset is large. VOCDetection doesn't have a direct way to subset
    # before loading all annotations, but we can wrap it or slice it later.
    # For now, we'll use the 'val' set as it's smaller than 'train' for a quicker demo.
    # Using `image_set='train'` can be slow for the first download and processing.
    # Let's use 'val' for a quicker demo, or a small custom set if available.
    # For submission, the prompt asked for "train" and "val" examples.
    # We will use 'val' for the __getitem__ demo for speed, and 'train' for DataLoader if practical.
    
    try:
        train_dataset = PascalVOCDataset(
            root_dir=dataset_root_dir,
            year="2012",
            image_set="train", # Using 'train' as requested for one example
            transform=train_transform,
            download=True # Set to False if already downloaded and to avoid re-downloads
        )
        print(f"Training dataset loaded. Number of samples: {len(train_dataset)}")

        # 2. Get a sample from the training dataset
        if len(train_dataset) > 0:
            print("\n--- Sample from Training Dataset ---")
            # Fetch a sample that is likely to have objects
            # This index might need adjustment if the first few images are empty or problematic
            sample_idx = 0 
            try:
                # Try to find an image with annotations for a better demo
                for i in range(min(100, len(train_dataset))): # Check first 100 images
                    _, temp_target = train_dataset.voc_data[i] # Get raw annotation
                    if temp_target['annotation']['object']:
                        sample_idx = i
                        break
                
                img, target = train_dataset[sample_idx]
                print(f"Sample index: {sample_idx}")
                print(f"Image shape: {img.shape}") # Expected: [3, resize_demo_size[0], resize_demo_size[1]]
                print(f"Target: {target}")
                print(f"Number of objects: {target['boxes'].shape[0]}")
                assert img.shape == (3, resize_demo_size[0], resize_demo_size[1]), f"Image shape mismatch: {img.shape}"
                if target['boxes'].shape[0] > 0:
                    assert target['boxes'].shape[1] == 4, "Boxes shape mismatch"
                    assert target['labels'].shape[0] == target['boxes'].shape[0], "Labels mismatch with boxes"
                    # Check if box coordinates are within the new image dimensions
                    max_x = target['boxes'][:, 2].max().item() if target['boxes'].numel() > 0 else 0
                    max_y = target['boxes'][:, 3].max().item() if target['boxes'].numel() > 0 else 0
                    assert max_x < resize_demo_size[1], f"Max x coordinate {max_x} out of bounds {resize_demo_size[1]}"
                    assert max_y < resize_demo_size[0], f"Max y coordinate {max_y} out of bounds {resize_demo_size[0]}"

            except Exception as e:
                print(f"Error getting sample from training set: {e}")
                print("This might happen if the dataset is not yet fully downloaded/extracted, or sample_idx is out of bounds.")
        else:
            print("Training dataset is empty or failed to load.")

    except Exception as e:
        print(f"Could not load 'train' dataset: {e}. This demo might take a while if downloading for the first time.")


    # 3. Instantiate validation dataset
    print("\n--- Loading Validation Data (will download if not present) ---")
    val_transform = get_transform(train=False, resize_size=resize_demo_size)
    try:
        val_dataset = PascalVOCDataset(
            root_dir=dataset_root_dir,
            year="2012",
            image_set="val",
            transform=val_transform,
            download=True
        )
        print(f"Validation dataset loaded. Number of samples: {len(val_dataset)}")

        # 4. Use with DataLoader
        if len(val_dataset) > 0:
            print("\n--- DataLoader Demo (with validation set) ---")
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=4, # Small batch for demo
                shuffle=False, # No need to shuffle for val demo
                collate_fn=collate_fn,
                num_workers=0 # For simplicity in demo, use 0. For training, >0 is better.
            )

            # Iterate a few samples
            num_batches_to_show = 2
            for i, (images, targets) in enumerate(val_dataloader):
                if i >= num_batches_to_show:
                    break
                print(f"\nBatch {i+1}:")
                print(f"Images batch shape: {images.shape}") # Expected: [batch_size, 3, resize_h, resize_w]
                print(f"Targets batch length: {len(targets)}") # Expected: batch_size
                for j, target_item in enumerate(targets):
                    print(f"  Target {j}: boxes shape {target_item['boxes'].shape}, labels shape {target_item['labels'].shape}")
                    if target_item['boxes'].shape[0] > 0:
                         assert target_item['boxes'].shape[1] == 4
                         # Check scaled box coordinates again
                         max_x_batch = target_item['boxes'][:, 2].max().item() if target_item['boxes'].numel() > 0 else 0
                         max_y_batch = target_item['boxes'][:, 3].max().item() if target_item['boxes'].numel() > 0 else 0
                         assert max_x_batch < resize_demo_size[1], f"Batch target {j} Max x coord {max_x_batch} out of bounds {resize_demo_size[1]}"
                         assert max_y_batch < resize_demo_size[0], f"Batch target {j} Max y coord {max_y_batch} out of bounds {resize_demo_size[0]}"


            assert images.shape[0] <= 4 # Batch size
            assert images.shape[1] == 3 # Channels
            assert images.shape[2] == resize_demo_size[0] # Height
            assert images.shape[3] == resize_demo_size[1] # Width
            print("\nDataLoader demo successful.")
        else:
            print("Validation dataset is empty or failed to load. Skipping DataLoader demo.")

    except Exception as e:
        print(f"Could not load 'val' dataset: {e}. This demo might take a while if downloading for the first time.")
        print("If you are running this for the first time, dataset download and extraction can take several minutes.")
        print("Please ensure you have internet connectivity and sufficient disk space.")

    print("\n--- PASCAL_VOC_CLASSES ---")
    print(PASCAL_VOC_CLASSES)
    print(f"Number of classes: {len(PASCAL_VOC_CLASSES)}")

    print("\nScript execution finished.")
