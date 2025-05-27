import torch
import torchvision
import torchvision.transforms as T

def get_transform(train):
    """
    Returns a list of transforms to be applied to the images.
    Args:
        train (bool): If True, include random horizontal flip.
    Returns:
        transforms.Compose: A composed list of transforms.
    """
    transforms = []
    # Common transforms
    transforms.append(T.Resize((300, 300)))  # Resize images to 300x300 for SSD
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    transforms.append(T.ToTensor())
    # Normalization for pre-trained backbones (e.g., ResNet)
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)

def get_voc_datasets(root_dir='./data', year='2012', download=True):
    """
    Loads the PASCAL VOC detection datasets.
    Args:
        root_dir (str): Root directory for the dataset.
        year (str): Year of the VOC dataset (e.g., '2007', '2012').
        download (bool): If True, download the dataset if not found locally.
    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    # Training dataset
    dataset_train = torchvision.datasets.VOCDetection(
        root=root_dir,
        year=year,
        image_set='train',
        transform=get_transform(train=True),
        target_transform=None,  # Targets are returned as dicts
        download=download
    )
    
    # Validation dataset
    dataset_val = torchvision.datasets.VOCDetection(
        root=root_dir,
        year=year,
        image_set='val',
        transform=get_transform(train=False),
        target_transform=None, # Targets are returned as dicts
        download=download
    )
    
    return dataset_train, dataset_val

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    Stacks images and gathers targets into a list.
    Args:
        batch (list): A list of (image, target) tuples.
    Returns:
        tuple: A tuple containing a batch of images and a list of targets.
    """
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

def get_dataloaders(batch_size=4, num_workers=2, root_dir='./data', year='2012', download=True):
    """
    Creates DataLoaders for the training and validation datasets.
    Args:
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of worker processes for data loading.
        root_dir (str): Root directory for the dataset.
        year (str): Year of the VOC dataset.
        download (bool): If True, download the dataset if not found locally.
    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """
    dataset_train, dataset_val = get_voc_datasets(root_dir=root_dir, year=year, download=download)
    
    # Training DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True # Recommended for faster data transfer to GPU
    )
    
    # Validation DataLoader
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True # Recommended for faster data transfer to GPU
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    print("Attempting to create DataLoaders...")
    # Parameters for DataLoaders
    batch_size = 2 # Smaller batch size for testing
    num_workers = 1 # Fewer workers for testing to reduce resource usage

    # Get DataLoaders, set download to True for the first run if dataset is not present
    # Set to False if dataset is already downloaded to save time.
    # Note: Downloading and extracting VOC2012 can take a significant amount of time and disk space.
    try:
        train_dataloader, val_dataloader = get_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers,
            download=True # Set to False if dataset already downloaded
        )
        print("DataLoaders created successfully.")

        # Fetch one batch from the training loader
        print("\nFetching one batch from the training loader...")
        images, targets = next(iter(train_dataloader))
        
        # Print shape of images and structure of targets
        print(f"Images batch shape: {images.shape}")
        print(f"Targets batch length: {len(targets)}")
        if len(targets) > 0:
            print("Structure of the first target in the batch:")
            # VOCDetection returns target as a dictionary.
            # The 'annotation' key contains details about objects in the image.
            # Example: {'annotation': {'folder': 'VOC2012', 'filename': '...', 'size': {'width': ..., 'height': ..., 'depth': ...}, 
            # 'segmented': '0', 'object': [{'name': '...', 'pose': '...', ...}, ...]}}
            print(targets[0]) 
        
        print("\nSuccessfully loaded and tested one batch from the training dataset.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the dataset is correctly downloaded and accessible in the './data' directory.")
        print("If you haven't downloaded it, ensure `download=True` in `get_dataloaders` or `get_voc_datasets`.")

    print("\nDataset script execution finished.")
