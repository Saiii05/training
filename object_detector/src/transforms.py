import torchvision.transforms as T

# Define typical ImageNet means and stds for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform(train, resize_size=(300, 300)):
    """
    Returns a composition of transforms for image preprocessing.

    Args:
        train (bool): If True, returns training transforms (with augmentation).
                      If False, returns validation/testing transforms.
        resize_size (tuple): The target size (height, width) for resizing images.

    Returns:
        torchvision.transforms.Compose: A composition of transforms.
    """
    transforms = []

    # Common transforms
    transforms.append(T.Resize(resize_size))

    if train:
        # Training specific transforms
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        # Optional: ColorJitter for more augmentation
        transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        # Add more augmentations here if needed, e.g., RandomRotation, RandomResizedCrop (careful with bboxes)
    
    # Convert PIL image to PyTorch tensor
    transforms.append(T.ToTensor())
    
    # Normalize the tensor
    transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    
    return T.Compose(transforms)

if __name__ == '__main__':
    # Example usage:
    from PIL import Image
    import torch

    # Create a dummy PIL Image (e.g., 3 channels, 600x800)
    dummy_pil_image = Image.new('RGB', (800, 600), color = 'red')

    # Get training transforms
    train_transforms = get_transform(train=True, resize_size=(300,300))
    transformed_train_image = train_transforms(dummy_pil_image)
    print("--- Training Transforms ---")
    print(f"Input PIL image size: {dummy_pil_image.size}")
    print(f"Transformed image type: {type(transformed_train_image)}")
    print(f"Transformed image shape: {transformed_train_image.shape}") # Expected: torch.Size([3, 300, 300])
    assert transformed_train_image.shape == (3, 300, 300)

    # Get validation/testing transforms
    val_transforms = get_transform(train=False, resize_size=(256,256))
    transformed_val_image = val_transforms(dummy_pil_image)
    print("\n--- Validation Transforms ---")
    print(f"Input PIL image size: {dummy_pil_image.size}")
    print(f"Transformed image type: {type(transformed_val_image)}")
    print(f"Transformed image shape: {transformed_val_image.shape}") # Expected: torch.Size([3, 256, 256])
    assert transformed_val_image.shape == (3, 256, 256)
    
    print("\nTransforms defined successfully and example usage verified.")
