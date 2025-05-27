import torch
import torch.optim as optim
import torch.utils.data
import argparse
import os
import time

# Assuming src.dataset, src.transforms, src.model, src.loss are in PYTHONPATH
# or the script is run from a location where they can be imported.
# If running from object_detector/ directory, then:
# from src.dataset import PascalVOCDataset, collate_fn
# from src.transforms import get_transform
# from src.model import ObjectDetectionModel
# from src.loss import DetectionLoss
# For direct execution from object_detector/src or if src is added to path:
from dataset import PascalVOCDataset, collate_fn
from transforms import get_transform
from model import ObjectDetectionModel
from loss import DetectionLoss

def setup_args():
    parser = argparse.ArgumentParser(description='Object Detection Training Script')
    parser.add_argument('--data-path', type=str, default='./data/VOCdevkit', # Adjusted default for typical layout
                        help='Path to VOCdevkit directory (e.g., where PASCAL VOC data is).')
    parser.add_argument('--num-classes-fg', type=int, default=20,
                        help='Number of foreground classes (e.g., 20 for Pascal VOC).')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--output-dir', type=str, default='./models_checkpoints/', # Adjusted default
                        help='Directory to save model checkpoints.')
    parser.add_argument('--image-size', type=int, default=300, help='Size to resize images to (image_size x image_size).')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from.')
    parser.add_argument('--print-freq', type=int, default=10, help='Frequency of printing training statistics (batches).')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader.')
    return parser

def train(args):
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Directory Setup ---
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # --- Data Loading ---
    print("Loading datasets...")
    resize_shape = (args.image_size, args.image_size)
    transform_train = get_transform(train=True, resize_size=resize_shape)
    transform_val = get_transform(train=False, resize_size=resize_shape)

    # Note: PascalVOCDataset expects root_dir to be the parent of VOCdevkit
    # If args.data_path is './data/VOCdevkit', then root_dir for PascalVOCDataset should be './data'
    # Let's adjust data_path usage for PascalVOCDataset
    voc_root_dir = os.path.abspath(os.path.join(args.data_path, "..")) if "VOCdevkit" in args.data_path else args.data_path
    print(f"PascalVOCDataset root directory: {voc_root_dir}")


    try:
        train_dataset = PascalVOCDataset(
            root_dir=voc_root_dir, 
            year="2012", # Or use a different year/split as needed
            image_set='train', 
            transform=transform_train, 
            download=True
        )
        val_dataset = PascalVOCDataset(
            root_dir=voc_root_dir, 
            year="2012", 
            image_set='val', 
            transform=transform_val, 
            download=True
        )
    except Exception as e:
        print(f"Error loading PascalVOCDataset: {e}")
        print("Please ensure your --data-path is correctly set to the parent of VOCdevkit, or to a directory where VOCdevkit can be downloaded.")
        print("Example: if VOCdevkit is at './data/VOCdevkit', --data-path should be './data/VOCdevkit' or './data'.")
        return


    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Training data: {len(train_dataset)} samples, {len(train_loader)} batches.")
    print(f"Validation data: {len(val_dataset)} samples, {len(val_loader)} batches.")

    # --- Model ---
    print("Initializing model...")
    model = ObjectDetectionModel(
        num_classes_fg=args.num_classes_fg,
        image_size_for_default_boxes=(args.image_size, args.image_size)
    )
    model.to(device)

    # --- Loss & Optimizer ---
    criterion = DetectionLoss(
        num_classes=args.num_classes_fg + 1, # +1 for background
        iou_threshold_positive=0.5,
        iou_threshold_negative_upper=0.4,
        neg_pos_ratio=3
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- Resuming Training ---
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'lr_scheduler_state_dict' in checkpoint and lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_total_loss_sum = 0
        train_loc_loss_sum = 0
        train_conf_loss_sum = 0

        for i, (images, targets_batch) in enumerate(train_loader):
            images = images.to(device)
            # Move targets to device
            targets_on_device = []
            for t in targets_batch:
                targets_on_device.append({k: v.to(device) for k, v in t.items()})
            targets_batch = targets_on_device

            # Extract GT boxes and labels for the loss function
            gt_boxes_batch = [t['boxes'] for t in targets_batch]
            gt_labels_batch = [t['labels'] for t in targets_batch]

            optimizer.zero_grad()

            # Forward pass
            cls_logits, bbox_pred_cxcywh, default_boxes_xyxy = model(images)

            # Calculate loss
            # Ensure default_boxes_xyxy is on the correct device
            total_loss, loc_loss, conf_loss = criterion(
                cls_logits, 
                bbox_pred_cxcywh, 
                gt_boxes_batch, 
                gt_labels_batch, 
                default_boxes_xyxy.to(device) # Model buffer might not auto-move with model.to(device)
            )

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            train_total_loss_sum += total_loss.item()
            train_loc_loss_sum += loc_loss.item()
            train_conf_loss_sum += conf_loss.item()

            if (i + 1) % args.print_freq == 0:
                avg_total_loss = train_total_loss_sum / (i + 1)
                avg_loc_loss = train_loc_loss_sum / (i + 1)
                avg_conf_loss = train_conf_loss_sum / (i + 1)
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Total Loss: {avg_total_loss:.4f}, Loc Loss: {avg_loc_loss:.4f}, Conf Loss: {avg_conf_loss:.4f}")

        avg_train_epoch_loss = train_total_loss_sum / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] Training Average Loss: {avg_train_epoch_loss:.4f}")
        
        if lr_scheduler:
            lr_scheduler.step()

        # --- Validation Loop ---
        model.eval()
        val_total_loss_sum = 0
        val_loc_loss_sum = 0
        val_conf_loss_sum = 0
        num_val_batches = 0

        with torch.no_grad():
            for images_val, targets_batch_val in val_loader:
                images_val = images_val.to(device)
                targets_on_device_val = []
                for t in targets_batch_val:
                    targets_on_device_val.append({k: v.to(device) for k, v in t.items()})
                targets_batch_val = targets_on_device_val

                gt_boxes_batch_val = [t['boxes'] for t in targets_batch_val]
                gt_labels_batch_val = [t['labels'] for t in targets_batch_val]
                
                cls_logits_val, bbox_pred_cxcywh_val, default_boxes_xyxy_val = model(images_val)
                
                total_loss_val, loc_loss_val, conf_loss_val = criterion(
                    cls_logits_val, 
                    bbox_pred_cxcywh_val, 
                    gt_boxes_batch_val, 
                    gt_labels_batch_val, 
                    default_boxes_xyxy_val.to(device)
                )
                
                val_total_loss_sum += total_loss_val.item()
                val_loc_loss_sum += loc_loss_val.item()
                val_conf_loss_sum += conf_loss_val.item()
                num_val_batches += 1
        
        if num_val_batches > 0:
            avg_val_total_loss = val_total_loss_sum / num_val_batches
            avg_val_loc_loss = val_loc_loss_sum / num_val_batches
            avg_val_conf_loss = val_conf_loss_sum / num_val_batches
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Validation: Total Loss: {avg_val_total_loss:.4f}, "
                  f"Loc Loss: {avg_val_loc_loss:.4f}, Conf Loss: {avg_val_conf_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{args.num_epochs}] Validation: No batches processed.")
            avg_val_total_loss = float('inf') # Set to infinity if no val batches

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{args.num_epochs}] completed in {epoch_duration:.2f}s.")

        # --- Checkpointing ---
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
            'best_val_loss': best_val_loss
        }
        if lr_scheduler:
            checkpoint_data['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

        # Save regular checkpoint
        save_path_regular = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint_data, save_path_regular)
        print(f"Saved checkpoint to {save_path_regular}")

        # Save best model based on validation loss
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            checkpoint_data['best_val_loss'] = best_val_loss # Update best_val_loss in checkpoint
            save_path_best = os.path.join(args.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint_data, save_path_best)
            print(f"Saved new best model checkpoint to {save_path_best} (Val Loss: {best_val_loss:.4f})")
            
    print("Training finished.")


if __name__ == '__main__':
    # Setup command line arguments
    arg_parser = setup_args()
    args = arg_parser.parse_args()
    
    # Adjust data_path for common structures if needed by user
    # Example: if --data-path is "VOCdevkit", change to "." for PascalVOCDataset's root
    if os.path.basename(args.data_path) == "VOCdevkit":
        print(f"Adjusting data_path for PascalVOCDataset: '{args.data_path}' -> '{os.path.dirname(args.data_path)}'")
        # args.data_path = os.path.dirname(args.data_path) # This is handled by voc_root_dir logic now

    # Call the main training function
    train(args)
