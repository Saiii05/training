import torch
import torch.utils.data
import argparse
import os
import numpy as np # For mAP calculation, can be replaced with torch ops mostly

# Assuming src.dataset, src.transforms, src.model, src.utils are in PYTHONPATH
# or the script is run from a location where they can be imported.
from dataset import PascalVOCDataset, PASCAL_VOC_CLASSES, collate_fn
from transforms import get_transform
from model import ObjectDetectionModel
import utils # For box_cxcywh_to_xyxy, non_max_suppression, matrix_iou


def setup_args():
    parser = argparse.ArgumentParser(description='Object Detection Evaluation Script')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to the model checkpoint (.pth) file.')
    parser.add_argument('--data-path', type=str, default='./data/VOCdevkit',
                        help='Path to VOCdevkit directory.')
    parser.add_argument('--num-classes-fg', type=int, default=20,
                        help='Number of foreground classes (e.g., 20 for Pascal VOC).')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--image-size', type=int, default=300, help='Size to resize images to (image_size x image_size).')
    parser.add_argument('--iou-threshold-map', type=float, default=0.5,
                        help='IoU threshold for considering a detection as a true positive in mAP calculation.')
    parser.add_argument('--score-threshold-nms', type=float, default=0.01,
                        help='Score threshold for filtering detections before Non-Maximum Suppression.')
    parser.add_argument('--iou-threshold-nms', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader.')
    return parser

def calculate_average_precision(precisions, recalls):
    """
    Calculates 11-point interpolated average precision.

    Args:
        precisions (torch.Tensor): Precision values for different score thresholds.
        recalls (torch.Tensor): Recall values for different score thresholds.

    Returns:
        torch.Tensor: Average precision.
    """
    if not isinstance(precisions, torch.Tensor):
        precisions = torch.tensor(precisions)
    if not isinstance(recalls, torch.Tensor):
        recalls = torch.tensor(recalls)
        
    device = precisions.device
    recall_levels = torch.linspace(0, 1, 11, device=device)
    interpolated_precisions = []

    for r_level in recall_levels:
        # Find precision values where recalls >= r_level
        relevant_precisions = precisions[recalls >= r_level]
        if relevant_precisions.numel() == 0:
            max_prec = torch.tensor(0.0, device=device)
        else:
            max_prec = relevant_precisions.max()
        interpolated_precisions.append(max_prec)
    
    # Ensure interpolated_precisions is a tensor for .mean()
    ap = torch.stack(interpolated_precisions).mean()
    return ap

def evaluate(args):
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading dataset for evaluation...")
    resize_shape = (args.image_size, args.image_size)
    transform_eval = get_transform(train=False, resize_size=resize_shape)

    voc_root_dir = os.path.abspath(os.path.join(args.data_path, "..")) if "VOCdevkit" in args.data_path else args.data_path
    print(f"PascalVOCDataset root directory for evaluation: {voc_root_dir}")
    
    try:
        # Using 'val' set for evaluation is typical. 'test' set might not have public GT.
        eval_dataset = PascalVOCDataset(
            root_dir=voc_root_dir,
            year="2012", # Or a specific year like "2007" for VOC07 test
            image_set='val', # Common choice for evaluation
            transform=transform_eval,
            download=True # Download if not present
        )
    except Exception as e:
        print(f"Error loading PascalVOCDataset for evaluation: {e}")
        return

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Evaluation data: {len(eval_dataset)} samples, {len(eval_loader)} batches.")

    # --- Model ---
    print("Initializing model...")
    model = ObjectDetectionModel(
        num_classes_fg=args.num_classes_fg,
        image_size_for_default_boxes=(args.image_size, args.image_size)
    )
    
    if not os.path.isfile(args.checkpoint_path):
        print(f"Checkpoint file not found at '{args.checkpoint_path}'. Exiting.")
        return
        
    print(f"Loading checkpoint from '{args.checkpoint_path}'")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Handle potential DataParallel prefix 'module.'
    model_state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in model_state_dict.keys()):
        print("Removing 'module.' prefix from checkpoint keys.")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(model_state_dict)
        
    model.to(device)
    model.eval()

    # --- Collect Predictions and Ground Truths ---
    print("Collecting predictions and ground truths...")
    all_pred_boxes_list = []    # List per image of [N_preds, 4]
    all_pred_labels_list = []   # List per image of [N_preds]
    all_pred_scores_list = []   # List per image of [N_preds]
    all_gt_boxes_list = []      # List per image of [N_gt, 4]
    all_gt_labels_list = []     # List per image of [N_gt]
    # To map predictions back to their original image for mAP calculation per class
    all_pred_img_indices_list = [] # List per image, contains original image index for each pred box

    with torch.no_grad():
        for batch_idx, (images, targets_batch) in enumerate(eval_loader):
            images = images.to(device)
            # No need to move targets to device if only accessing them on CPU later for GT storage
            # However, if any part of target processing before storage needs CUDA, then move.

            cls_logits, bbox_pred_cxcywh, default_boxes_xyxy = model(images)
            default_boxes_xyxy = default_boxes_xyxy.to(device) # Ensure on device

            for i in range(images.size(0)): # Iterate through images in the batch
                original_image_idx = batch_idx * args.batch_size + i # Global image index

                # Store Ground Truth
                gt_boxes_img = targets_batch[i]['boxes'] # Assuming these are on CPU from DataLoader
                gt_labels_img = targets_batch[i]['labels']
                all_gt_boxes_list.append(gt_boxes_img.clone())
                all_gt_labels_list.append(gt_labels_img.clone())

                # Process Predictions for this image
                scores_img = torch.softmax(cls_logits[i], dim=-1) # (num_default_boxes, num_classes_loss)
                boxes_img_cxcywh = bbox_pred_cxcywh[i]            # (num_default_boxes, 4)
                
                # Convert predicted boxes to xyxy (still normalized by image size at this stage)
                # The model's default boxes are normalized, and predictions are also normalized.
                # For NMS and IoU, absolute coordinates are usually better if images vary in size,
                # but here images are resized to fixed size, so normalized is fine.
                # However, utils.matrix_iou and utils.non_max_suppression expect absolute coords.
                # Let's scale them by image_size for now.
                # This assumes default_boxes and predictions are normalized [0,1].
                # The `bbox_pred_cxcywh` from model is normalized.
                # `utils.box_cxcywh_to_xyxy` will keep them normalized.
                # For NMS/IoU, it's generally better to have unnormalized boxes if possible,
                # but if everything is consistently normalized to the *same* image dimensions, it's okay.
                # Since all input images are resized to args.image_size, normalized coords work like absolute.

                boxes_img_xyxy = utils.box_cxcywh_to_xyxy(boxes_img_cxcywh) # Still normalized [0,1]
                
                img_pred_boxes_collected = []
                img_pred_labels_collected = []
                img_pred_scores_collected = []

                for cls_idx in range(args.num_classes_fg): # Iterate PASCAL_VOC_CLASSES
                    class_scores = scores_img[:, cls_idx + 1] # Skip background (idx 0)
                    
                    score_mask = class_scores > args.score_threshold_nms
                    if not score_mask.any():
                        continue

                    selected_scores = class_scores[score_mask]
                    selected_boxes_normalized = boxes_img_xyxy[score_mask] # Normalized
                    
                    # NMS expects absolute coordinates. Let's scale by image_size.
                    # This is only for NMS. mAP will use these processed boxes.
                    selected_boxes_abs = selected_boxes_normalized * args.image_size

                    keep_indices = utils.non_max_suppression(
                        selected_boxes_abs, # Absolute for NMS
                        selected_scores, 
                        args.iou_threshold_nms
                    )
                    
                    # Store kept boxes (normalized), labels, scores
                    img_pred_boxes_collected.append(selected_boxes_normalized[keep_indices])
                    img_pred_labels_collected.append(torch.full_like(selected_scores[keep_indices], cls_idx, dtype=torch.long, device=device))
                    img_pred_scores_collected.append(selected_scores[keep_indices])

                if len(img_pred_boxes_collected) > 0:
                    all_pred_boxes_list.append(torch.cat(img_pred_boxes_collected).cpu())
                    all_pred_labels_list.append(torch.cat(img_pred_labels_collected).cpu())
                    all_pred_scores_list.append(torch.cat(img_pred_scores_collected).cpu())
                else: # No predictions for this image after filtering and NMS
                    all_pred_boxes_list.append(torch.empty(0, 4))
                    all_pred_labels_list.append(torch.empty(0, dtype=torch.long))
                    all_pred_scores_list.append(torch.empty(0))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(eval_loader)} batches for prediction collection.")

    print(f"Collected {len(all_pred_boxes_list)} predictions and {len(all_gt_boxes_list)} ground truths.")

    # --- Calculate mAP ---
    print("Calculating mAP...")
    average_precisions_per_class = torch.zeros(args.num_classes_fg, device='cpu') # mAP calculation on CPU

    for cls_idx in range(args.num_classes_fg):
        # Gather all predictions for this class
        class_all_pred_boxes = []
        class_all_pred_scores = []
        class_all_pred_img_indices = [] # To know which image each prediction belongs to

        for img_idx in range(len(all_pred_labels_list)):
            pred_labels_img = all_pred_labels_list[img_idx]
            mask_this_class = (pred_labels_img == cls_idx)
            if mask_this_class.any():
                class_all_pred_boxes.append(all_pred_boxes_list[img_idx][mask_this_class])
                class_all_pred_scores.append(all_pred_scores_list[img_idx][mask_this_class])
                # Store image index for each of these predictions
                class_all_pred_img_indices.extend([img_idx] * mask_this_class.sum().item())
        
        if not class_all_pred_scores: # No predictions for this class
            average_precisions_per_class[cls_idx] = 0.0
            print(f"Class '{PASCAL_VOC_CLASSES[cls_idx]}' (ID {cls_idx}): AP = 0.0 (No predictions)")
            continue
        
        class_all_pred_boxes = torch.cat(class_all_pred_boxes)
        class_all_pred_scores = torch.cat(class_all_pred_scores)
        class_all_pred_img_indices = torch.tensor(class_all_pred_img_indices, dtype=torch.long)

        # Sort predictions by score (descending)
        sort_indices = torch.argsort(class_all_pred_scores, descending=True)
        class_all_pred_boxes = class_all_pred_boxes[sort_indices]
        class_all_pred_scores = class_all_pred_scores[sort_indices] # Not strictly needed after sorting
        class_all_pred_img_indices = class_all_pred_img_indices[sort_indices]

        # Gather all GTs for this class and count total
        num_total_gt_for_class = 0
        # Keep track of matched GTs: list of lists of bool tensors (one per image, per GT box in that image for this class)
        gt_matched_map_for_class_per_image = [] 
        
        for img_idx in range(len(all_gt_labels_list)):
            gt_labels_img = all_gt_labels_list[img_idx]
            gt_boxes_img = all_gt_boxes_list[img_idx]
            mask_gt_this_class = (gt_labels_img == cls_idx)
            num_total_gt_for_class += mask_gt_this_class.sum().item()
            gt_matched_map_for_class_per_image.append(torch.zeros(mask_gt_this_class.sum().item(), dtype=torch.bool))
            
        if num_total_gt_for_class == 0: # No GT for this class
            average_precisions_per_class[cls_idx] = 0.0 # Or handle as NaN or skip if class not in dataset
            print(f"Class '{PASCAL_VOC_CLASSES[cls_idx]}' (ID {cls_idx}): AP = 0.0 (No ground truth objects)")
            continue

        tp = torch.zeros(len(class_all_pred_scores), device='cpu')
        fp = torch.zeros(len(class_all_pred_scores), device='cpu')

        for pred_k in range(len(class_all_pred_scores)):
            current_pred_box = class_all_pred_boxes[pred_k] # Normalized
            original_image_idx = class_all_pred_img_indices[pred_k].item()

            # Get GT boxes for this class in the specific image
            gt_labels_in_img = all_gt_labels_list[original_image_idx]
            gt_boxes_in_img = all_gt_boxes_list[original_image_idx] # Normalized
            
            mask_gt_this_class_in_img = (gt_labels_in_img == cls_idx)
            gt_boxes_for_class_in_image = gt_boxes_in_img[mask_gt_this_class_in_img]

            if gt_boxes_for_class_in_image.numel() == 0:
                fp[pred_k] = 1 # No GT for this class in this image, so it's a false positive
                continue

            # Calculate IoU: matrix_iou expects (M,4) and (N,4)
            # Here, current_pred_box is (4), gt_boxes_for_class_in_image is (NumGT_in_img_for_cls, 4)
            # utils.matrix_iou needs both to be 2D.
            iou_with_gts = utils.matrix_iou(current_pred_box.unsqueeze(0), gt_boxes_for_class_in_image)[0]
            
            max_iou, best_gt_match_idx_in_class_gts = iou_with_gts.max(dim=0)

            if max_iou >= args.iou_threshold_map:
                if not gt_matched_map_for_class_per_image[original_image_idx][best_gt_match_idx_in_class_gts]:
                    tp[pred_k] = 1
                    gt_matched_map_for_class_per_image[original_image_idx][best_gt_match_idx_in_class_gts] = True
                else: # GT already matched by a higher scoring prediction
                    fp[pred_k] = 1
            else: # Max IoU below threshold
                fp[pred_k] = 1
        
        tp_cumulative = torch.cumsum(tp, dim=0)
        fp_cumulative = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumulative / max(1.0, float(num_total_gt_for_class)) # Avoid div by zero if no GT
        precisions = tp_cumulative / (tp_cumulative + fp_cumulative + 1e-10) # Avoid div by zero

        ap = calculate_average_precision(precisions, recalls)
        average_precisions_per_class[cls_idx] = ap
        print(f"Class '{PASCAL_VOC_CLASSES[cls_idx]}' (ID {cls_idx}): AP = {ap.item():.4f}")

    mean_ap = average_precisions_per_class.mean()
    print(f"\n--- Evaluation Results ---")
    for cls_idx, ap_val in enumerate(average_precisions_per_class):
        print(f"AP for {PASCAL_VOC_CLASSES[cls_idx]:<15}: {ap_val.item():.4f}")
    print(f"\nMean Average Precision (mAP) @ IoU={args.iou_threshold_map}: {mean_ap.item():.4f}")


if __name__ == '__main__':
    arg_parser = setup_args()
    args = arg_parser.parse_args()
    
    # Adjust data_path for common structures if needed by user
    if os.path.basename(args.data_path) == "VOCdevkit":
        print(f"Adjusting data_path for PascalVOCDataset: '{args.data_path}' -> '{os.path.dirname(args.data_path)}'")
        # This adjustment is now handled by voc_root_dir inside evaluate()

    evaluate(args)
