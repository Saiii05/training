import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming utils.py is in the same directory or PYTHONPATH is configured
from utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, matrix_iou, calculate_iou

class DetectionLoss(nn.Module):
    """
    Calculates the detection loss, which includes localization loss (Smooth L1)
    and classification loss (Cross-Entropy) for object detection.
    """
    def __init__(self, num_classes, iou_threshold_positive=0.5, 
                 iou_threshold_negative_upper=0.4, neg_pos_ratio=3):
        """
        Args:
            num_classes (int): Total number of classes (e.g., 20 foreground + 1 background = 21).
            iou_threshold_positive (float): IoU threshold to consider a default box as positive.
            iou_threshold_negative_upper (float): IoU threshold below which a default box is negative.
                                                 Boxes with IoU between this and positive_threshold are ignored.
            neg_pos_ratio (int): Ratio of negatives to positives for hard negative mining.
        """
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_threshold_positive = iou_threshold_positive
        self.iou_threshold_negative_upper = iou_threshold_negative_upper
        self.neg_pos_ratio = neg_pos_ratio

        # Reduction is 'none' to manually sum/average losses after hard negative mining
        self.classification_loss = nn.CrossEntropyLoss(reduction='none')
        self.localization_loss = nn.SmoothL1Loss(reduction='none') # Also known as Huber loss

    def forward(self, cls_logits, bbox_pred_cxcywh, gt_boxes_batch, gt_labels_batch, default_boxes_xyxy):
        """
        Calculates the total detection loss.

        Args:
            cls_logits (torch.Tensor): Predicted class logits.
                                       Shape: (batch_size, num_default_boxes, num_classes).
            bbox_pred_cxcywh (torch.Tensor): Predicted bounding box coordinates in
                                             [center_x, center_y, width, height] absolute format.
                                             Shape: (batch_size, num_default_boxes, 4).
            gt_boxes_batch (list[torch.Tensor]): List of ground truth bounding boxes for each image.
                                                 Each tensor shape: (num_gt_objects, 4) in [xmin, ymin, xmax, ymax].
            gt_labels_batch (list[torch.Tensor]): List of ground truth labels for each image.
                                                  Each tensor shape: (num_gt_objects,), labels 0 to num_fg_classes-1.
            default_boxes_xyxy (torch.Tensor): Default anchor boxes in [xmin, ymin, xmax, ymax] format.
                                               Shape: (num_default_boxes, 4).

        Returns:
            tuple: (total_loss, normalized_loc_loss, normalized_conf_loss)
        """
        batch_size = cls_logits.size(0)
        num_default_boxes = default_boxes_xyxy.size(0)

        # 1. Pre-processing: Convert predicted boxes to [xmin, ymin, xmax, ymax]
        # bbox_pred_xyxy will have shape (batch_size, num_default_boxes, 4)
        bbox_pred_xyxy = box_cxcywh_to_xyxy(bbox_pred_cxcywh.view(-1, 4)).view(batch_size, num_default_boxes, 4)

        batch_loc_loss_sum = 0.0
        batch_conf_loss_sum = 0.0
        total_num_positives_in_batch = 0

        for i in range(batch_size):
            gt_boxes_img = gt_boxes_batch[i]         # (num_gt_objects, 4)
            gt_labels_img = gt_labels_batch[i]       # (num_gt_objects,)
            current_bbox_pred_xyxy = bbox_pred_xyxy[i] # (num_default_boxes, 4)
            current_cls_logits = cls_logits[i]       # (num_default_boxes, num_classes)

            if gt_boxes_img.numel() == 0: # No ground truth objects in this image
                # Only negative classification loss, no localization loss
                conf_target_labels = torch.zeros(num_default_boxes, dtype=torch.long, device=cls_logits.device)
                conf_loss_all = self.classification_loss(current_cls_logits, conf_target_labels)
                # Consider all default boxes as negative if no GT
                # Simple strategy: take a fixed number of top negatives or all if neg_pos_ratio is 0
                if self.neg_pos_ratio > 0: # Avoid issues if neg_pos_ratio is 0
                    # num_neg_to_keep = min(num_default_boxes // 10, conf_loss_all.size(0)) # e.g. 10%
                    # A fixed number, e.g. 100, or related to neg_pos_ratio like self.neg_pos_ratio * 10 (heuristic)
                    num_neg_to_keep = min(int(self.neg_pos_ratio * 20), conf_loss_all.size(0)) # Heuristic: some negatives
                    if num_neg_to_keep > 0 and conf_loss_all.numel() > 0:
                        conf_loss_neg_hard, _ = torch.topk(conf_loss_all, num_neg_to_keep)
                        batch_conf_loss_sum += conf_loss_neg_hard.sum()
                # else: no positive, no explicit negative mining here, effectively zero conf loss for this image
                # or, if neg_pos_ratio is 0, it means we sum all negatives.
                # This case needs clarification for "no positives in image".
                # For now, we'll sum a few hard negatives.
                continue # Move to next image

            # 2. Matching default boxes to ground truth boxes
            # iou_matrix: (num_gt_objects, num_default_boxes)
            iou_matrix = matrix_iou(gt_boxes_img, default_boxes_xyxy)

            # max_iou_per_default: IoU of the best matching GT for each default box. Shape: (num_default_boxes,)
            # matched_gt_idx_per_default: Index of that best GT. Shape: (num_default_boxes,)
            max_iou_per_default, matched_gt_idx_per_default = torch.max(iou_matrix, dim=0)

            # Determine positive and negative default boxes
            # Positive: Default box has IoU with *any* GT >= iou_threshold_positive
            positive_mask = max_iou_per_default >= self.iou_threshold_positive
            
            # Negative: Default box has IoU with *all* GTs < iou_threshold_negative_upper
            # This is correctly captured by max_iou_per_default < negative_threshold
            # Those between positive and negative thresholds are ignored for classification loss
            # (unless also matched by a second rule, e.g. forcing a match for each GT box, not done here for simplicity)
            negative_mask = max_iou_per_default < self.iou_threshold_negative_upper
            # Ensure positives are not considered negative
            negative_mask[positive_mask] = False 

            num_img_positives = positive_mask.sum().item()
            total_num_positives_in_batch += num_img_positives

            # 3. Localization Loss (only for positive default boxes)
            if num_img_positives > 0:
                matched_pred_boxes_for_loc = current_bbox_pred_xyxy[positive_mask] # (num_img_positives, 4)
                # Get the GT boxes corresponding to these positive default boxes
                matched_gt_boxes_for_loc = gt_boxes_img[matched_gt_idx_per_default[positive_mask]] # (num_img_positives, 4)
                
                # Calculate localization loss (Smooth L1)
                loc_loss_per_img = self.localization_loss(matched_pred_boxes_for_loc, matched_gt_boxes_for_loc)
                batch_loc_loss_sum += loc_loss_per_img.sum() # Sum over coordinates and then over boxes

            # 4. Classification Loss (for positive and hard negative default boxes)
            # Target labels for classification:
            # - Background class (0) for negative default boxes
            # - Foreground class (1 to num_classes-1) for positive default boxes
            conf_target_labels = torch.zeros(num_default_boxes, dtype=torch.long, device=cls_logits.device)
            
            if num_img_positives > 0:
                # gt_labels_img are 0 to num_fg_classes-1. Add 1 for background class.
                conf_target_labels[positive_mask] = gt_labels_img[matched_gt_idx_per_default[positive_mask]] + 1
            
            # Calculate classification loss for all default boxes
            conf_loss_all = self.classification_loss(current_cls_logits, conf_target_labels)

            # Hard Negative Mining
            conf_loss_pos = conf_loss_all[positive_mask]
            
            # Consider only true negatives for hard negative mining
            conf_loss_neg_candidates = conf_loss_all[negative_mask]

            num_neg_to_keep = 0
            if num_img_positives > 0 : # Standard hard negative mining
                 num_neg_to_keep = min(int(num_img_positives * self.neg_pos_ratio), conf_loss_neg_candidates.size(0))
            elif conf_loss_neg_candidates.numel() > 0 and self.neg_pos_ratio > 0: 
                # No positives, but we have negatives and a ratio suggesting we should keep some.
                # Heuristic: keep a small fixed number or a small fraction of total defaults
                num_neg_to_keep = min(int(num_default_boxes * 0.05), conf_loss_neg_candidates.size(0)) # e.g. 5% of defaults up to available negatives
                # Alternative: self.neg_pos_ratio * some_base_number_of_positives_if_none_found (e.g. 10)
                # num_neg_to_keep = min(int(self.neg_pos_ratio * 10), conf_loss_neg_candidates.size(0))


            if num_neg_to_keep > 0 and conf_loss_neg_candidates.numel() > 0:
                conf_loss_neg_hard, _ = torch.topk(conf_loss_neg_candidates, num_neg_to_keep)
                batch_conf_loss_sum += conf_loss_pos.sum() + conf_loss_neg_hard.sum()
            elif num_img_positives > 0: # Has positives, but no negatives to keep (e.g. ratio is 0, or no negatives available)
                batch_conf_loss_sum += conf_loss_pos.sum()
            # If no positives and no negatives to keep, conf loss for this image is 0.

        # 5. Combine and Normalize losses
        if total_num_positives_in_batch == 0:
            # Avoid division by zero.
            # Localization loss is 0 if no positives.
            normalized_loc_loss = torch.tensor(0.0, device=cls_logits.device)
            # For classification loss, if no positives, normalize by total number of default boxes in batch
            # to provide some gradient signal from negatives if they were selected.
            # If batch_conf_loss_sum is also 0 (e.g. no GT, no negatives selected), then loss is 0.
            if batch_conf_loss_sum > 0 :
                 normalized_conf_loss = batch_conf_loss_sum / (batch_size * num_default_boxes) # Normalize by total predictions
            else:
                 normalized_conf_loss = torch.tensor(0.0, device=cls_logits.device)

        else:
            normalized_loc_loss = batch_loc_loss_sum / total_num_positives_in_batch
            normalized_conf_loss = batch_conf_loss_sum / total_num_positives_in_batch
            
        total_loss = normalized_loc_loss + normalized_conf_loss
        return total_loss, normalized_loc_loss, normalized_conf_loss


if __name__ == '__main__':
    print("--- DetectionLoss Demo ---")
    
    # Hyperparameters
    # num_fg_classes = 20
    # num_classes = num_fg_classes + 1 # Including background
    num_classes_for_loss = 21 # Pascal VOC: 20 fg classes + 1 bg class
    batch_size = 2
    num_default_boxes_val = 100 # Example number of default boxes
    
    # Instantiate loss function
    detection_loss_fn = DetectionLoss(
        num_classes=num_classes_for_loss,
        iou_threshold_positive=0.5,
        iou_threshold_negative_upper=0.4,
        neg_pos_ratio=3
    )

    # Dummy model outputs
    # cls_logits: (batch_size, num_default_boxes, num_classes)
    cls_logits_dummy = torch.randn(batch_size, num_default_boxes_val, num_classes_for_loss)
    # bbox_pred_cxcywh: (batch_size, num_default_boxes, 4) - absolute coords
    bbox_pred_cxcywh_dummy = torch.rand(batch_size, num_default_boxes_val, 4) * 200 # Assuming image size around 200x200

    # Dummy ground truth data (list of tensors)
    gt_boxes_batch_dummy = []
    gt_labels_batch_dummy = []

    # Image 1: Has objects
    num_gt_objects_img1 = 3
    gt_boxes_img1 = torch.rand(num_gt_objects_img1, 4) * 150 # xyxy, smaller than preds for potential overlap
    gt_boxes_img1[:, 2:] += gt_boxes_img1[:, :2] + 10 # Ensure xmax > xmin, ymax > ymin and some width/height
    gt_labels_img1 = torch.randint(0, num_classes_for_loss - 1, (num_gt_objects_img1,)) # Labels 0 to num_fg_classes-1
    gt_boxes_batch_dummy.append(gt_boxes_img1)
    gt_labels_batch_dummy.append(gt_labels_img1)

    # Image 2: No objects (to test that case)
    # gt_boxes_batch_dummy.append(torch.empty(0, 4))
    # gt_labels_batch_dummy.append(torch.empty(0, dtype=torch.long))
    # Image 2: Has one object
    num_gt_objects_img2 = 1
    gt_boxes_img2 = torch.rand(num_gt_objects_img2, 4) * 100 
    gt_boxes_img2[:, 2:] += gt_boxes_img2[:, :2] + 20 
    gt_labels_img2 = torch.randint(0, num_classes_for_loss - 1, (num_gt_objects_img2,))
    gt_boxes_batch_dummy.append(gt_boxes_img2)
    gt_labels_batch_dummy.append(gt_labels_img2)


    # Dummy default boxes (xyxy format)
    default_boxes_xyxy_dummy = torch.rand(num_default_boxes_val, 4) * 200
    default_boxes_xyxy_dummy[:, 2:] += default_boxes_xyxy_dummy[:, :2] + 5 # Ensure w,h > 0
    # Sort to ensure xmin < xmax, ymin < ymax if not already guaranteed by rand and add
    default_boxes_xyxy_dummy_min, _ = torch.min(default_boxes_xyxy_dummy[:, :2], default_boxes_xyxy_dummy[:, 2:], dim=1, keepdim=True)
    default_boxes_xyxy_dummy_max, _ = torch.max(default_boxes_xyxy_dummy[:, :2], default_boxes_xyxy_dummy[:, 2:], dim=1, keepdim=True)
    # default_boxes_xyxy_dummy = torch.cat([default_boxes_xyxy_dummy_min, default_boxes_xyxy_dummy_max], dim=1)
    # Simpler way: ensure x1 < x2, y1 < y2
    x1 = torch.rand(num_default_boxes_val, 1) * 190
    y1 = torch.rand(num_default_boxes_val, 1) * 190
    x2 = x1 + torch.rand(num_default_boxes_val, 1) * 10 + 5
    y2 = y1 + torch.rand(num_default_boxes_val, 1) * 10 + 5
    default_boxes_xyxy_dummy = torch.cat([x1,y1,x2,y2], dim=1)


    print(f"Cls Logits shape: {cls_logits_dummy.shape}")
    print(f"BBox Pred (cxcywh) shape: {bbox_pred_cxcywh_dummy.shape}")
    print(f"Default Boxes (xyxy) shape: {default_boxes_xyxy_dummy.shape}")
    for i in range(batch_size):
        print(f"GT Boxes Img {i} shape: {gt_boxes_batch_dummy[i].shape}")
        print(f"GT Labels Img {i} shape: {gt_labels_batch_dummy[i].shape}")


    # Calculate loss
    total_loss, loc_loss, conf_loss = detection_loss_fn(
        cls_logits_dummy,
        bbox_pred_cxcywh_dummy,
        gt_boxes_batch_dummy,
        gt_labels_batch_dummy,
        default_boxes_xyxy_dummy
    )

    print(f"\nTotal Loss: {total_loss.item()}")
    print(f"Localization Loss: {loc_loss.item()}")
    print(f"Classification Loss: {conf_loss.item()}")

    # Test with an image that has no GT objects
    print("\n--- Test with one image having no GT objects ---")
    gt_boxes_batch_no_obj = [gt_boxes_img1, torch.empty(0,4)]
    gt_labels_batch_no_obj = [gt_labels_img1, torch.empty(0, dtype=torch.long)]
    
    total_loss_no_obj, loc_loss_no_obj, conf_loss_no_obj = detection_loss_fn(
        cls_logits_dummy, # Using same logits and preds
        bbox_pred_cxcywh_dummy,
        gt_boxes_batch_no_obj,
        gt_labels_batch_no_obj,
        default_boxes_xyxy_dummy
    )
    print(f"Total Loss (one img no GT): {total_loss_no_obj.item()}")
    print(f"Localization Loss (one img no GT): {loc_loss_no_obj.item()}") # Should be based only on first image
    print(f"Classification Loss (one img no GT): {conf_loss_no_obj.item()}")
    
    # Test with a batch where NO images have GT objects
    print("\n--- Test with a batch where NO images have GT objects ---")
    gt_boxes_batch_all_no_obj = [torch.empty(0,4), torch.empty(0,4)]
    gt_labels_batch_all_no_obj = [torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)]

    total_loss_all_no, loc_loss_all_no, conf_loss_all_no = detection_loss_fn(
        cls_logits_dummy, 
        bbox_pred_cxcywh_dummy,
        gt_boxes_batch_all_no_obj,
        gt_labels_batch_all_no_obj,
        default_boxes_xyxy_dummy
    )
    print(f"Total Loss (all imgs no GT): {total_loss_all_no.item()}")
    print(f"Localization Loss (all imgs no GT): {loc_loss_all_no.item()}") # Should be 0
    print(f"Classification Loss (all imgs no GT): {conf_loss_all_no.item()}") # Should be non-zero if negatives are mined

    assert loc_loss_all_no.item() == 0.0, "Localization loss should be 0 if no GT objects in batch."
    # conf_loss_all_no can be > 0 if negatives are selected.

    print("\nDemo finished.")
