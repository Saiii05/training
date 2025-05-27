import torch
import torchvision

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between a single box (box1)
    and multiple boxes (box2).

    Args:
        box1 (torch.Tensor): A single bounding box, shape (4,), [xmin, ymin, xmax, ymax].
        box2 (torch.Tensor): Multiple bounding boxes, shape (N, 4), [xmin, ymin, xmax, ymax].

    Returns:
        torch.Tensor: IoU values, shape (N,).
    """
    # Ensure box1 is [1, 4] for broadcasting
    box1 = box1.unsqueeze(0) # Shape: [1, 4]

    # Calculate intersection coordinates
    # Top-left corner of intersection
    # max(box1_xmin, box2_xmin) and max(box1_ymin, box2_ymin)
    inter_xmin = torch.max(box1[:, 0], box2[:, 0])
    inter_ymin = torch.max(box1[:, 1], box2[:, 1])
    # Bottom-right corner of intersection
    # min(box1_xmax, box2_xmax) and min(box1_ymax, box2_ymax)
    inter_xmax = torch.min(box1[:, 2], box2[:, 2])
    inter_ymax = torch.min(box1[:, 3], box2[:, 3])

    # Calculate intersection area
    # Width of intersection: inter_xmax - inter_xmin
    # Height of intersection: inter_ymax - inter_ymin
    # If width or height is negative, intersection area is 0
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    intersection_area = inter_width * inter_height # Shape: (N,)

    # Calculate areas of individual boxes
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) # Shape: (1,)
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]) # Shape: (N,)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area # Shape: (N,)

    # Calculate IoU
    # Avoid division by zero if union_area is 0 (e.g., both boxes are empty or identical and empty)
    iou = intersection_area / (union_area + 1e-6) # Add epsilon for numerical stability

    return iou

def box_cxcywh_to_xyxy(boxes_cxcywh):
    """
    Converts boxes from [center_x, center_y, width, height] format
    to [xmin, ymin, xmax, ymax] format.

    Args:
        boxes_cxcywh (torch.Tensor): Boxes in [cx, cy, w, h] format, shape (N, 4).

    Returns:
        torch.Tensor: Boxes in [xmin, ymin, xmax, ymax] format, shape (N, 4).
    """
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def box_xyxy_to_cxcywh(boxes_xyxy):
    """
    Converts boxes from [xmin, ymin, xmax, ymax] format
    to [center_x, center_y, width, height] format.

    Args:
        boxes_xyxy (torch.Tensor): Boxes in [xmin, ymin, xmax, ymax] format, shape (N, 4).

    Returns:
        torch.Tensor: Boxes in [cx, cy, w, h] format, shape (N, 4).
    """
    xmin, ymin, xmax, ymax = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w / 2
    cy = ymin + h / 2
    return torch.stack((cx, cy, w, h), dim=1)

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (torch.Tensor): Bounding boxes, shape (N, 4), [xmin, ymin, xmax, ymax].
        scores (torch.Tensor): Confidence scores for each box, shape (N,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        torch.Tensor: Indices of the boxes to keep, sorted by score.
    """
    # torchvision.ops.nms expects boxes in [x1, y1, x2, y2] format which is what we have.
    # It returns the indices of the elements to keep.
    return torchvision.ops.nms(boxes, scores, iou_threshold)


if __name__ == '__main__':
    print("--- Testing Utility Functions ---")

    # 1. Test calculate_iou
    print("\n--- Testing calculate_iou ---")
    box1 = torch.tensor([10, 10, 50, 50], dtype=torch.float32) # xmin, ymin, xmax, ymax (area 40*40 = 1600)

    # Case 1: Perfect overlap
    box2_perfect = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    iou_perfect = calculate_iou(box1, box2_perfect)
    print(f"Box1: {box1}")
    print(f"Box2 (perfect overlap): {box2_perfect}")
    print(f"IoU (perfect overlap): {iou_perfect}") # Expected: [1.0]
    assert torch.isclose(iou_perfect, torch.tensor([1.0])).all()

    # Case 2: Partial overlap
    # Intersection: xmin=20, ymin=20, xmax=50, ymax=50 -> w=30, h=30 -> area=900
    # Union: area1 + area2 - intersection = 1600 (40*40) + 1600 (40*40) - 900 = 3200 - 900 = 2300
    # IoU = 900 / 2300 = 9/23 approx 0.3913
    box2_partial = torch.tensor([[20, 20, 60, 60]], dtype=torch.float32)
    iou_partial = calculate_iou(box1, box2_partial)
    print(f"Box2 (partial overlap): {box2_partial}")
    print(f"IoU (partial overlap): {iou_partial}")
    expected_iou_partial = 900.0 / (1600.0 + (40.0*40.0) - 900.0)
    assert torch.isclose(iou_partial, torch.tensor([expected_iou_partial])).all()


    # Case 3: No overlap
    box2_no_overlap = torch.tensor([[100, 100, 120, 120]], dtype=torch.float32)
    iou_no_overlap = calculate_iou(box1, box2_no_overlap)
    print(f"Box2 (no overlap): {box2_no_overlap}")
    print(f"IoU (no overlap): {iou_no_overlap}") # Expected: [0.0]
    assert torch.isclose(iou_no_overlap, torch.tensor([0.0])).all()

    # Case 4: Multiple boxes in box2
    box2_multiple = torch.tensor([
        [10, 10, 50, 50],    # Perfect overlap
        [20, 20, 60, 60],    # Partial overlap
        [100, 100, 120, 120] # No overlap
    ], dtype=torch.float32)
    iou_multiple = calculate_iou(box1, box2_multiple)
    print(f"Box2 (multiple): {box2_multiple}")
    print(f"IoU (multiple): {iou_multiple}") # Expected: [1.0, approx 0.3913, 0.0]
    assert torch.isclose(iou_multiple, torch.tensor([1.0, expected_iou_partial, 0.0])).all()
    
    # Case 5: box1 contains box2
    # box1: [10, 10, 50, 50], area 1600
    # box2_contained: [20, 20, 40, 40], area 20*20 = 400
    # Intersection: area of box2_contained = 400
    # Union: area of box1 = 1600
    # IoU = 400 / 1600 = 0.25
    box2_contained = torch.tensor([[20, 20, 40, 40]], dtype=torch.float32)
    iou_contained = calculate_iou(box1, box2_contained)
    print(f"Box2 (contained within box1): {box2_contained}")
    print(f"IoU (contained): {iou_contained}")
    assert torch.isclose(iou_contained, torch.tensor([400.0/1600.0])).all()

    # Case 6: box2 contains box1
    # box1: [20, 20, 40, 40], area 400
    # box2_contains: [10, 10, 50, 50], area 1600
    # Intersection: area of box1 = 400
    # Union: area of box2_contains = 1600
    # IoU = 400 / 1600 = 0.25
    box1_small = torch.tensor([20,20,40,40], dtype=torch.float32)
    box2_contains = torch.tensor([[10,10,50,50]], dtype=torch.float32)
    iou_contains = calculate_iou(box1_small, box2_contains)
    print(f"Box1_small: {box1_small}")
    print(f"Box2 (contains box1_small): {box2_contains}")
    print(f"IoU (contains): {iou_contains}")
    assert torch.isclose(iou_contains, torch.tensor([400.0/1600.0])).all()


    # 2. Test box_cxcywh_to_xyxy and box_xyxy_to_cxcywh
    print("\n--- Testing box_cxcywh_to_xyxy and box_xyxy_to_cxcywh ---")
    # cx, cy, w, h
    boxes_cxcywh = torch.tensor([[30, 30, 40, 40], [70, 80, 20, 30]], dtype=torch.float32)
    # Expected xyxy:
    # Box 1: xmin=30-20=10, ymin=30-20=10, xmax=30+20=50, ymax=30+20=50 -> [10, 10, 50, 50]
    # Box 2: xmin=70-10=60, ymin=80-15=65, xmax=70+10=80, ymax=80+15=95 -> [60, 65, 80, 95]
    expected_xyxy = torch.tensor([[10, 10, 50, 50], [60, 65, 80, 95]], dtype=torch.float32)
    
    converted_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    print(f"Original (cxcywh): {boxes_cxcywh}")
    print(f"Converted to xyxy: {converted_xyxy}")
    assert torch.allclose(converted_xyxy, expected_xyxy)

    converted_back_cxcywh = box_xyxy_to_cxcywh(converted_xyxy)
    print(f"Converted back to cxcywh: {converted_back_cxcywh}")
    assert torch.allclose(converted_back_cxcywh, boxes_cxcywh)


    # 3. Test non_max_suppression
    print("\n--- Testing non_max_suppression ---")
    # Example from torchvision.ops.nms documentation
    boxes_nms = torch.tensor([
        [0, 0, 10, 10],  # Box 0
        [1, 1, 11, 11],  # Box 1 (overlaps significantly with Box 0)
        [20, 20, 30, 30],# Box 2 (no overlap with Box 0 or 1)
        [0, 0, 9, 9],    # Box 3 (overlaps significantly with Box 0, lower score)
        [21, 21, 29, 29] # Box 4 (overlaps with Box 2, lower score)
    ], dtype=torch.float32)
    
    scores_nms = torch.tensor([0.9, 0.85, 0.7, 0.6, 0.75], dtype=torch.float32)
    iou_threshold_nms = 0.5

    # Expected behavior:
    # 1. Box 0 (score 0.9) is selected.
    # 2. Box 1 (score 0.85) vs Box 0: IoU is high. Box 1 suppressed.
    #    IoU([0,0,10,10], [1,1,11,11]): intersection area = (10-1)*(10-1) = 81
    #    area1 = 100, area2 = 100. Union = 100+100-81 = 119. IoU = 81/119 approx 0.68 > 0.5. Suppress.
    # 3. Box 3 (score 0.6) vs Box 0: IoU is high. Box 3 suppressed.
    #    IoU([0,0,10,10], [0,0,9,9]): intersection area = 9*9 = 81
    #    area1 = 100, area2 = 81. Union = 100+81-81 = 100. IoU = 81/100 = 0.81 > 0.5. Suppress.
    # 4. Box 2 (score 0.7) is selected (no overlap with Box 0).
    # 5. Box 4 (score 0.75) vs Box 2: IoU is high. Box 2 (lower score) would be compared.
    #    Actually, NMS sorts by score.
    #    Order by score: Box 0 (0.9), Box 1 (0.85), Box 4 (0.75), Box 2 (0.7), Box 3 (0.6)
    #    1. Keep Box 0 (0.9).
    #    2. Box 1 (0.85) vs Box 0: IoU > 0.5. Suppress Box 1.
    #    3. Box 4 (0.75) vs Box 0: No overlap. Keep Box 4 for now.
    #    4. Box 2 (0.7) vs Box 0: No overlap. vs Box 4:
    #       IoU([20,20,30,30], [21,21,29,29]): intersection area = (29-21)*(29-21) = 8*8 = 64
    #       area1 = 100, area2 = (29-21)*(29-21)=64. Union = 100+64-64 = 100. IoU = 64/100 = 0.64 > 0.5.
    #       Suppress Box 2 because Box 4 has higher score (0.75) and overlaps.
    #    5. Box 3 (0.6) vs Box 0: IoU > 0.5. Suppress Box 3. vs Box 4: No overlap.
    #    Kept: Box 0 (idx 0), Box 4 (idx 4)
    #    Wait, torchvision.ops.nms processes in the original order of scores if they are sorted.
    #    If scores are not sorted, it sorts them first.
    #    Let's re-check the example: scores are [0.9, 0.85, 0.7, 0.6, 0.75]
    #    Sorted indices by score (desc): 0 (0.9), 1 (0.85), 4 (0.75), 2 (0.7), 3 (0.6)

    #    1. Select Box 0 (score 0.9). Add 0 to keep list.
    #    2. Box 1 (score 0.85). IoU(Box1, Box0) = 0.68. Suppress Box 1.
    #    3. Box 4 (score 0.75). IoU(Box4, Box0) = 0. Suppress nothing. Add 4 to keep list.
    #    4. Box 2 (score 0.7). IoU(Box2, Box0) = 0. IoU(Box2, Box4) = 0.64. Suppress Box 2.
    #    5. Box 3 (score 0.6). IoU(Box3, Box0) = 0.81. Suppress Box 3. IoU(Box3, Box4)=0.
    #    Result should be indices [0, 4]

    kept_indices = non_max_suppression(boxes_nms, scores_nms, iou_threshold_nms)
    print(f"Original boxes:\n{boxes_nms}")
    print(f"Original scores: {scores_nms}")
    print(f"IoU threshold: {iou_threshold_nms}")
    print(f"Kept indices after NMS: {kept_indices}")
    
    # Expected: tensor([0, 4]) based on the logic above.
    # Let's verify.
    # Box 0 (0.9)
    # Box 1 (0.85) vs Box 0 (0.9) -> IoU approx 0.68 -> suppress Box 1
    # Box 4 (0.75) vs Box 0 (0.9) -> IoU 0 -> keep Box 4
    # Box 2 (0.7) vs Box 0 (0.9) -> IoU 0
    #             vs Box 4 (0.75) -> IoU approx 0.64 -> suppress Box 2 (because Box 4 has higher score)
    # Box 3 (0.6) vs Box 0 (0.9) -> IoU 0.81 -> suppress Box 3
    #             vs Box 4 (0.75) -> IoU 0
    # So, indices kept are 0 and 4.
    expected_kept_indices = torch.tensor([0, 4])
    assert torch.equal(kept_indices, expected_kept_indices), f"NMS output {kept_indices} does not match expected {expected_kept_indices}"
    
    # Test with another NMS case (boxes are identical)
    boxes_identical = torch.tensor([
        [0,0,10,10],
        [0,0,10,10],
        [0,0,10,10]
    ], dtype=torch.float32)
    scores_identical = torch.tensor([0.9, 0.8, 0.7])
    kept_identical = non_max_suppression(boxes_identical, scores_identical, 0.5)
    print(f"\nNMS with identical boxes, scores {scores_identical}, kept: {kept_identical}")
    # Expected: only the one with the highest score: [0]
    assert torch.equal(kept_identical, torch.tensor([0]))

    # Test NMS with disjoint boxes
    boxes_disjoint = torch.tensor([
        [0,0,10,10],
        [20,20,30,30]
    ], dtype=torch.float32)
    scores_disjoint = torch.tensor([0.9, 0.8])
    kept_disjoint = non_max_suppression(boxes_disjoint, scores_disjoint, 0.5)
    print(f"NMS with disjoint boxes, kept: {kept_disjoint}")
    # Expected: both boxes kept, sorted by score: [0, 1]
    assert torch.equal(kept_disjoint, torch.tensor([0, 1]))

    print("\nAll utility functions tested successfully.")


def matrix_iou(boxes1_xyxy, boxes2_xyxy):
    """
    Calculates a matrix of Intersection over Union (IoU) values between
    two sets of bounding boxes.

    Args:
        boxes1_xyxy (torch.Tensor): First set of bounding boxes, shape (M, 4),
                                 in [xmin, ymin, xmax, ymax] format.
        boxes2_xyxy (torch.Tensor): Second set of bounding boxes, shape (N, 4),
                                 in [xmin, ymin, xmax, ymax] format.

    Returns:
        torch.Tensor: An M x N matrix where element (i, j) is the IoU between
                      boxes1_xyxy[i] and boxes2_xyxy[j].
    """
    if boxes1_xyxy.numel() == 0 or boxes2_xyxy.numel() == 0:
        return torch.zeros(boxes1_xyxy.size(0), boxes2_xyxy.size(0), device=boxes1_xyxy.device)

    # Expand dimensions to allow broadcasting for intersection calculation
    # boxes1: (M, 1, 4)
    # boxes2: (1, N, 4)
    boxes1_expanded = boxes1_xyxy.unsqueeze(1)
    boxes2_expanded = boxes2_xyxy.unsqueeze(0)

    # Calculate intersection coordinates
    # Top-left corner of intersection
    inter_xmin = torch.max(boxes1_expanded[..., 0], boxes2_expanded[..., 0])
    inter_ymin = torch.max(boxes1_expanded[..., 1], boxes2_expanded[..., 1])
    # Bottom-right corner of intersection
    inter_xmax = torch.min(boxes1_expanded[..., 2], boxes2_expanded[..., 2])
    inter_ymax = torch.min(boxes1_expanded[..., 3], boxes2_expanded[..., 3])

    # Calculate intersection area
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    intersection_area = inter_width * inter_height # Shape: (M, N)

    # Calculate areas of individual boxes
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]) # Shape: (M,)
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]) # Shape: (N,)

    # Expand areas for broadcasting
    area1_expanded = area1.unsqueeze(1) # Shape: (M, 1)
    area2_expanded = area2.unsqueeze(0) # Shape: (1, N)

    # Calculate union area
    union_area = area1_expanded + area2_expanded - intersection_area # Shape: (M, N)

    # Calculate IoU
    iou = intersection_area / (union_area + 1e-6) # Add epsilon for numerical stability

    return iou


if __name__ == '__main__':
    # Existing tests ...
    print("\n--- Testing Utility Functions ---")

    # 1. Test calculate_iou
    print("\n--- Testing calculate_iou ---")
    box1 = torch.tensor([10, 10, 50, 50], dtype=torch.float32) # xmin, ymin, xmax, ymax (area 40*40 = 1600)

    # Case 1: Perfect overlap
    box2_perfect = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
    iou_perfect = calculate_iou(box1, box2_perfect)
    print(f"Box1: {box1}")
    print(f"Box2 (perfect overlap): {box2_perfect}")
    print(f"IoU (perfect overlap): {iou_perfect}") # Expected: [1.0]
    assert torch.isclose(iou_perfect, torch.tensor([1.0])).all()

    # Case 2: Partial overlap
    # Intersection: xmin=20, ymin=20, xmax=50, ymax=50 -> w=30, h=30 -> area=900
    # Union: area1 + area2 - intersection = 1600 (40*40) + 1600 (40*40) - 900 = 3200 - 900 = 2300
    # IoU = 900 / 2300 = 9/23 approx 0.3913
    box2_partial = torch.tensor([[20, 20, 60, 60]], dtype=torch.float32)
    iou_partial = calculate_iou(box1, box2_partial)
    print(f"Box2 (partial overlap): {box2_partial}")
    print(f"IoU (partial overlap): {iou_partial}")
    expected_iou_partial = 900.0 / (1600.0 + (40.0*40.0) - 900.0)
    assert torch.isclose(iou_partial, torch.tensor([expected_iou_partial])).all()


    # Case 3: No overlap
    box2_no_overlap = torch.tensor([[100, 100, 120, 120]], dtype=torch.float32)
    iou_no_overlap = calculate_iou(box1, box2_no_overlap)
    print(f"Box2 (no overlap): {box2_no_overlap}")
    print(f"IoU (no overlap): {iou_no_overlap}") # Expected: [0.0]
    assert torch.isclose(iou_no_overlap, torch.tensor([0.0])).all()

    # Case 4: Multiple boxes in box2
    box2_multiple = torch.tensor([
        [10, 10, 50, 50],    # Perfect overlap
        [20, 20, 60, 60],    # Partial overlap
        [100, 100, 120, 120] # No overlap
    ], dtype=torch.float32)
    iou_multiple = calculate_iou(box1, box2_multiple)
    print(f"Box2 (multiple): {box2_multiple}")
    print(f"IoU (multiple): {iou_multiple}") # Expected: [1.0, approx 0.3913, 0.0]
    assert torch.isclose(iou_multiple, torch.tensor([1.0, expected_iou_partial, 0.0])).all()
    
    # Case 5: box1 contains box2
    # box1: [10, 10, 50, 50], area 1600
    # box2_contained: [20, 20, 40, 40], area 20*20 = 400
    # Intersection: area of box2_contained = 400
    # Union: area of box1 = 1600
    # IoU = 400 / 1600 = 0.25
    box2_contained = torch.tensor([[20, 20, 40, 40]], dtype=torch.float32)
    iou_contained = calculate_iou(box1, box2_contained)
    print(f"Box2 (contained within box1): {box2_contained}")
    print(f"IoU (contained): {iou_contained}")
    assert torch.isclose(iou_contained, torch.tensor([400.0/1600.0])).all()

    # Case 6: box2 contains box1
    # box1: [20, 20, 40, 40], area 400
    # box2_contains: [10, 10, 50, 50], area 1600
    # Intersection: area of box1 = 400
    # Union: area of box2_contains = 1600
    # IoU = 400 / 1600 = 0.25
    box1_small = torch.tensor([20,20,40,40], dtype=torch.float32)
    box2_contains = torch.tensor([[10,10,50,50]], dtype=torch.float32)
    iou_contains = calculate_iou(box1_small, box2_contains)
    print(f"Box1_small: {box1_small}")
    print(f"Box2 (contains box1_small): {box2_contains}")
    print(f"IoU (contains): {iou_contains}")
    assert torch.isclose(iou_contains, torch.tensor([400.0/1600.0])).all()


    # 2. Test box_cxcywh_to_xyxy and box_xyxy_to_cxcywh
    print("\n--- Testing box_cxcywh_to_xyxy and box_xyxy_to_cxcywh ---")
    # cx, cy, w, h
    boxes_cxcywh = torch.tensor([[30, 30, 40, 40], [70, 80, 20, 30]], dtype=torch.float32)
    # Expected xyxy:
    # Box 1: xmin=30-20=10, ymin=30-20=10, xmax=30+20=50, ymax=30+20=50 -> [10, 10, 50, 50]
    # Box 2: xmin=70-10=60, ymin=80-15=65, xmax=70+10=80, ymax=80+15=95 -> [60, 65, 80, 95]
    expected_xyxy = torch.tensor([[10, 10, 50, 50], [60, 65, 80, 95]], dtype=torch.float32)
    
    converted_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    print(f"Original (cxcywh): {boxes_cxcywh}")
    print(f"Converted to xyxy: {converted_xyxy}")
    assert torch.allclose(converted_xyxy, expected_xyxy)

    converted_back_cxcywh = box_xyxy_to_cxcywh(converted_xyxy)
    print(f"Converted back to cxcywh: {converted_back_cxcywh}")
    assert torch.allclose(converted_back_cxcywh, boxes_cxcywh)


    # 3. Test non_max_suppression
    print("\n--- Testing non_max_suppression ---")
    # Example from torchvision.ops.nms documentation
    boxes_nms = torch.tensor([
        [0, 0, 10, 10],  # Box 0
        [1, 1, 11, 11],  # Box 1 (overlaps significantly with Box 0)
        [20, 20, 30, 30],# Box 2 (no overlap with Box 0 or 1)
        [0, 0, 9, 9],    # Box 3 (overlaps significantly with Box 0, lower score)
        [21, 21, 29, 29] # Box 4 (overlaps with Box 2, lower score)
    ], dtype=torch.float32)
    
    scores_nms = torch.tensor([0.9, 0.85, 0.7, 0.6, 0.75], dtype=torch.float32)
    iou_threshold_nms = 0.5
    kept_indices = non_max_suppression(boxes_nms, scores_nms, iou_threshold_nms)
    print(f"Original boxes:\n{boxes_nms}")
    print(f"Original scores: {scores_nms}")
    print(f"IoU threshold: {iou_threshold_nms}")
    print(f"Kept indices after NMS: {kept_indices}")
    expected_kept_indices = torch.tensor([0, 4])
    assert torch.equal(kept_indices, expected_kept_indices), f"NMS output {kept_indices} does not match expected {expected_kept_indices}"

    # Test with another NMS case (boxes are identical)
    boxes_identical = torch.tensor([
        [0,0,10,10],
        [0,0,10,10],
        [0,0,10,10]
    ], dtype=torch.float32)
    scores_identical = torch.tensor([0.9, 0.8, 0.7])
    kept_identical = non_max_suppression(boxes_identical, scores_identical, 0.5)
    print(f"\nNMS with identical boxes, scores {scores_identical}, kept: {kept_identical}")
    assert torch.equal(kept_identical, torch.tensor([0]))

    # Test NMS with disjoint boxes
    boxes_disjoint = torch.tensor([
        [0,0,10,10],
        [20,20,30,30]
    ], dtype=torch.float32)
    scores_disjoint = torch.tensor([0.9, 0.8])
    kept_disjoint = non_max_suppression(boxes_disjoint, scores_disjoint, 0.5)
    print(f"NMS with disjoint boxes, kept: {kept_disjoint}")
    assert torch.equal(kept_disjoint, torch.tensor([0, 1]))

    print("\nAll utility function tests passed so far!")

    # 4. Test matrix_iou
    print("\n--- Testing matrix_iou ---")
    boxes_m1 = torch.tensor([
        [0,0,10,10], #b1
        [5,5,15,15]  #b2
    ], dtype=torch.float32)
    boxes_m2 = torch.tensor([
        [0,0,10,10], #b3 (iou(b1,b3)=1, iou(b2,b3)=25/((100+100)-25)=25/175=1/7)
        [20,20,30,30] #b4 (iou(b1,b4)=0, iou(b2,b4)=0)
    ], dtype=torch.float32)

    # Expected iou_matrix:
    #        b3      b4
    # b1   [1.0     0.0]
    # b2   [1/7     0.0]
    expected_iou_m = torch.tensor([
        [1.0, 0.0],
        [1.0/7.0, 0.0]
    ], dtype=torch.float32)
    
    iou_m = matrix_iou(boxes_m1, boxes_m2)
    print(f"Boxes M1:\n{boxes_m1}")
    print(f"Boxes M2:\n{boxes_m2}")
    print(f"IoU Matrix:\n{iou_m}")
    assert torch.allclose(iou_m, expected_iou_m, atol=1e-5)

    # Test with empty inputs
    iou_empty1 = matrix_iou(torch.empty(0,4), boxes_m2)
    assert iou_empty1.shape == (0, 2)
    iou_empty2 = matrix_iou(boxes_m1, torch.empty(0,4))
    assert iou_empty2.shape == (2, 0)
    iou_empty_all = matrix_iou(torch.empty(0,4), torch.empty(0,4))
    assert iou_empty_all.shape == (0,0)
    print("matrix_iou tests passed!")

    print("\nAll tests passed!")

except Exception as e:
    print(f"An error occurred during testing: {e}")
    import traceback
    traceback.print_exc()
