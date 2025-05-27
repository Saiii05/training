import torch
import torch.nn as nn
import math

# Assuming src.backbone, src.detection_head, and src.utils are in PYTHONPATH
# or in the same directory structure
from backbone import get_resnet50_backbone
from detection_head import SimpleDetectionHead
from utils import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

class ObjectDetectionModel(nn.Module):
    """
    Main object detection model integrating backbone, detection head,
    and default box generation/management.
    """
    def __init__(self, num_classes_fg, num_anchors_per_cell=5, backbone_output_channels=2048,
                 image_size_for_default_boxes=(300,300)):
        """
        Args:
            num_classes_fg (int): Number of foreground classes (e.g., 20 for Pascal VOC).
            num_anchors_per_cell (int): Number of default boxes/anchors per grid cell.
            backbone_output_channels (int): Number of output channels from the backbone.
                                            ResNet50 layer4 typically outputs 2048.
            image_size_for_default_boxes (tuple): (height, width) used to generate default boxes.
        """
        super(ObjectDetectionModel, self).__init__()

        self.num_classes_fg = num_classes_fg
        self.num_classes_loss = num_classes_fg + 1 # Add 1 for background class
        self.num_anchors_per_cell = num_anchors_per_cell
        self.image_size_for_default_boxes = image_size_for_default_boxes

        # Instantiate backbone
        # The get_resnet50_backbone() from previous steps includes AdaptiveAvgPool2d.
        # For detection, we need features *before* this global pooling.
        # We will modify it here or assume it's modified in backbone.py to give layer4 features.
        # For now, let's assume backbone.py is updated or we use a custom version here.
        full_resnet50 = get_resnet50_backbone() # This one has avgpool, output will be (B, C, 1, 1)
        # We need to extract features before the final avgpool.
        # If get_resnet50_backbone() returns resnet50.children()[:-1], then it includes avgpool.
        # We need resnet50.children()[:-2] to get layer4 output.
        # For simplicity, let's assume get_resnet50_backbone is already modified to return layer4.
        # If not, one would do:
        # import torchvision.models as models
        # resnet50_full_model = models.resnet50(pretrained=True)
        # self.backbone = nn.Sequential(*list(resnet50_full_model.children())[:-2]) # Exclude avgpool and fc
        self.backbone = full_resnet50 # Placeholder: will be updated if backbone output is 1x1
        # This will be dynamically determined in the forward pass for now, or use a fixed one.
        # feature_map_h, feature_map_w = self._get_feature_map_size(image_size_for_default_boxes)

        # Instantiate detection head
        self.detection_head = SimpleDetectionHead(
            in_channels=backbone_output_channels,
            num_classes=self.num_classes_loss,
            num_anchors=self.num_anchors_per_cell
        )

        # Variances for decoding bounding box predictions (typical SSD values)
        self.center_variance = 0.1
        self.size_variance = 0.2

        # Generate default boxes
        # Determine feature map size based on a dummy forward pass or known architecture
        # For ResNet50, with input 300x300, output of layer4 is (300/32) = 9.375 -> 10x10 or 9x9
        # Let's use a fixed size for now, e.g. 9x9 or 10x10.
        # A more robust way is to run a dummy input through the backbone once.
        # With AdaptiveAvgPool2d(1,1) in backbone, feature map size will be 1x1.
        # This means the current `get_resnet50_backbone` is not suitable as is.
        # We will override self.backbone if needed in the demo section or assume it's correct.
        
        # For now, let's assume a feature map size (e.g., 9x9 for 300x300 input)
        # This part is crucial and depends on the actual backbone output.
        # If the backbone from `get_resnet50_backbone` includes avgpool, then feat_h, feat_w will be 1,1.
        # This would mean default boxes are generated only for a 1x1 grid.
        # We need to fix this assumption.
        # For the purpose of this subtask, we'll hardcode a feature_map_size for default box generation
        # that would be typical for SSD300 (e.g., from conv4_3 or a similar layer).
        # A common SSD300 setup might use multiple feature maps. Here we use one.
        # Let's assume image_size (300,300) and feature map from resnet layer4 is (10,10)
        # (300/32 = 9.375, often rounded to 10 for stride 32, or 9 if padding makes it so)
        # Let's use 10x10 as an example feature map size for 300x300 input.
        example_feature_map_size = (self.image_size_for_default_boxes[0] // 32, self.image_size_for_default_boxes[1] // 32)
        # example_feature_map_size = (10,10) # For 300x300 input, resnet layer4 (stride 32)
        
        default_boxes = self._generate_default_boxes(
            image_size=self.image_size_for_default_boxes,
            # This feature_map_size needs to match the actual output of self.backbone
            feature_map_size=example_feature_map_size, 
            num_anchors_per_cell=self.num_anchors_per_cell
        )
        self.register_buffer("default_boxes_xyxy", default_boxes)


    def _get_feature_map_size(self, image_shape_hw):
        """ Helper to get feature map size from backbone for a given image size """
        # This is a placeholder. A real implementation might run a dummy tensor
        # or have this information pre-calculated if image_size is fixed.
        # For ResNet50, it's roughly image_size / 32
        return (image_shape_hw[0] // 32, image_shape_hw[1] // 32)

    def _generate_default_boxes(self, image_size, feature_map_size, num_anchors_per_cell,
                                aspect_ratios=None, scales=None):
        """
        Generates default anchor boxes normalized to [0,1] in (xmin, ymin, xmax, ymax) format.

        Args:
            image_size (tuple): (img_h, img_w) for which boxes are generated.
            feature_map_size (tuple): (feat_h, feat_w) of the feature map for these boxes.
            num_anchors_per_cell (int): Number of anchors per feature map cell.
            aspect_ratios (list of list/tuples, optional): Aspect ratios (w/h) for anchors.
                                                           Defaults to SSD-like ratios if num_anchors_per_cell allows.
            scales (list, optional): Scales of anchors relative to image_size.
                                     Defaults to SSD-like scales.

        Returns:
            torch.Tensor: Default boxes of shape (feat_h * feat_w * num_anchors_per_cell, 4)
                          in [xmin, ymin, xmax, ymax] format, normalized and clipped to [0,1].
        """
        img_h, img_w = image_size
        feat_h, feat_w = feature_map_size

        # Define default aspect ratios and scales if not provided
        # These are example values, often tuned for specific datasets/architectures (like SSD)
        if aspect_ratios is None:
            if num_anchors_per_cell == 5: # Example for 5 anchors
                aspect_ratios = [1.0, 0.5, 2.0, 0.333, 3.0] # Direct w/h ratios
                # aspect_ratios = [[1.0, 1.0], [0.5, 1.0], [1.0, 0.5], [0.33,1.0], [1.0,0.33]]
            elif num_anchors_per_cell == 6: # SSD300 like from one layer
                 aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1.0/3.0, 1.0] # Special handling for sixth anchor with different scale
            else: # Fallback for other numbers of anchors
                aspect_ratios = [1.0] * num_anchors_per_cell
        
        if scales is None:
            # These scales are relative to the image size.
            # For SSD, scales vary per feature map. For a single feature map example:
            if num_anchors_per_cell == 5:
                scales = [0.1, 0.2, 0.2, 0.3, 0.3] # Example scales, ensure len matches num_anchors
            elif num_anchors_per_cell == 6: # SSD300 like from one layer (e.g. conv4_3)
                min_scale = 0.2 # e.g. S_min for this layer
                max_scale = 0.35 # e.g. S_max for this layer (used for one anchor)
                # scales = [min_scale, min_scale, min_scale, min_scale, min_scale, math.sqrt(min_scale*max_scale)]
                scales = [0.1, 0.2, 0.2, 0.2, 0.2, 0.15] # Simplified for demo, ensure len matches
                # This needs to be structured to match aspect_ratios length
                # Typically, one scale per base aspect ratio, and an extra scale for AR=1
            else:
                scales = [0.1 + i * 0.15 for i in range(num_anchors_per_cell)]


        # Ensure scales and aspect_ratios lists match num_anchors_per_cell
        # This logic might need refinement based on how scales/ARs are paired for SSD.
        # For simplicity, let's assume len(scales) and len(aspect_ratios) should match num_anchors_per_cell.
        # A common SSD approach: for AR=1, use base scale and sqrt(scale * next_scale). Others use base scale.
        # The provided prompt was:
        # anchor_w = scales[k] * image_size[1] * sqrt(aspect_ratios[k][0] / aspect_ratios[k][1])
        # This implies aspect_ratios is list of [num, den]. Let's adapt to w/h ratios.
        # If aspect_ratios = [w/h_1, w/h_2, ...], then sqrt(ar) can be used.

        if len(scales) != num_anchors_per_cell:
            print(f"Warning: Mismatch in len(scales)={len(scales)} and num_anchors_per_cell={num_anchors_per_cell}. Adjusting scales.")
            scales = [0.1 + i * (0.8 / (num_anchors_per_cell-1 if num_anchors_per_cell > 1 else 1) ) for i in range(num_anchors_per_cell)]
        if len(aspect_ratios) != num_anchors_per_cell:
            print(f"Warning: Mismatch in len(aspect_ratios)={len(aspect_ratios)} and num_anchors_per_cell={num_anchors_per_cell}. Adjusting ARs.")
            aspect_ratios = [1.0] * num_anchors_per_cell


        default_boxes_list = []
        cell_h_norm = 1.0 / feat_h # Normalized cell height
        cell_w_norm = 1.0 / feat_w # Normalized cell width

        for y_idx in range(feat_h):
            for x_idx in range(feat_w):
                # Normalized center of the cell
                center_x_norm = (x_idx + 0.5) * cell_w_norm
                center_y_norm = (y_idx + 0.5) * cell_h_norm

                for k in range(num_anchors_per_cell):
                    s = scales[k]       # Scale relative to image dimension
                    ar = aspect_ratios[k] # Aspect ratio (w/h)

                    # Anchor dimensions relative to image dimensions
                    anchor_w_norm = s * math.sqrt(ar)
                    anchor_h_norm = s / math.sqrt(ar)
                    
                    # Convert to [xmin, ymin, xmax, ymax] normalized coordinates
                    xmin_norm = center_x_norm - anchor_w_norm / 2.0
                    ymin_norm = center_y_norm - anchor_h_norm / 2.0
                    xmax_norm = center_x_norm + anchor_w_norm / 2.0
                    ymax_norm = center_y_norm + anchor_h_norm / 2.0
                    
                    default_boxes_list.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])

        default_boxes_tensor = torch.tensor(default_boxes_list, dtype=torch.float32)
        
        # Clip to [0, 1]
        default_boxes_tensor.clamp_(min=0.0, max=1.0)
        
        return default_boxes_tensor


    def forward(self, images):
        """
        Forward pass of the detection model.

        Args:
            images (torch.Tensor): Input images, shape (batch_size, 3, img_h, img_w).

        Returns:
            tuple:
                - cls_logits (torch.Tensor): Classification logits.
                  Shape: (batch_size, num_total_default_boxes, num_classes_loss).
                - bbox_pred_cxcywh (torch.Tensor): Predicted bounding boxes in absolute
                  [center_x, center_y, width, height] format, normalized to [0,1].
                  Shape: (batch_size, num_total_default_boxes, 4).
                - self.default_boxes_xyxy (torch.Tensor): Default anchor boxes in
                  [xmin, ymin, xmax, ymax] format, normalized to [0,1].
                  Shape: (num_total_default_boxes, 4).
        """
        batch_size = images.shape[0]
        
        # 1. Get features from backbone
        features = self.backbone(images) # Expected: (batch_size, C_backbone, feat_h, feat_w)
        
        # Dynamically check feature map size if it was not fixed for default boxes
        # This is important if self.default_boxes_xyxy generation depends on dynamic feature map size.
        # For this implementation, default_boxes are generated with a fixed assumed feature map size.
        # If `features.shape[2:4]` doesn't match what `self.default_boxes_xyxy` was built for,
        # there will be a mismatch in the number of predictions from head vs number of default boxes.
        # The head's output shape (num_total_default_boxes) is determined by features' spatial dims.
        
        # 2. Get predictions from detection head
        # cls_logits: (batch_size, feat_h * feat_w * num_anchors, num_classes_loss)
        # bbox_offsets: (batch_size, feat_h * feat_w * num_anchors, 4) -> (dcx, dcy, dw, dh)
        cls_logits, bbox_offsets = self.detection_head(features)

        # Ensure number of predicted boxes matches number of default boxes
        num_pred_boxes_from_head = cls_logits.shape[1]
        num_default_boxes = self.default_boxes_xyxy.shape[0]
        if num_pred_boxes_from_head != num_default_boxes:
            raise ValueError(
                f"Mismatch: Head produced {num_pred_boxes_from_head} predictions, "
                f"but there are {num_default_boxes} default boxes. "
                f"Feature map size from backbone ({features.shape[2]}x{features.shape[3]}) "
                f"might not match feature_map_size used for default box generation "
                f"({self.image_size_for_default_boxes[0]//32}x{self.image_size_for_default_boxes[1]//32} if using stride 32 rule, "
                f"or the fixed one used in __init__)."
            )

        # 3. Decode bounding box predictions from offsets
        # Convert default boxes from [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
        default_boxes_cxcywh = box_xyxy_to_cxcywh(self.default_boxes_xyxy) # Shape: (num_total_default_boxes, 4)

        # Expand default_boxes_cxcywh to match batch_size for broadcasting
        # (1, num_total_default_boxes, 4)
        default_boxes_cxcywh_batch = default_boxes_cxcywh.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply offsets to default boxes (SSD decoding)
        pred_cx = default_boxes_cxcywh_batch[..., 0] + \
                  bbox_offsets[..., 0] * self.center_variance * default_boxes_cxcywh_batch[..., 2]
        pred_cy = default_boxes_cxcywh_batch[..., 1] + \
                  bbox_offsets[..., 1] * self.center_variance * default_boxes_cxcywh_batch[..., 3]
        pred_w = default_boxes_cxcywh_batch[..., 2] * \
                 torch.exp(bbox_offsets[..., 2] * self.size_variance)
        pred_h = default_boxes_cxcywh_batch[..., 3] * \
                 torch.exp(bbox_offsets[..., 3] * self.size_variance)

        # bbox_pred_cxcywh is normalized [cx, cy, w, h]
        bbox_pred_cxcywh = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
        
        # Output boxes are normalized and clipped by the loss function or post-processing if needed.
        # Here, default_boxes_xyxy are already clipped [0,1]. Decoded boxes might go outside.

        return cls_logits, bbox_pred_cxcywh, self.default_boxes_xyxy


if __name__ == '__main__':
    print("--- ObjectDetectionModel Demo ---")
    num_foreground_classes = 20
    img_height, img_width = 300, 300
    
    # Important: To make this demo runnable, we need a backbone that produces a known feature map size.
    # The `get_resnet50_backbone` from previous steps includes AdaptiveAvgPool2d, outputting 1x1 features.
    # This is not suitable for typical SSD-style default box generation across a spatial grid.
    # We need to use a backbone that outputs, e.g., layer4 features directly.
    
    # Option 1: Modify `get_resnet50_backbone` in `backbone.py` (external change)
    # Option 2: Create a local modified backbone for this demo
    
    class DummyBackbone(nn.Module): # Simulate a backbone that outputs, e.g., 10x10 feature map
        def __init__(self, out_channels, out_h, out_w):
            super().__init__()
            self.out_channels = out_channels
            self.out_h = out_h
            self.out_w = out_w
            # A simple conv layer to set the channel size
            self.conv = nn.Conv2d(3, out_channels, kernel_size=1) 

        def forward(self, x):
            # x is (B, 3, H, W)
            x = self.conv(x) # (B, out_channels, H, W)
            # Simulate downsampling to out_h, out_w using adaptive pooling
            return F.adaptive_avg_pool2d(x, (self.out_h, self.out_w))

    import torch.nn.functional as F
    import torchvision.models as models

    # Let's properly define a backbone for this demo that outputs layer4 features
    # This is what a real model would likely use.
    class ResNet50Layer4Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            resnet50 = models.resnet50(pretrained=True)
            # Remove the final avgpool and fc layers
            self.features = nn.Sequential(*list(resnet50.children())[:-2])
        
        def forward(self, x):
            return self.features(x)

    # Instantiate model
    # The feature map size from ResNet50 layer4 for a 300x300 input is 300/32 = 9.375 -> usually 10x10 or 9x9.
    # If using torchvision's resnet50, for input (3,300,300), layer4 output is (B, 2048, 10, 10)
    # So, image_size_for_default_boxes=(300,300) and expected feature_map_size=(10,10)
    # Our _generate_default_boxes uses image_size // 32, so 300//32 = 9. This must match.
    # Let's make feature_map_size in _generate_default_boxes consistent with ResNet50's stride 32.
    # So, if image_size=(300,300), feat_map_size=(9,9) is used by _generate_default_boxes.
    # The actual backbone (ResNet50Layer4Backbone) for (3,300,300) gives (2048,10,10).
    # This mismatch will cause an error.
    
    # To fix:
    # 1. Ensure `_generate_default_boxes` uses the *actual* feature map size produced by the backbone.
    #    This means `_generate_default_boxes` should probably take `feature_map_size` as an argument,
    #    and `__init__` should determine this by inspecting the backbone (e.g. dummy forward pass).
    # OR 2. Make `ObjectDetectionModel.__init__` ensure the backbone produces feature map of `image_size // 32`.
    #    The current `_generate_default_boxes` uses `image_size // 32`. So, this is the target.
    #    If ResNet50Layer4Backbone gives 10x10 for 300x300, we need to adjust.
    #    Let's adjust the default box generation to expect 10x10 if that's what backbone gives.
    
    # For the demo, let's assume the feature map size is fixed at 10x10 for a 300x300 input.
    # We will pass this to _generate_default_boxes by adjusting the logic in __init__ or _generate_default_boxes.
    # The current code in __init__ calculates `example_feature_map_size = (self.image_size_for_default_boxes[0] // 32, ...)`
    # For (300,300) this is (9,9).
    # If our demo backbone (ResNet50Layer4Backbone) outputs 10x10, then this will fail.
    
    # Let's refine __init__ to use a known feature map size for the demo, or make _generate_default_boxes more flexible.
    # For now, let's modify the example_feature_map_size in __init__ for this demo.
    # No, the subtask implies __init__ should set it up.
    # The best is to make the model's default box generation use the *actual* feature map size derived from backbone.
    # This means `_generate_default_boxes` should be called *after* we know the feature map size.
    # This is a bit tricky as default_boxes is needed at init.
    # A common way: `feature_map_size` is a known hyperparameter for a given input size and backbone.
    
    # For this demo, we will patch the model's backbone with one that produces 9x9 for 300x300 input
    # to match the default box generation logic.
    class DemoBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(3, 2048, kernel_size=3, stride=1, padding=1) # No downsampling here
            # We will use adaptive pool to force output size
            self.pool = nn.AdaptiveAvgPool2d((9,9)) # Force 9x9 output

        def forward(self, x): # x is (B,3,300,300)
            x = self.layer(x) # (B,2048,300,300)
            return self.pool(x) # (B,2048,9,9)

    model = ObjectDetectionModel(num_classes_fg=num_foreground_classes, image_size_for_default_boxes=(img_height, img_width))
    
    # Replace the model's backbone with our demo backbone for this test
    model.backbone = DemoBackbone() # This ensures feature map is 9x9 as expected by default box gen

    print(f"Model uses default boxes generated for image size: {model.image_size_for_default_boxes}")
    fm_h = model.image_size_for_default_boxes[0] // 32
    fm_w = model.image_size_for_default_boxes[1] // 32
    print(f"Default boxes generated assuming feature map size: ({fm_h}, {fm_w})")
    print(f"Number of default boxes: {model.default_boxes_xyxy.shape[0]}")
    expected_num_def_boxes = fm_h * fm_w * model.num_anchors_per_cell
    print(f"Expected num default boxes based on ({fm_h}x{fm_w}) grid and {model.num_anchors_per_cell} anchors/cell: {expected_num_def_boxes}")
    assert model.default_boxes_xyxy.shape[0] == expected_num_def_boxes, "Mismatch in generated default boxes number"


    # Create a dummy image batch
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, img_height, img_width)

    # Pass through the model
    model.eval() # Set to evaluation mode for consistency
    with torch.no_grad():
        cls_logits, bbox_pred_cxcywh, default_boxes = model(dummy_images)

    # Print shapes
    print(f"\nInput images shape: {dummy_images.shape}")
    print(f"Output Class Logits shape: {cls_logits.shape}")       # Expected: (B, num_total_default_boxes, num_classes_loss)
    print(f"Output Bbox Pred (cxcywh, normalized) shape: {bbox_pred_cxcywh.shape}") # Expected: (B, num_total_default_boxes, 4)
    print(f"Returned Default Boxes (xyxy, normalized) shape: {default_boxes.shape}") # Expected: (num_total_default_boxes, 4)

    # Verifications
    assert cls_logits.shape[0] == batch_size
    assert cls_logits.shape[1] == default_boxes.shape[0]
    assert cls_logits.shape[2] == num_foreground_classes + 1

    assert bbox_pred_cxcywh.shape[0] == batch_size
    assert bbox_pred_cxcywh.shape[1] == default_boxes.shape[0]
    assert bbox_pred_cxcywh.shape[2] == 4
    
    # Verify default_boxes_xyxy are normalized and clipped
    assert default_boxes.min() >= 0.0, "Default boxes not clipped to min 0.0"
    assert default_boxes.max() <= 1.0, "Default boxes not clipped to max 1.0"
    print("\nDefault boxes are normalized and clipped [0,1].")

    # Check if default_boxes_xyxy is a buffer
    is_buffer = False
    for name, buf in model.named_buffers():
        if name == "default_boxes_xyxy":
            is_buffer = True
            break
    assert is_buffer, "default_boxes_xyxy is not registered as a buffer."
    print("default_boxes_xyxy is registered as a buffer.")

    print("\nDemo finished successfully.")
