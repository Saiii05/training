import torch
import torch.nn as nn

# Assuming backbone.py is in the same directory or PYTHONPATH is set
# from .backbone import get_resnet50_backbone
# For standalone execution if backbone.py is one level up and src is in PYTHONPATH
# from ..src.backbone import get_resnet50_backbone
# For the specific structure where this script is in src/ and backbone.py is also in src/
from backbone import get_resnet50_backbone


class SimpleDetectionHead(nn.Module):
    """
    A simple detection head for object detection.

    It takes feature maps from a backbone network and outputs classification
    scores and bounding box predictions.
    """
    def __init__(self, in_channels, num_classes, num_anchors):
        """
        Initializes the SimpleDetectionHead.

        Args:
            in_channels (int): Number of input channels from the backbone.
                               (e.g., 2048 for ResNet50's last conv block).
            num_classes (int): Number of classes to predict (excluding background).
            num_anchors (int): Number of anchors per spatial location.
        """
        super(SimpleDetectionHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Intermediate convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Classification head: 1x1 conv layer
        # Outputs num_anchors * num_classes channels for classification scores
        self.classification_head = nn.Conv2d(512, num_anchors * num_classes, kernel_size=1, stride=1, padding=0)

        # Regression head: 1x1 conv layer
        # Outputs num_anchors * 4 channels for bounding box predictions (dx, dy, dw, dh)
        self.regression_head = nn.Conv2d(512, num_anchors * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, feature_map):
        """
        Forward pass of the detection head.

        Args:
            feature_map (torch.Tensor): Feature map from the backbone network.
                                        Shape: (batch_size, in_channels, grid_h, grid_w).

        Returns:
            tuple:
                - cls_logits (torch.Tensor): Classification logits.
                  Shape: (batch_size, grid_h * grid_w * num_anchors, num_classes).
                - bbox_pred (torch.Tensor): Bounding box predictions.
                  Shape: (batch_size, grid_h * grid_w * num_anchors, 4).
        """
        batch_size, _, grid_h, grid_w = feature_map.shape

        # Pass through the intermediate conv layer
        # Input: (batch_size, in_channels, grid_h, grid_w)
        # Output: (batch_size, 512, grid_h, grid_w)
        intermediate_features = self.conv1(feature_map)

        # Classification logits
        # Input: (batch_size, 512, grid_h, grid_w)
        # Output: (batch_size, num_anchors * num_classes, grid_h, grid_w)
        cls_logits = self.classification_head(intermediate_features)

        # Reshape classification logits
        # (batch_size, num_anchors * num_classes, grid_h, grid_w) ->
        # (batch_size, grid_h, grid_w, num_anchors * num_classes)
        cls_logits = cls_logits.permute(0, 2, 3, 1)
        # (batch_size, grid_h, grid_w, num_anchors * num_classes) ->
        # (batch_size, grid_h * grid_w * num_anchors, num_classes)
        cls_logits = cls_logits.reshape(batch_size, grid_h * grid_w * self.num_anchors, self.num_classes)

        # Bounding box predictions
        # Input: (batch_size, 512, grid_h, grid_w)
        # Output: (batch_size, num_anchors * 4, grid_h, grid_w)
        bbox_pred = self.regression_head(intermediate_features)

        # Reshape bounding box predictions
        # (batch_size, num_anchors * 4, grid_h, grid_w) ->
        # (batch_size, grid_h, grid_w, num_anchors * 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        # (batch_size, grid_h, grid_w, num_anchors * 4) ->
        # (batch_size, grid_h * grid_w * num_anchors, 4)
        bbox_pred = bbox_pred.reshape(batch_size, grid_h * grid_w * self.num_anchors, 4)

        return cls_logits, bbox_pred


if __name__ == '__main__':
    # Example usage:
    print("--- SimpleDetectionHead Example Usage ---")

    # 1. Instantiate ResNet50 backbone
    # The backbone from src.backbone.py is ResNet50 without the FC layer,
    # which acts as a feature extractor.
    # The output of resnet50.children()[:-1] is the output of the last conv block (layer4)
    # followed by AdaptiveAvgPool2d(output_size=(1,1)).
    # For detection, we typically want the feature map *before* global average pooling.
    # Let's modify the backbone loading slightly for this example to get layer4's output.

    # For this example, we'll assume the `get_resnet50_backbone` from `backbone.py`
    # indeed gives features before any final pooling that would make it 1x1.
    # If it includes AdaptiveAvgPool2d, the spatial dimensions (grid_h, grid_w) would be 1x1.
    # The original `get_resnet50_backbone` in the previous step uses `nn.Sequential(*list(resnet50.children())[:-1])`
    # which means it takes all children of resnet50 except the last one (the fc layer).
    # The layer before 'fc' in ResNet50 is 'avgpool', which is an AdaptiveAvgPool2d layer.
    # So the output of that backbone would be (batch_size, 2048, 1, 1).
    # For a detection head, we usually want richer spatial features.
    # Let's assume for this example, we want features from 'layer4' of ResNet.
    # We will simulate this by taking the output of the provided backbone.
    # If backbone() outputs (batch_size, 2048, 1, 1), then grid_h=1, grid_w=1.
    # For a more realistic scenario, one would tap into an earlier layer or modify the backbone.
    
    # For now, let's use the provided backbone and see its output.
    # The `get_resnet50_backbone` already removes the FC layer and includes avgpool.
    # So the output spatial dimension will be 1x1.
    
    # Let's re-define backbone for this test to output layer4 features (before avgpool)
    import torchvision.models as models
    resnet50_full = models.resnet50(pretrained=True)
    # Typically, for detection, we take features from before the final average pooling layer.
    # This corresponds to resnet50.layer4
    example_backbone = nn.Sequential(*list(resnet50_full.children())[:-2]) # Exclude fc and avgpool

    # 2. Create a dummy input tensor
    batch_size = 2
    # Input image size (example: 256x256)
    # If input is (3, 256, 256), ResNet50 layer4 output is (2048, 8, 8)
    # because 256 / (2^5) = 256 / 32 = 8 (5 downsamples: conv1, layer1, layer2, layer3, layer4)
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    print(f"Dummy input shape: {dummy_input.shape}")

    # 3. Pass dummy input through the example backbone (ResNet50's layer4)
    example_backbone.eval() # Set to evaluation mode
    with torch.no_grad(): # No need to track gradients for this example
        backbone_features = example_backbone(dummy_input)
    # Expected output from layer4 of ResNet50 for a 256x256 input: (batch_size, 2048, 8, 8)
    print(f"Backbone feature map shape: {backbone_features.shape}")

    # 4. Instantiate SimpleDetectionHead
    in_channels_from_backbone = backbone_features.shape[1] # Should be 2048 for ResNet50 layer4
    num_classes_to_predict = 20 # Example: 20 object classes
    num_anchors_per_location = 5 # Example: 5 anchors

    detection_head = SimpleDetectionHead(
        in_channels=in_channels_from_backbone,
        num_classes=num_classes_to_predict,
        num_anchors=num_anchors_per_location
    )
    detection_head.eval() # Set to evaluation mode

    # 5. Pass the backbone's output feature map to the detection head
    with torch.no_grad():
        cls_logits, bbox_pred = detection_head(backbone_features)

    # 6. Print the shapes of the returned classification logits and bounding box predictions
    # Expected cls_logits shape: (batch_size, grid_h * grid_w * num_anchors, num_classes)
    #   grid_h = 8, grid_w = 8 for this example
    #   (2, 8 * 8 * 5, 20) = (2, 320, 20)
    print(f"Classification logits shape: {cls_logits.shape}")

    # Expected bbox_pred shape: (batch_size, grid_h * grid_w * num_anchors, 4)
    #   (2, 8 * 8 * 5, 4) = (2, 320, 4)
    print(f"Bounding box predictions shape: {bbox_pred.shape}")

    # Verify output values are not all NaN or Inf (basic sanity check)
    assert not torch.isnan(cls_logits).any(), "cls_logits contains NaN values"
    assert not torch.isinf(cls_logits).any(), "cls_logits contains Inf values"
    assert not torch.isnan(bbox_pred).any(), "bbox_pred contains NaN values"
    assert not torch.isinf(bbox_pred).any(), "bbox_pred contains Inf values"
    print("Output tensors seem valid (no NaNs or Infs).")

    # Test with the backbone from src.backbone
    # This backbone includes the avgpool layer, so output will be (batch_size, 2048, 1, 1)
    print("\n--- Testing with backbone from src.backbone (includes avgpool) ---")
    official_backbone = get_resnet50_backbone() # This is (resnet_children_except_fc)
    official_backbone.eval()
    with torch.no_grad():
        backbone_features_official = official_backbone(dummy_input)
    print(f"Official backbone feature map shape: {backbone_features_official.shape}") # e.g., (2, 2048, 1, 1) for ResNet50

    in_channels_official = backbone_features_official.shape[1]
    detection_head_official = SimpleDetectionHead(
        in_channels=in_channels_official,
        num_classes=num_classes_to_predict,
        num_anchors=num_anchors_per_location
    )
    detection_head_official.eval()
    with torch.no_grad():
        cls_logits_official, bbox_pred_official = detection_head_official(backbone_features_official)
    
    # grid_h = 1, grid_w = 1
    # Expected cls_logits_official shape: (batch_size, 1 * 1 * num_anchors, num_classes)
    #   (2, 1 * 1 * 5, 20) = (2, 5, 20)
    print(f"Official Classification logits shape: {cls_logits_official.shape}")
    # Expected bbox_pred_official shape: (batch_size, 1 * 1 * num_anchors, 4)
    #   (2, 1 * 1 * 5, 4) = (2, 5, 4)
    print(f"Official Bounding box predictions shape: {bbox_pred_official.shape}")
    assert cls_logits_official.shape == (batch_size, 1 * 1 * num_anchors_per_location, num_classes_to_predict)
    assert bbox_pred_official.shape == (batch_size, 1 * 1 * num_anchors_per_location, 4)
    print("Shapes with official backbone (including avgpool) are as expected.")
