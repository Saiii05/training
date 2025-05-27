import torch
import torchvision.models as models
import torch.nn as nn

def get_resnet50_backbone():
    """
    Loads a pre-trained ResNet50 model and removes its final fully connected layer.

    Returns:
        torch.nn.Module: The modified ResNet50 model (feature extractor).
    """
    # Load a ResNet50 model pre-trained on ImageNet
    resnet50 = models.resnet50(pretrained=True)

    # Freeze all parameters in the model
    for param in resnet50.parameters():
        param.requires_grad = False

    # Remove the final fully connected layer
    # Method 1: Replace with an Identity layer
    # resnet50.fc = nn.Identity()

    # Method 2: Create a new sequential model without the last layer
    # This is generally more robust if the layer name 'fc' changes in future torchvision versions
    modules = list(resnet50.children())[:-1]
    backbone = nn.Sequential(*modules)

    return backbone

if __name__ == '__main__':
    # Example of how to use the function
    backbone = get_resnet50_backbone()
    print("ResNet50 backbone loaded successfully.")
    print("Output shape of a dummy input (batch_size=1, channels=3, height=224, width=224):")
    dummy_input = torch.randn(1, 3, 224, 224)
    output = backbone(dummy_input)
    print(output.shape) # Expected output shape: torch.Size([1, 2048, 1, 1]) or similar after global pooling
    # The output shape will be [batch_size, num_features, 1, 1] for ResNet50
    # as the nn.Sequential(*list(resnet50.children())[:-1]) removes the fc layer
    # but keeps the adaptive average pooling layer that reduces spatial dimensions to 1x1.
    # If you need a specific feature map size, further modifications might be needed
    # or you might tap into earlier layers.
    
    # To verify that the parameters are frozen:
    for name, param in backbone.named_parameters():
        if 'fc' not in name: # fc layer is removed, so we check other layers
            assert not param.requires_grad, f"Parameter {name} is not frozen!"
    print("All parameters (except FC layer which is removed) are frozen.")
