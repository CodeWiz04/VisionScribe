from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

class CNNEncoder:
    """
    Extract image features using a pretrained ResNet-50.
    """
    def __init__(self, device: torch.device):
        self.device = device
        # Load ResNet-50 pretrained on ImageNet.
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(
            weights=weights
        )
        # Remove the final classification layer.
        #
        # Original:
        # ResNet-50 → 1000 ImageNet classes
        #
        # We want:
        # ResNet-50 → 2048 visual features
        self.model.fc = nn.Identity()   #pass the input as it is
        
        #Freeze Resnet
        for parameter in self.model.parameters():
            parameter.requires_grad = False
            
        self.model = self.model.to(self.device)    #move resnet to the device which we are using
        # Evaluation mode because we are using ResNet
        # only as a feature extractor.
        self.model.eval()
        # Use the preprocessing associated with the
        # pretrained ResNet-50 weights.
        self.transform = weights.transforms()       #resize,crop,convert to tensor,normalize,resnet50    
        
    def extract(self, image_path: Path) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not load image: {image_path}"
            ) from error
        image_tensor = self.transform(image)
        # Add batch dimension.
        #
        # Before:
        # [3, 224, 224]
        #
        # After:
        # [1, 3, 224, 224]
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        # We are not training the CNN, so gradients
        # are unnecessary.
        with torch.no_grad():

            features = self.model(image_tensor)
        # Remove the batch dimension.
        #
        # [1, 2048] → [2048]
        features = features.squeeze(0)

        return features.cpu()
    
def cache_features(
    encoder,
    image_dir,
    output_dir
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        image_dir.glob("*.jpg")
    )

    print(f"Found {len(image_paths)} images.")

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        output_path = (
            output_dir
            / f"{image_path.stem}.pt"
        )

        # Don't recompute features that already exist.
        if output_path.exists():
            continue

        features = encoder.extract(
            image_path
        )

        torch.save(
            features,
            output_path
        )

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(image_paths)}"
            )
if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    IMAGE_DIR = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "Images"
    )

    # Use one known Flickr8k image.
    image_path = (
        IMAGE_DIR
        / "1000268201_693b08cb0e.jpg"
    )

    # Use GPU if available, otherwise CPU.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    encoder = CNNEncoder(
        device=device
    )

    features = encoder.extract(
        image_path
    )

    print(f"Feature shape: {features.shape}")
    print(f"Feature dtype: {features.dtype}")