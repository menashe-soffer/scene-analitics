import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchreid
import torch
from torchvision import transforms

from paths import *

class reid_features:

    def __init__(self):

        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=4101,
            loss='softmax',
        pretrained=False
        )

        weight_path = os.path.join(MODEL_FOLDER, 'osnet_x1_0_msmt17.pth')
        torchreid.utils.load_pretrained_weights(model, weight_path)

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # --- 2. Create Dummy Data and Preprocessing ---
        # ReID models typically expect 256x128 input size.
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create a dummy image tensor (replace with your actual data loader)
        dummy_image = torch.randn(1, 3, 256, 128) # 1 image, 3 channels, 256x128
        dummy_input = dummy_image.to(device)

        # --- 3. Extract the Embedding Vector (Feature) ---

        # The crucial step is the forward pass.
        # The 'torchreid' OSNet implementation (and many others)
        # is designed to return the feature vector (embedding) when in model.eval() mode.
        with torch.no_grad(): # Disable gradient calculations for faster inference
            # Pass the input through the model
            embedding_vector = model(dummy_input)

        self.model = model
        self.transform = transform
        self.device = device


    def __call__(self, frame, bbox, mask):

        x1, y1, x2, y2 = bbox.flatten().astype(int)

        # 1. Crop the image and mask using the bounding box
        cropped_image_np = frame[y1:y2, x1:x2]
        cropped_mask_np = mask[y1:y2, x1:x2]

        masked_crop_np = cropped_image_np.copy()
        masked_crop_np = masked_crop_np * np.tile(np.expand_dims(cropped_mask_np, axis=-1), 3)
        # #
        # # this is intended to confirm the masking is done properly,
        # plt.imshow(cropped_mask_np)
        # plt.show()
        # plt.imshow(masked_crop_np)
        # plt.show()
        # #
        masked_crop_pil = Image.fromarray(masked_crop_np)

        input_tensor = self.transform(masked_crop_pil).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # The output of a Re-ID model is typically the feature vector
            features = self.model(input_tensor)

        # Post-process the feature (e.g., normalize and convert to numpy)
        features = features.squeeze(0).cpu().numpy()

        # Normalization (if not already done by the model)
        features = features / np.linalg.norm(features)

        return features





