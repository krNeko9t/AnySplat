from pathlib import Path
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image

# Load the model from Hugging Face
model = AnySplat.from_pretrained("lhjiang/anysplat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

from pathlib import Path

img_root = Path("examples/vrnerf/riverview")

image_names = [p for p in img_root.iterdir()]

# Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"] 
images = [process_image(image_name) for image_name in image_names]
images = torch.stack(images, dim=0).unsqueeze(0).to(device) # [1, K, 3, 448, 448]
b, v, _, h, w = images.shape

# Run Inference
gaussians, pred_context_pose = model.inference((images+1)*0.5)

from save_gs import save_tensors_to_ply

save_tensors_to_ply('tmp.ply',gaussians.means[0],gaussians.scales[0],gaussians.rotations[0],gaussians.opacities[0],gaussians.harmonics[0])

 pred_all_extrinsic = pred_context_pose['extrinsic']
pred_all_intrinsic = pred_context_pose['intrinsic']
save_interpolated_video(pred_all_extrinsic, pred_all_intrinsic, b, h, w, gaussians, './res', model.decoder)