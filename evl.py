import os
import glob
import shutil
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm
from brisque import BRISQUE
import clip
from torch import nn
from pathlib import Path
import requests
from pytorch_fid import fid_score
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_ = torch.manual_seed(123)


name2text = {}


def load_binary_masks(map_dir):
    binary_masks = {}
    map_files = glob.glob(os.path.join(map_dir, '*.png'))
    for map_file in map_files:
        map_name = os.path.basename(map_file)
        name_without_ext = os.path.splitext(map_name)[0]
        img = Image.open(map_file).convert('L')
        binary_mask = np.array(img) / 255.0
        binary_masks[name_without_ext] = binary_mask
    return binary_masks

def load_saliency_map(image_path):
    image = Image.open(image_path).convert("L")
    return np.array(image) / 255.0


def calculate_occlusion(binary_mask, saliency_map):
    overlap_area = np.sum(saliency_map * binary_mask)
    element_area = np.sum(binary_mask) + np.sum(saliency_map)
    return overlap_area / element_area if element_area > 0 else 0


def cal_clip(img, captions, clip_score):
    img_ = (img * 255).type(torch.uint8)
    clip_score(img_, captions)



device = "cuda:0" if torch.cuda.is_available() else "cpu"





def filter_image_files(folder_path, image_extensions=['*.png', '*.jpg', '*.jpeg']):
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_files

def calculate_fid_with_pytorch_fid(real_img_folder, fake_img_folder, batch_size=32, device='cuda:0', dims=192):
    real_images = filter_image_files(real_img_folder)
    fake_images = filter_image_files(fake_img_folder)

    real_tmp_folder = '/tmp/real_images'
    fake_tmp_folder = '/tmp/fake_images'

    os.makedirs(real_tmp_folder, exist_ok=True)
    os.makedirs(fake_tmp_folder, exist_ok=True)

    for file in real_images:
        shutil.copy(file, os.path.join(real_tmp_folder, os.path.basename(file)))

    for file in fake_images:
        shutil.copy(file, os.path.join(fake_tmp_folder, os.path.basename(file)))

    fid_value = fid_score.calculate_fid_given_paths(
        [real_tmp_folder, fake_tmp_folder],
        batch_size=batch_size,
        device=device,
        dims=dims
    )

    shutil.rmtree(real_tmp_folder)
    shutil.rmtree(fake_tmp_folder)

    return fid_value


def evaluate_folder(src_dir):
    clip_score = CLIPScore(model_name_or_path="your/openai-clip-vit-base-patch16/path/").to(device)

    dst_dir = 'data/fiter_coco_with_json'
    map_dir = 'data/coco_maps'
    saliency_dir = f'{src_dir}-map'
    files = os.listdir(src_dir)
    batch_size = 50
    src_img, dst_img, captions = [], [], []
    occlusion_scores = []

    meta_path = 'data/filtered_2017_validation_data.json'
    with open(meta_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            name2text[item['file_name']] = item['captions'][0]

    binary_masks = load_binary_masks(map_dir)

    for name in tqdm(files, desc=f"Evaluating {src_dir}"):
        if name.endswith(".png"):
            src_path = os.path.join(src_dir, name)
            name_without_ext = os.path.splitext(name)[0].split('_')[0]
            dst_name = f"{name_without_ext}.jpg"
            dst_path = os.path.join(dst_dir, dst_name)
            saliency_path = os.path.join(saliency_dir, f"{os.path.splitext(name)[0]}_map.png")

            src_img.append(torchvision.transforms.functional.to_tensor(Image.open(src_path)).unsqueeze(0))
            dst_img.append(torchvision.transforms.functional.to_tensor(Image.open(dst_path)).unsqueeze(0))
            captions.append(name2text.get(dst_name))


            saliency_map = load_saliency_map(saliency_path)
            binary_mask = binary_masks.get(name_without_ext, np.zeros_like(saliency_map))
            occlusion_scores.append(calculate_occlusion(binary_mask, saliency_map))

            if len(src_img) == batch_size or name == files[-1]:
                src_tensor = torch.cat(src_img).to(device)
                try:
                    cal_clip(src_tensor, captions, clip_score)
                except:
                    pass
                src_img, captions = [], []

    fid_value = calculate_fid_with_pytorch_fid(dst_dir, src_dir)

    print('Metrics for:', src_dir)
    print('\tFID↓: %.2f' % fid_value)
    print('\tCLIP↑: %.3f' % clip_score.compute().item())
    print(f'\tOcclusion↓: {np.mean(occlusion_scores):.4f}')


if __name__ == '__main__':
    folders = [
        "data/evl_img/img"
    ]

    for folder_path in folders:
        print(f"\nEvaluating folder: {folder_path}")
        evaluate_folder(folder_path)











