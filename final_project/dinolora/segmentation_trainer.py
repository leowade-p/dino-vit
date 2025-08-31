# segmentation_trainer.py

import os
import logging
from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from dino_finetune import DINOV2EncoderLoRA
# 从项目根目录导入模块
import config
import utils
import models

class SegmentationDataset(Dataset):
    """
    为分割任务创建的数据集。
    它会根据图像文件名自动查找对应的掩码（mask）文件。
    """
    def __init__(self, image_paths: List[str]): # img_dim 不再需要，因为变换流程会处理尺寸
        self.image_paths = image_paths
        
        # --- 只需要定义一个变换流程 ---
        self.transform = A.ReplayCompose([
            A.Rotate(limit=30, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomScale(scale_limit=0.2, p=0.2),
            A.LongestMaxSize(max_size=392),
            A.PadIfNeeded(392, 392, border_mode=cv2.BORDER_CONSTANT, value=0),
            # 像素增强在 Normalize 之前
            A.RandomGamma(gamma_limit=(70, 150), p=0.1),
            # A.GaussNoise(var_limit=(0.0, 10.0), p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    # In your SegmentationDataset class:
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # --- More robust path construction ---
        try:
            img_dir, img_filename = os.path.split(img_path)
            base_dir, data_split_folder = os.path.split(img_dir)
            
            # Build the corresponding mask folder name
            mask_split_folder = data_split_folder.replace('train', 'panoptic_train').replace('val', 'panoptic_val').replace('test', 'panoptic_test')
            
            mask_path = os.path.join(base_dir, mask_split_folder, img_filename)

            # --- Load data ---
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image file: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask file: {mask_path}")
            mask[mask > 0] = 1
        except (FileNotFoundError, AttributeError) as e:
            logging.error(f"Error loading data for img: {img_path}. Details: {e}")
            # Return dummy data to avoid crashing the loader
            # For a 3-channel image and a 1-channel mask
            return torch.zeros(3, 392, 392), torch.zeros(1, 392, 392).long()

        # --- Apply augmentations ---
        augmented = self.transform(image=image, mask=mask)
        
        image = augmented['image']
        mask = augmented['mask']
    
        
        return image, mask.long()

def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    # 将 target 转换为 one-hot 编码
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def compute_iou(pred, target, n_classes):
    iou_scores = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            iou_scores.append(float('nan'))
        else:
            iou_scores.append(intersection / union)
    return np.nanmean(iou_scores)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    total_val_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            total_val_loss += loss.item()
            
            iou = compute_iou(logits.cpu(), masks.cpu(), model.n_classes)
            total_val_iou += iou

    avg_loss = total_val_loss / len(val_loader)
    avg_iou = total_val_iou / len(val_loader)
    
    return avg_loss, avg_iou
# 非amp版
def run_segmentation_training(fold_num: int, train_pids: List[str], val_pids: List[str],test_pids: List[str],) -> str:
    """
    为指定的折执行完整的分割模型训练流程。

    Args:
        fold_num (int): 当前的折数 (用于日志和保存路径)。
        train_pids (List[str]): 用于训练的病人ID列表。
        val_pids (List[str]): 用于验证的病人ID列表。

    Returns:
        str: 训练好的最佳分割模型的权重文件路径。
    """
    logging.info(f"========== Starting Segmentation Training for Fold {fold_num} ==========")
    
    # 1. 设置路径
    fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
    best_model_path = os.path.join(fold_dir, "best_segmentation_model.pt")

    # 2. 准备数据
    logging.info("Preparing data loaders...")
    train_files = utils.get_image_files_for_pids(config.IMAGE_DIRS, train_pids)
    val_files = utils.get_image_files_for_pids(config.IMAGE_DIRS, val_pids)
    test_files = utils.get_image_files_for_pids(config.IMAGE_DIRS, test_pids)
    
    logging.info(f"Found {len(train_files)} training images for {len(train_pids)} patients.")
    logging.info(f"Found {len(val_files)} validation images for {len(val_pids)} patients.")

    train_dataset = SegmentationDataset(train_files, )
    val_dataset = SegmentationDataset(val_files, )
    test_dataset = SegmentationDataset(test_files, )

    
    train_loader = DataLoader(train_dataset, batch_size=config.SEG_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.SEG_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.SEG_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


    # 3. 初始化模型
    logging.info("Initializing DINOv2+LoRA segmentation model...")
    base_encoder = torch.hub.load("../facebookresearch/dinov2", "dinov2_vitl14_reg", source='local')
    base_encoder.load_state_dict(torch.load(config.DINOV2_PRETRAINED_WEIGHTS))
    
    model =DINOV2EncoderLoRA(
        encoder=base_encoder,
        r=config.SEG_R_LORA,
        emb_dim=config.SEG_EMB_DIM,
        img_dim=config.SEG_IMG_DIM,
        n_classes=config.SEG_N_CLASSES,
        use_lora=config.USE_LORA,
        use_fpn=True
    ).to(config.DEVICE)
    
    # 4. 设置损失函数、优化器和调度器
    ce_loss = nn.CrossEntropyLoss(ignore_index=255).to(config.DEVICE)
    def combined_loss(logits, targets):
        return 0.6 * ce_loss(logits, targets) + 0.4 * dice_loss(logits, targets)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.SEG_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SEG_EPOCHS)

    # 5. 训练循环
    best_val_iou = -1.0
    logging.info(f"Starting training for {config.SEG_EPOCHS} epochs...")
    for epoch in range(config.SEG_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.SEG_EPOCHS} [Training]")
        for images, masks in progress_bar:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        # 每5个epoch或最后一个epoch进行验证
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config.SEG_EPOCHS:
            logging.info("Running validation...")
            val_loss, val_iou = validate_epoch(model, val_loader, combined_loss, config.DEVICE)
            logging.info(f"Validation | Loss: {val_loss:.4f} | Mean IoU: {val_iou:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                logging.info(f"🎉 New best validation IoU: {best_val_iou:.4f}. Saving model to {best_model_path}")
                model.save_parameters(best_model_path)
    _, test_iou = validate_epoch(model, test_loader, combined_loss, config.DEVICE)
    logging.info(f" {fold_num} test iou {test_iou}==========")
   
    logging.info(f"========== Segmentation Training for Fold {fold_num} Finished ==========")
    return best_model_path

# amp版
# def run_segmentation_training(fold_num: int, train_pids: List[str], val_pids: List[str]) -> str:
#     """
#     为指定的折执行完整的分割模型训练流程。

#     Args:
#         fold_num (int): 当前的折数 (用于日志和保存路径)。
#         train_pids (List[str]): 用于训练的病人ID列表。
#         val_pids (List[str]): 用于验证的病人ID列表。

#     Returns:
#         str: 训练好的最佳分割模型的权重文件路径。
#     """
#     logging.info(f"========== Starting Segmentation Training for Fold {fold_num} ==========")
    
#     # 1. 设置路径
#     fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
#     best_model_path = os.path.join(fold_dir, "best_segmentation_model.pt")

#     # 2. 准备数据
#     logging.info("Preparing data loaders...")
#     train_files = utils.get_image_files_for_pids(config.IMAGE_DIRS, train_pids)
#     val_files = utils.get_image_files_for_pids(config.IMAGE_DIRS, val_pids)
    
#     logging.info(f"Found {len(train_files)} training images for {len(train_pids)} patients.")
#     logging.info(f"Found {len(val_files)} validation images for {len(val_pids)} patients.")

#     train_dataset = SegmentationDataset(train_files, config.SEG_IMG_DIM)
#     val_dataset = SegmentationDataset(val_files, config.SEG_IMG_DIM)
    
#     train_loader = DataLoader(train_dataset, batch_size=config.SEG_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=config.SEG_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

#     # 3. 初始化模型
#     logging.info("Initializing DINOv2+LoRA segmentation model...")
#     base_encoder = torch.hub.load("../facebookresearch/dinov2", "dinov2_vitl14_reg", source='local')
#     base_encoder.load_state_dict(torch.load(config.DINOV2_PRETRAINED_WEIGHTS))
    
#     model =DINOV2EncoderLoRA(
#         encoder=base_encoder,
#         r=config.SEG_R_LORA,
#         emb_dim=config.SEG_EMB_DIM,
#         img_dim=config.SEG_IMG_DIM,
#         n_classes=config.SEG_N_CLASSES,
#         use_lora=config.USE_LORA,
#         use_fpn=True
#     ).to(config.DEVICE)
    
#     # 4. 设置损失函数、优化器和调度器
#     ce_loss = nn.CrossEntropyLoss(ignore_index=255).to(config.DEVICE)
#     def combined_loss(logits, targets):
#         return 0.6 * ce_loss(logits, targets) + 0.4 * dice_loss(logits, targets)

#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.SEG_LR)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SEG_EPOCHS)
#     from torch.cuda.amp import autocast, GradScaler
#     scaler = GradScaler()
#     # 5. 训练循环
#     best_val_iou = -1.0
#     logging.info(f"Starting training for {config.SEG_EPOCHS} epochs...")
#     for epoch in range(config.SEG_EPOCHS):
#         model.train()
#         total_train_loss = 0.0
        
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.SEG_EPOCHS} [Training]")
#         for images, masks in progress_bar:
#             images = images.to(config.DEVICE)
#             masks = masks.to(config.DEVICE)

#             optimizer.zero_grad()

#             with autocast():
#                 logits = model(images)
#                 loss = combined_loss(logits, masks)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_train_loss += loss.item()
#             progress_bar.set_postfix(loss=loss.item())

#         avg_train_loss = total_train_loss / len(train_loader)
#         logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
#         scheduler.step()
#         torch.cuda.empty_cache()
#         # 每5个epoch或最后一个epoch进行验证
#         if (epoch + 1) % 5 == 0 or (epoch + 1) == config.SEG_EPOCHS:
#             logging.info("Running validation...")
#             val_loss, val_iou = validate_epoch(model, val_loader, combined_loss, config.DEVICE)
#             logging.info(f"Validation | Loss: {val_loss:.4f} | Mean IoU: {val_iou:.4f}")

#             if val_iou > best_val_iou:
#                 best_val_iou = val_iou
#                 logging.info(f"🎉 New best validation IoU: {best_val_iou:.4f}. Saving model to {best_model_path}")
#                 model.save_parameters(best_model_path)

#     logging.info(f"========== Segmentation Training for Fold {fold_num} Finished ==========")
#     return best_model_path


