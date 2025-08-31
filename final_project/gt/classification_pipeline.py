import os
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align
import torch.optim as optim  # <--- 在这里添加！
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import cv2

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 从项目根目录导入模块
import config
import models
import utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
# --- 数据集定义 ---
class PatientRoIDataset(Dataset):
    """
    为下游分类器准备数据的数据集。
    负责对每个病人的RoI特征序列进行填充或截断。
    """
    def __init__(self, sequences: List[torch.Tensor], labels: List[int], pids: List[str]):
        self.sequences = sequences
        self.labels = labels
        self.pids = pids
        self.max_rois = config.MAX_ROIS

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        feat_maps = self.sequences[idx]
        label = self.labels[idx]
        pid = self.pids[idx]
        num_rois, C, H, W = feat_maps.shape
        
        effective_len = min(num_rois, self.max_rois)
        mask = torch.zeros(self.max_rois, dtype=torch.bool)
        mask[:effective_len] = True

        if num_rois >= self.max_rois:
            padded_maps = feat_maps[:self.max_rois]
        else:
            pad_len = self.max_rois - num_rois
            padding = torch.zeros(pad_len, C, H, W, dtype=feat_maps.dtype)
            padded_maps = torch.cat([feat_maps, padding], dim=0)

        return padded_maps, mask, torch.tensor(label, dtype=torch.long), pid



# --------------------------- 主函数 --------------------------- #
def extract_features_for_pids(
    pids: List[str],
    feature_backbone: nn.Module,
    fold_num: int,
    dataset_type: str,
    class_aug_ratio: Dict[int, int],        # {label: K 倍数}
) -> Tuple[List[torch.Tensor], List[int], List[str]]:

    cache_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}", "features_v2")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_type}_features.pth")
    if os.path.exists(cache_path):
        logging.info(f"[✓] Load cached {dataset_type} features: {cache_path}")
        return torch.load(cache_path)

    logging.info(f"Extracting RoI features ({dataset_type}) with 3-D patient-level augmentation …")
    feature_backbone.eval()

    pid2imgs = utils.group_slice_paths_by_pid(config.IMAGE_DIRS, pids)   # {pid: [slice1.png, …]}
    pid2masks = utils.group_slice_paths_by_pid(config.MASK_DIRS, pids)   # 同样结构

    patient_bank, sequences, labels, patient_ids = defaultdict(list), [], [], []
    if dataset_type =='train':
        slice_tf = A.Compose([
    # ----- 纯 2-D pipeline，保持彩色 -----
    A.Rotate(limit=30, p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.2),
    A.RandomGamma(gamma_limit=(70,150), p=0.1),
    A.LongestMaxSize(max_size=392),
    A.PadIfNeeded(392, 392, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    # A.GaussNoise(var_limit=(0,0.01), p=0.1),
    ToTensorV2(),
])
    else:
        slice_tf = A.Compose([
    A.LongestMaxSize(max_size=392),
    A.PadIfNeeded(392, 392, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2(),
])
    for pid in tqdm(pids, desc=f"Patients ({dataset_type})"):
        img_paths = sorted(pid2imgs.get(pid, []), key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
        msk_paths = sorted(pid2masks.get(pid, []), key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
        if not img_paths:
            logging.warning(f"No slices found for {pid}, skipped.")
            continue

        label = config.ALL_PATIENTS_LABELS[pid]
        K = class_aug_ratio.get(label, 1) if dataset_type == 'train' else 1

        # —— 确保同一病人 K 次增广内部各 slice 参数一致 ——
        base_seed = (hash(pid) & 0x7FFFFFFF) 

        for k in range(K):
            aug_seed = base_seed + k              # 每个增广版本不同，但同病人内部一致
            roi_list = []
            for img_p, msk_p in zip(img_paths, msk_paths):
                random.seed(aug_seed)
                np.random.seed(aug_seed)
                img_bgr  = cv2.imread(img_p, cv2.IMREAD_COLOR)
                msk_gray = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
                if img_bgr is None or msk_gray is None:
                    logging.warning(f"skip {img_p}")
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                augmented = slice_tf(image=img_rgb, mask=msk_gray)
                tensor = augmented['image'].unsqueeze(0).to(config.DEVICE)
                mask_np = augmented['mask'].cpu().numpy().squeeze().astype(np.uint8)

                with torch.no_grad():
                    feat_map = feature_backbone(tensor)

                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                sx = feat_map.shape[3] / tensor.shape[3]
                sy = feat_map.shape[2] / tensor.shape[2]
                boxes = [[x*sx, y*sy, (x+w)*sx, (y+h)*sy]
                        for (x,y,w,h) in (cv2.boundingRect(cnt) for cnt in contours)]
                if not boxes:
                    continue

                boxes_t = torch.tensor(boxes, dtype=torch.float32, device=config.DEVICE)
                idxs = torch.zeros(len(boxes_t), 1, dtype=torch.float32, device=config.DEVICE)
                with torch.no_grad():
                    roi_feats = roi_align(feat_map, torch.cat([idxs, boxes_t], 1),
                                        output_size=(config.ROI_SIZE, config.ROI_SIZE),
                                        aligned=True)
                roi_list.append(roi_feats.cpu())

            if roi_list:   # 一个增广版本完成
                feats = torch.cat(roi_list, 0)  # (#RoI, C)
                sequences.append(feats)
                labels.append(label)
                patient_ids.append(f"{pid}_aug{k}" if dataset_type == 'train' else pid)

    # ---- 7. 缓存 ----
    torch.save((sequences, labels, patient_ids), cache_path)
    logging.info(f"[↓] Saved {dataset_type} features → {cache_path}")
    return sequences, labels, patient_ids

# --- 特征提取核心函数 ---
# def extract_features_for_pids(
#     pids: List[str],
#     feature_backbone: nn.Module,
#     fold_num: int,
#     dataset_type: str
# ) -> Tuple[List[torch.Tensor], List[int], List[str]]:
#     """
#     为给定的病人ID列表提取RoI特征。

#     Args:
#         pids (List[str]): 病人ID列表。
#         segmentation_model (nn.Module): 加载了当前折权重的分割模型。
#         feature_backbone (nn.Module): ResNet50特征提取器。
#         fold_num (int): 当前折数，用于缓存路径。
#         dataset_type (str): 'train', 'val', 或 'test'，用于日志和缓存。

#     Returns:
#         A tuple containing (sequences, labels, patient_ids).
#     """
#     # 检查缓存
#     cache_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}", "features")
#     os.makedirs(cache_dir, exist_ok=True)
#     cache_path = os.path.join(cache_dir, f"{dataset_type}_features.pth")

#     if os.path.exists(cache_path):
#         logging.info(f"Loading cached {dataset_type} features from {cache_path}")
#         return torch.load(cache_path)

#     logging.info(f"Extracting RoI features for {len(pids)} patients ({dataset_type} set)...")
    

#     feature_backbone.eval()

#     # 图像预处理
#     if dataset_type == 'train':
#         logging.info("Applying TRAINING augmentations and pre-processing.")
#         transform = A.Compose([
#             # 1. 尺寸调整
#             A.LongestMaxSize(max_size=392),
#             A.PadIfNeeded(min_height=392, min_width=392, border_mode=cv2.BORDER_CONSTANT, value=0),
#             A.Normalize(mean=[0.485, 0.456, 0.406], 
#                         std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])
#     else:  # For 'val' and 'test' sets
#         logging.info(f"Applying VALIDATION/TEST pre-processing for '{dataset_type}' set (no augmentation).")
#         transform = A.Compose([
#             # 1. 尺寸调整 (确定性操作)
#             A.LongestMaxSize(max_size=392),
#             A.PadIfNeeded(min_height=392, min_width=392, border_mode=cv2.BORDER_CONSTANT, value=0),
#             A.Normalize(mean=[0.485, 0.456, 0.406], 
#                         std=[0.229, 0.224, 0.225]),
#             ToTensorV2(),
#         ])
#     patient_bank = defaultdict(list)
#     image_files = utils.get_image_files_for_pids(config.IMAGE_DIRS, pids)

#     progress_bar = tqdm(image_files, desc=f"Extracting Features ({dataset_type})")
#     for img_path in progress_bar:
#         pid_prefix = os.path.basename(img_path).split('_')[0]
#         img_dir, img_filename = os.path.split(img_path)
#         base_dir, data_split_folder = os.path.split(img_dir)
#         mask_split_folder = data_split_folder.replace('train', 'panoptic_train').replace('val', 'panoptic_val').replace('test', 'panoptic_test')
#         mask_path = os.path.join(base_dir, mask_split_folder, img_filename)

#         # 2. 加载图像和 GT Mask
#         try:
#             image_np = cv2.imread(img_path)
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#             mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#             if image_np is None or mask_np is None:
#                 raise FileNotFoundError
#         except (FileNotFoundError, AttributeError):
#             logging.warning(f"Skipping because image or mask not found for: {img_path}")
#             continue

#         # 3. 将图像和掩码同时传入选择好的变换流程
#         #    Albumentations 会确保几何变换同步，而颜色变换只作用于图像
#         augmented = transform(image=image_np, mask=mask_np)
        
#         # 4. 从结果中获取最终的张量和掩码
#         # 'image' 已经是归一化并转换为Tensor的结果
#         tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
#         # 'mask' 是一个 [H, W] 的Tensor。我们需要将其转回numpy给cv2使用
#         mask_tensor = augmented['mask']
#         mask_for_contours = mask_tensor.cpu().numpy().astype(np.uint8)

#         # 5. 提取特征图
#         with torch.no_grad():
#             feat_map = feature_backbone(tensor)

#         contours, _ = cv2.findContours(mask_for_contours , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             continue

#         spatial_scale_x = feat_map.shape[3] / tensor.shape[3]
#         spatial_scale_y = feat_map.shape[2] / tensor.shape[2]
        
#         boxes_scaled = [
#             [
#                 c[0] * spatial_scale_x, 
#                 c[1] * spatial_scale_y, 
#                 (c[0] + c[2]) * spatial_scale_x, 
#                 (c[1] + c[3]) * spatial_scale_y
#             ]
#             for cnt in contours for c in [cv2.boundingRect(cnt)]
#         ]
        
#         if not boxes_scaled:
#             continue

#         boxes_tensor = torch.tensor(boxes_scaled, device=config.DEVICE, dtype=torch.float32)
#         box_indices = torch.zeros(len(boxes_tensor), 1, device=config.DEVICE)

#         # 3. RoI Align
#         with torch.no_grad():
#             roi_feats = roi_align(
#                 feat_map, 
#                 torch.cat([box_indices, boxes_tensor], dim=1),
#                 output_size=(config.ROI_SIZE, config.ROI_SIZE),
#                 aligned=True
#             )
        
#         patient_bank[f"{pid_prefix}.nii"].append(roi_feats.cpu())

#     # 聚合每个病人的特征
#     sequences, labels, patient_ids = [], [], []
#     for pid, feats_list in patient_bank.items():
#         if pid in config.ALL_PATIENTS_LABELS:
#             all_feats = torch.cat(feats_list, dim=0)
#             sequences.append(all_feats)
#             labels.append(config.ALL_PATIENTS_LABELS[pid])
#             patient_ids.append(pid)
#             logging.debug(f"Patient {pid}: {all_feats.shape[0]} total RoIs aggregated.")

#     # 保存到缓存
#     torch.save((sequences, labels, patient_ids), cache_path)
#     logging.info(f"Saved extracted {dataset_type} features to {cache_path}")
    
#     return sequences, labels, patient_ids


# --- 训练与评估函数 ---
def evaluate_classifier(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module,
    device: str,
    is_test_set: bool = False
) -> Dict[str, Any]:
    """
    在验证集或测试集上评估分类器。
    """
    model.eval()
    total_loss = 0.0
    all_targets, all_pred_probs = [], []

    with torch.no_grad():
        for feat_maps, roi_masks, targets, _ in loader:
            feat_maps, roi_masks, targets = feat_maps.to(device), roi_masks.to(device), targets.to(device)
            
            logits = model(feat_maps, roi_masks)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    
    # 计算指标
    metrics = {'loss': avg_loss}
    if len(np.unique(all_targets)) > 1:
        metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
        fpr, tpr, thresholds = roc_curve(all_targets, all_pred_probs)
        # 找到最佳阈值 (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else: # 只有一个类别
        metrics['auc'] = 0.5
        optimal_threshold = 0.5
        
    metrics['optimal_threshold'] = optimal_threshold
    
    # 使用最佳阈值计算其他指标
    binary_preds = (np.array(all_pred_probs) >= optimal_threshold).astype(int)
    metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
    metrics['sensitivity'] = recall_score(all_targets, binary_preds, zero_division=0)
    metrics['precision'] = precision_score(all_targets, binary_preds, zero_division=0)
    metrics['f1_score'] = f1_score(all_targets, binary_preds, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        metrics['specificity'] = 0

    return metrics


def run_classification_training(
    fold_num: int, 
    train_pids: List[str], 
    val_pids: List[str], 
) -> Tuple[str, float]:
    """
    为指定的折执行完整的分类模型训练流程。
    """
    logging.info(f"========== Starting Classification Training for Fold {fold_num} ==========")
    
    # 1. 加载当前折的分割模型和特征提取器
    logging.info("Loading models for feature extraction...")
    # base_encoder = torch.hub.load("../facebookresearch/dinov2", "dinov2_vitl14_reg", source='local')
    # segmentation_model = DINOV2EncoderLoRA(
    #     encoder=base_encoder, r=config.SEG_R_LORA, emb_dim=config.SEG_EMB_DIM, img_dim=config.SEG_IMG_DIM,
    #     n_classes=config.SEG_N_CLASSES, use_lora=True, use_fpn=True
    # ).to(config.DEVICE)
    # segmentation_model.load_parameters(segmentation_model_path)
    
    feature_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]).to(config.DEVICE)

    # 2. 提取训练集和验证集的特征
    train_seqs, train_y, train_pids_out = extract_features_for_pids(train_pids, feature_backbone, fold_num, 'train',class_aug_ratio= {0: 5,   1: 5})
    val_seqs, val_y, val_pids_out = extract_features_for_pids(val_pids,  feature_backbone, fold_num, 'val',class_aug_ratio={0: 1, 1: 1})


    train_ds = PatientRoIDataset(train_seqs, train_y, train_pids_out)
    val_ds = PatientRoIDataset(val_seqs, val_y, val_pids_out)
    
    train_loader = DataLoader(train_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 3. 初始化分类模型
    logging.info("Initializing Transformer classifier model...")
    pos_embedder = models.PatchPositionEmbedding(grid_size=config.ROI_SIZE, dim=config.BACKBONE_OUTPUT_DIM, learnable=config.CLS_POS_EMBED_LEARNABLE)
    transformer = models.ROIBasedTransformerClassifier(
        input_dim=config.BACKBONE_OUTPUT_DIM, hidden_dim=config.CLS_HIDDEN_DIM,
        n_heads=config.CLS_N_HEADS, n_layers=config.CLS_N_LAYERS
    )
    model = models.FullTumorClassifier(pos_embedder, transformer).to(config.DEVICE)

    # 4. 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.CLS_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)

    # 5. 训练循环
    best_val_auc = -1.0
    best_model_info = {}
    fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
    best_model_path = os.path.join(fold_dir, "best_classifier_model.pth")
    
    logging.info(f"Starting training for {config.CLS_EPOCHS} epochs...")
    for epoch in range(config.CLS_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CLS_EPOCHS} [Training]")
        for feat_maps, roi_masks, targets, _ in progress_bar:
            feat_maps, roi_masks, targets = feat_maps.to(config.DEVICE), roi_masks.to(config.DEVICE), targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(feat_maps, roi_masks)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        val_metrics = evaluate_classifier(model, val_loader, criterion, config.DEVICE)
        
        logging.info(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f}"
        )
        
        scheduler.step(val_metrics['auc'])

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            logging.info(f"🎉 New best validation AUC: {best_val_auc:.4f}. Saving model...")
            deep_copied_state_dict = {}
            for key, value in model.state_dict().items():
                    deep_copied_state_dict[key] = value.clone().detach()
            best_model_info = {
                'model_state_dict': deep_copied_state_dict,
                'optimal_threshold': val_metrics['optimal_threshold'],
                'val_auc': best_val_auc
            }
            torch.save(best_model_info, best_model_path)
            
    logging.info(f"========== Classification Training for Fold {fold_num} Finished ==========")
    return best_model_path


def run_classification_testing(
    fold_num: int, 
    test_pids: List[str],
    classifier_model_path: str
) -> Dict[str, Any]:
    """
    在测试集上评估最终的分类模型。
    """
    logging.info(f"========== Starting Classification Testing for Fold {fold_num} ==========")
    
    # 1. 加载模型
    logging.info("Loading models for testing...")
    # base_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", source='local')
    # segmentation_model = DINOV2EncoderLoRA(
    #     encoder=base_encoder, r=config.SEG_R_LORA, emb_dim=config.SEG_EMB_DIM, img_dim=config.SEG_IMG_DIM,
    #     n_classes=config.SEG_N_CLASSES, use_lora=True, use_fpn=True
    # ).to(config.DEVICE)
    # segmentation_model.load_parameters(segmentation_model_path)
    
    feature_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]).to(config.DEVICE)

    # 2. 提取测试集特征
    test_seqs, test_y, test_pids_out = extract_features_for_pids(test_pids,  feature_backbone, fold_num, 'test',class_aug_ratio={0: 1, 1: 1})
    test_ds = PatientRoIDataset(test_seqs, test_y, test_pids_out)
    test_loader = DataLoader(test_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. 初始化分类模型并加载最佳权重
    pos_embedder = models.PatchPositionEmbedding(grid_size=config.ROI_SIZE, dim=config.BACKBONE_OUTPUT_DIM, learnable=config.CLS_POS_EMBED_LEARNABLE)
    transformer = models.ROIBasedTransformerClassifier(
        input_dim=config.BACKBONE_OUTPUT_DIM, hidden_dim=config.CLS_HIDDEN_DIM,
        n_heads=config.CLS_N_HEADS, n_layers=config.CLS_N_LAYERS
    )
    model = models.FullTumorClassifier(pos_embedder, transformer).to(config.DEVICE)
    
    checkpoint = torch.load(classifier_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold_from_val = checkpoint['optimal_threshold']
    
    logging.info(f"Loaded best classifier model. Using threshold from validation: {optimal_threshold_from_val:.4f}")

    # 4. 在测试集上评估
    model.eval()
    all_targets, all_pred_probs = [], []
    with torch.no_grad():
        for feat_maps, roi_masks, targets, _ in tqdm(test_loader, desc="Testing"):
            feat_maps, roi_masks = feat_maps.to(config.DEVICE), roi_masks.to(config.DEVICE)
            logits = model(feat_maps, roi_masks)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(targets.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())

    # 使用从验证集得到的固定阈值进行评估
    binary_preds = (np.array(all_pred_probs) >= optimal_threshold_from_val).astype(int)
    
    test_metrics = {}
    if len(np.unique(all_targets)) > 1:
        test_metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
    else:
        test_metrics['auc'] = 0.5
        
    test_metrics['accuracy'] = accuracy_score(all_targets, binary_preds)
    test_metrics['sensitivity'] = recall_score(all_targets, binary_preds, zero_division=0)
    test_metrics['precision'] = precision_score(all_targets, binary_preds, zero_division=0)
    test_metrics['f1_score'] = f1_score(all_targets, binary_preds, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds).ravel()
        test_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    except ValueError:
        test_metrics['specificity'] = 0

    logging.info("--- Test Set Evaluation Report ---")
    for key, value in test_metrics.items():
        logging.info(f"{key.capitalize():<15}: {value:.4f}")
    
    logging.info(f"Confusion Matrix:\n{confusion_matrix(all_targets, binary_preds)}")
    logging.info(f"========== Classification Testing for Fold {fold_num} Finished ==========")
    
    return test_metrics