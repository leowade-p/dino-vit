# modules/classification_pipeline.py

import os
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch.optim as optim  # <--- 在这里添加！
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import SimpleITK as sitk

# 从项目根目录导入模块
import config
import models
import utils

# --- 辅助函数：图像预处理 ---
def normalize_slice(slice_np, window_center, window_width):
    """ 将CT切片归一化到 [0, 255] 范围 """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    slice_clipped = np.clip(slice_np, min_val, max_val)
    slice_scaled = 255.0 * (slice_clipped - min_val) / (window_width + 1e-5)
    return Image.fromarray(slice_scaled.astype(np.uint8))

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

# --- 特征提取核心函数 ---
def extract_features_for_pids(
    pids: List[str],
    masks_dir: str,
    feature_backbone: nn.Module,
    fold_num: int,
    dataset_type: str
) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """
    基于nnU-Net生成的3D掩码，为给定的病人ID列表提取RoI特征。
    """
    cache_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}", "features")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_type}_features.pth")

    if os.path.exists(cache_path):
        logging.info(f"Loading cached {dataset_type} features from {cache_path}")
        return torch.load(cache_path)

    logging.info(f"Extracting RoI features for {len(pids)} patients ({dataset_type} set) using masks from {masks_dir}")
    
    feature_backbone.eval()
    pid_map = utils.load_pid_map()
    patient_bank = defaultdict(list)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize((392, 392), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    progress_bar = tqdm(pids, desc=f"Extracting Features ({dataset_type})")
    for pid_nii in progress_bar:
        pid = pid_nii.split('.nii')[0]
        if pid not in pid_map:
            logging.warning(f"PID {pid} not found in map, skipping feature extraction.")
            continue
        
        case_id = pid_map[pid]
        base_case_name = f"{config.BASE_NNUNET_TASK_NAME}_{case_id:03d}"
        
        image_path = os.path.join(config.NNUNET_RAW_IMAGES_DIR, f"{base_case_name}_0000.nii.gz")
        mask_path = os.path.join(masks_dir, f"{base_case_name}.nii.gz")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            logging.warning(f"Image or Mask not found for PID {pid}, skipping.")
            continue

        try:
            image_3d = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            mask_3d = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        except Exception as e:
            logging.error(f"Error loading files for PID {pid}: {e}")
            continue

        for slice_idx in range(image_3d.shape[0]):
            mask_slice = mask_3d[slice_idx, :, :]
            if not mask_slice.any():
                continue

            image_slice = image_3d[slice_idx, :, :]
            
            gray_pil = normalize_slice(image_slice, window_center=165, window_width=200)  # 举例参数
            # 如果 normalize_slice 返回的是 numpy 数组 (H, W) uint8，请用：
            # gray_pil = Image.fromarray(normalize_slice(image_slice, window_center=40, window_width=400))
        
            pil_image = gray_pil  # 不再 merge/伪彩，不转 RGB，由 transform 里的 Grayscale(3) 处理
            # ---------- 改动结束 ----------
        
            tensor = transform(pil_image).unsqueeze(0).to(config.DEVICE)
        
            mask_binary = (mask_slice > 0).astype(np.uint8) * 255

            with torch.no_grad():
                feat_map = feature_backbone(tensor)

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue

            spatial_scale_x = feat_map.shape[3] / tensor.shape[3]
            spatial_scale_y = feat_map.shape[2] / tensor.shape[2]
            
            boxes_scaled = [
                [
                    cv2.boundingRect(cnt)[0] * spatial_scale_x, 
                    cv2.boundingRect(cnt)[1] * spatial_scale_y,
                    (cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2]) * spatial_scale_x,
                    (cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3]) * spatial_scale_y
                ] for cnt in contours
            ]

            boxes_tensor = torch.tensor(boxes_scaled, device=config.DEVICE, dtype=torch.float32)
            box_indices = torch.zeros(len(boxes_tensor), 1, device=config.DEVICE)

            with torch.no_grad():
                roi_feats = roi_align(feat_map, torch.cat([box_indices, boxes_tensor], dim=1),
                                      output_size=(config.ROI_SIZE, config.ROI_SIZE), aligned=True)
            
            patient_bank[pid_nii].append(roi_feats.cpu())

    sequences, labels, patient_ids_out = [], [], []
    for pid_nii, feats_list in patient_bank.items():
        if pid_nii in config.ALL_PATIENTS_LABELS:
            all_feats = torch.cat(feats_list, dim=0)
            sequences.append(all_feats)
            labels.append(config.ALL_PATIENTS_LABELS[pid_nii])
            patient_ids_out.append(pid_nii)
    
    torch.save((sequences, labels, patient_ids_out), cache_path)
    logging.info(f"Saved extracted {dataset_type} features to {cache_path}")
    
    return sequences, labels, patient_ids_out


def run_classification_pipeline_for_fold(
    fold_num: int,
    vit_train_pids: List[str],
    vit_val_pids: List[str],
    test_pids: List[str],
    all_masks_dir: str
) -> Dict[str, Any]:
    """
    为指定的折执行完整的分类模型训练、验证和测试流程。
    """
    logging.info(f"========== Starting Classification Pipeline for Fold {fold_num} ==========")
    
    # 1. 初始化特征提取骨干网络
    feature_backbone = nn.Sequential(*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-4]).to(config.DEVICE)
    
    # 2. 提取所有数据集的特征
    train_seqs, train_y, train_pids_out = extract_features_for_pids(vit_train_pids, all_masks_dir, feature_backbone, fold_num, 'train')
    val_seqs, val_y, val_pids_out = extract_features_for_pids(vit_val_pids, all_masks_dir, feature_backbone, fold_num, 'val')
    test_seqs, test_y, test_pids_out = extract_features_for_pids(test_pids, all_masks_dir, feature_backbone, fold_num, 'test')

    # 3. 创建数据集和数据加载器
    train_ds = PatientRoIDataset(train_seqs, train_y, train_pids_out)
    val_ds = PatientRoIDataset(val_seqs, val_y, val_pids_out)
    test_ds = PatientRoIDataset(test_seqs, test_y, test_pids_out)

    train_loader = DataLoader(train_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config.CLS_BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 4. 初始化分类模型
    logging.info("Initializing Transformer classifier model...")
    pos_embedder = models.PatchPositionEmbedding(grid_size=config.ROI_SIZE, dim=config.BACKBONE_OUTPUT_DIM, learnable=config.CLS_POS_EMBED_LEARNABLE)
    transformer = models.ROIBasedTransformerClassifier(
        input_dim=config.BACKBONE_OUTPUT_DIM, hidden_dim=config.CLS_HIDDEN_DIM,
        n_heads=config.CLS_N_HEADS, n_layers=config.CLS_N_LAYERS
    )
    model = models.FullTumorClassifier(pos_embedder, transformer).to(config.DEVICE)
    
    # 5. 训练分类器 (这部分逻辑可以从之前的版本复制或重写)
    # ... (训练循环, 在验证集上寻找最佳模型, 保存checkpoint) ...
    # 假设训练完成后，我们得到了 best_model_path
    
    # 6. 加载最佳模型并在测试集上评估
    # ... (加载 best_model_path, 在 test_loader 上运行, 得到 test_metrics) ...
    
    # 为了简化，我将把训练和测试逻辑合并在此
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.CLS_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=False)

    best_val_auc = -1.0
    best_model_path = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}", "best_classifier_model.pth")
    
    for epoch in range(config.CLS_EPOCHS):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.CLS_EPOCHS} [Training]")
        for feat_maps, roi_masks, targets, _ in  progress_bar:
            feat_maps, roi_masks, targets = feat_maps.to(config.DEVICE), roi_masks.to(config.DEVICE), targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(feat_maps, roi_masks)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        val_metrics = evaluate_classifier(model, val_loader, criterion, config.DEVICE)
        scheduler.step(val_metrics['auc'])
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            deep_copied_state_dict = {}
            for key, value in model.state_dict().items():
                    deep_copied_state_dict[key] = value.clone().detach()
            torch.save({
                'model_state_dict': deep_copied_state_dict,
                'optimal_threshold': val_metrics['optimal_threshold']
            }, best_model_path)
            logging.info(f"Epoch {epoch+1}: New best val AUC: {best_val_auc:.4f}. Model saved.")

    logging.info("Classifier training finished. Loading best model for testing.")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_classifier(model, test_loader, criterion, config.DEVICE, is_test_set=True, fixed_threshold=checkpoint['optimal_threshold'])
    
    logging.info(f"--- Fold {fold_num} Test Set Report ---")
    for key, value in test_metrics.items():
        logging.info(f"{key.capitalize():<15}: {value:.4f}")
        
    return test_metrics


# 你需要将 evaluate_classifier 函数的完整定义添加到这个文件中
# 为保持简洁，我暂时用 pass 占位，你可以从之前的回复中复制过来
def evaluate_classifier(model, loader, criterion, device, is_test_set=False, fixed_threshold=None):
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
    metrics = {'loss': avg_loss}
    if len(np.unique(all_targets)) > 1:
        metrics['auc'] = roc_auc_score(all_targets, all_pred_probs)
        fpr, tpr, thresholds = roc_curve(all_targets, all_pred_probs)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)] if not is_test_set else fixed_threshold
    else:
        metrics['auc'] = 0.5
        optimal_threshold = fixed_threshold if is_test_set else 0.5
    metrics['optimal_threshold'] = optimal_threshold
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