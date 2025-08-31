# modules/nnunet_pipeline.py

import os
import json
import logging
import shutil
import subprocess
from typing import List

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import config
import utils

def run_nnunet_training_for_fold(fold_num: int, train_pids: List[str]) -> str:
    """
    为指定的K-Fold折动态创建一个临时nnU-Net数据集，并训练模型。

    Args:
        fold_num (int): 当前的折数 (用于命名和路径)。
        train_pids (List[str]): 用于训练的病人ID列表 (通常是4折的数据)。

    Returns:
        str: 训练好的、专属于此折的nnU-Net模型结果文件夹路径。
    """
    # 动态生成一个唯一的任务ID和名称，避免与现有任务冲突
    temp_task_id = 900 + fold_num
    temp_task_name = f"Fold{fold_num}TumorSeg"
    temp_dataset_name = f"Dataset{temp_task_id:03d}_{temp_task_name}"
    
    temp_task_dir = os.path.join(config.NNUNET_RAW_DATA_DIR, temp_dataset_name)
    logging.info(f"Creating temporary nnU-Net dataset for Fold {fold_num} at: {temp_task_dir}")

    # 1. 创建临时数据集结构
    images_tr_dir = os.path.join(temp_task_dir, "imagesTr")
    labels_tr_dir = os.path.join(temp_task_dir, "labelsTr")
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)

    # 加载PID映射
    pid_map = utils.load_pid_map()
    
    # 2. 通过软链接填充数据集，避免大量文件复制
    logging.info(f"Linking {len(train_pids)} patient files to temporary dataset...")
    for pid_nii in train_pids:
        pid = pid_nii.split('.nii')[0]
        if pid in pid_map:
            case_id = pid_map[pid]
            base_case_name = f"{config.BASE_NNUNET_TASK_NAME}_{case_id:03d}"
            
            # 链接图像文件
            src_img_path = os.path.join(config.NNUNET_RAW_IMAGES_DIR, f"{base_case_name}_0000.nii.gz")
            dst_img_path = os.path.join(images_tr_dir, f"{temp_task_name}_{case_id:03d}_0000.nii.gz")
            if os.path.exists(src_img_path) and not os.path.exists(dst_img_path):
                os.symlink(src_img_path, dst_img_path)

            # 链接标签文件
            src_lbl_path = os.path.join(config.NNUNET_RAW_LABELS_DIR, f"{base_case_name}.nii.gz")
            dst_lbl_path = os.path.join(labels_tr_dir, f"{temp_task_name}_{case_id:03d}.nii.gz")
            if os.path.exists(src_lbl_path) and not os.path.exists(dst_lbl_path):
                os.symlink(src_lbl_path, dst_lbl_path)
        else:
            logging.warning(f"PID {pid} not found in pid_map. Skipping for nnU-Net training.")

    # 3. 创建 dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(train_pids),
        "file_ending": ".nii.gz",
        "name": temp_task_name
    }
    with open(os.path.join(temp_task_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_json, f, indent=4)

    # 4. 调用nnU-Net的命令行工具 (使用subprocess)
    logging.info("Starting nnU-Net plan and preprocess...")
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", str(temp_task_id), "--verify_dataset_integrity"], check=True)
    
    logging.info("Starting nnU-Net training (3d_fullres)...")
    # subprocess.run(["nnUNetv2_train", str(temp_task_id), "3d_fullres", "0"], check=True)
    # subprocess.run(["nnUNetv2_train", str(temp_task_id), "3d_fullres", "1"], check=True)
    # subprocess.run(["nnUNetv2_train", str(temp_task_id), "3d_fullres", "2"], check=True)
    # subprocess.run(["nnUNetv2_train", str(temp_task_id), "3d_fullres", "3"], check=True)
    # subprocess.run(["nnUNetv2_train", str(temp_task_id), "3d_fullres", "4"], check=True)
    subprocess.run(["nnUNetv2_train", str(temp_task_id), config.NNUNET_DIM, "all"], check=True)
    
    # 5. 确定模型路径并返回
    model_folder_name = f"nnUNetTrainer__nnUNetPlans__{config.NNUNET_DIM}"

# 使用 os.path.join 安全地拼接完整路径
    trained_model_folder = os.path.join(os.environ.get("nnUNet_results"), temp_dataset_name, model_folder_name)
    logging.info(f"nnU-Net training complete for Fold {fold_num}. Model folder: {trained_model_folder}")
    
    return trained_model_folder


def initialize_predictor_for_fold(trained_model_folder: str) -> nnUNetPredictor:
    """
    初始化一个 nnUNetPredictor 对象并加载指定折的模型。
    这是一个耗时操作，应该只执行一次。
    """
    logging.info("Initializing nnUNetPredictor for the current fold...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        trained_model_folder,
        use_folds=('all',),
        checkpoint_name='checkpoint_best.pth',
    )
    logging.info("nnUNetPredictor initialized successfully.")
    return predictor

def run_global_inference(
    predictor: nnUNetPredictor,
    fold_num: int
) -> str:
    """
    【修改后】使用一个已经初始化好的 nnUNetPredictor，对整个任务的 imagesTr 文件夹进行推理。
    """
    fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
    masks_output_dir = os.path.join(fold_dir, "nnunet_inference_output")
    os.makedirs(masks_output_dir, exist_ok=True)
    
    # 我们的输入文件夹现在是固定的，即原始的 imagesTr 目录
    input_folder = config.NNUNET_RAW_IMAGES_DIR
    
    logging.info(f"Starting GLOBAL nnU-Net inference for Fold {fold_num}.")
    logging.info(f"  - Input Folder: {input_folder}")
    logging.info(f"  - Output Folder: {masks_output_dir}")
    
    # 直接调用 predict_from_files，传入完整的输入和输出文件夹
    predictor.predict_from_files(
        input_folder,
        masks_output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2, 
        num_processes_segmentation_export=2
    )

    logging.info(f"Global nnU-Net inference complete. Masks saved to: {masks_output_dir}")
    return masks_output_dir