# main_kfold.py

import sys
import os
import logging
from typing import List

# 将项目根目录添加到Python搜索路径，确保可以找到自定义模块
# 假设 main_kfold.py 在 .../finalproject/ 目录下
# 我们需要将 .../ (即 finalproject 的父目录) 添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们创建的所有模块
import config
import utils
from pipelines import nnunet_pipeline
from pipelines import classification_pipeline

def main():
    """
    主函数，用于执行完整的5折交叉验证流程。
    该流程包含动态的nnU-Net训练和下游分类器训练。
    """
    # 1. 初始化设置
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    utils.set_seed(config.RANDOM_SEED)

    # 2. 在病人层面划分数据集为5折 (只执行一次)
    patient_folds: List[List[str]] = utils.create_patient_folds(
        patient_labels=config.ALL_PATIENTS_LABELS,
        num_folds=config.K_FOLDS,
        seed=config.RANDOM_SEED
    )
    all_pids_sorted = [pid for fold in patient_folds for pid in fold]

    all_fold_results = []

    # 3. 主循环，迭代5次，每次选择一折作为测试集
    for i in range(config.K_FOLDS):
        fold_num = i + 1
        
        # 为当前折设置独立的日志文件
        utils.setup_logging(fold_num)
        logging.info(f"#################### FOLD {fold_num}/{config.K_FOLDS} ####################")

        # --- 4. 数据划分 ---
        # 遵循您指定的 3-1-1 划分策略
        test_fold_index = i
        # 使用后一折作为验证集 (循环)
        val_fold_index_for_vit = (i + 1 ) % config.K_FOLDS 
        
        train_fold_indices_for_vit = [
            j for j in range(config.K_FOLDS) 
            if j != test_fold_index and j != val_fold_index_for_vit
        ]
        
        # nnU-Net的训练集是除了测试集之外的所有折
        # nnunet_train_indices = [j for j in range(config.K_FOLDS) if j != test_fold_index]

        # 获取对应的病人ID列表
        test_pids = patient_folds[test_fold_index]
        vit_val_pids = patient_folds[val_fold_index_for_vit]
        vit_train_pids = [pid for idx in train_fold_indices_for_vit for pid in patient_folds[idx]]
        nnunet_train_pids = vit_train_pids

        logging.info(f"Data Split for Fold {fold_num}:")
        logging.info(f"  - Test Set (ViT & nnU-Net): Fold {test_fold_index + 1} ({len(test_pids)} patients)")
        logging.info(f"  - Validation Set (ViT)    : Fold {val_fold_index_for_vit + 1} ({len(vit_val_pids)} patients)")
        logging.info(f"  - Training Set (ViT)      : Folds {[x+1 for x in train_fold_indices_for_vit]} ({len(vit_train_pids)} patients)")
        # logging.info(f"  - Training Set (nnU-Net)  : Folds {[x+1 for x in nnunet_train_indices]} ({len(nnunet_train_pids)} patients)")
        if fold_num == 1: continue
        # if fold_num == 2 :continue
        # --- 5. 动态训练 nnU-Net 分割模型 ---
        logging.info("\n--- Stage 1: Dynamic nnU-Net Training ---")

        # if fold_num == 1 and config.SKIP_NNUNET_TRAINING_FOR_FOLD_1:
        #     logging.info("!!! SKIPPING nnU-Net training for Fold 1 as per config. !!!")
        #     logging.info(f"Loading pre-trained model from: {config.PRETRAINED_FOLD_1_MODEL_PATH}")
        #     if not os.path.exists(config.PRETRAINED_FOLD_1_MODEL_PATH):
        #         logging.error("Pre-trained model path not found! Please check config.py.")
        #         # 抛出异常或直接退出，因为后续步骤无法进行
        #         raise FileNotFoundError(f"Path not found: {config.PRETRAINED_FOLD_1_MODEL_PATH}")
        #     nnunet_model_folder = config.PRETRAINED_FOLD_1_MODEL_PATH
        # elif fold_num == 2 and config.SKIP_NNUNET_TRAINING_FOR_FOLD_1:
        #     logging.info("!!! SKIPPING nnU-Net training for Fold 2 as per config. !!!")
        #     logging.info(f"Loading pre-trained model from: {config.PRETRAINED_FOLD_2_MODEL_PATH}")
        #     if not os.path.exists(config.PRETRAINED_FOLD_2_MODEL_PATH):
        #         logging.error("Pre-trained model path not found! Please check config.py.")
        #         # 抛出异常或直接退出，因为后续步骤无法进行
        #         raise FileNotFoundError(f"Path not found: {config.PRETRAINED_FOLD_2_MODEL_PATH}")
        #     nnunet_model_folder = config.PRETRAINED_FOLD_2_MODEL_PATH
        # elif fold_num == 3 and config.SKIP_NNUNET_TRAINING_FOR_FOLD_1:
        #     logging.info("!!! SKIPPING nnU-Net training for Fold 3 as per config. !!!")
        #     logging.info(f"Loading pre-trained model from: {config.PRETRAINED_FOLD_3_MODEL_PATH}")
        #     if not os.path.exists(config.PRETRAINED_FOLD_3_MODEL_PATH):
        #         logging.error("Pre-trained model path not found! Please check config.py.")
        #         # 抛出异常或直接退出，因为后续步骤无法进行
        #         raise FileNotFoundError(f"Path not found: {config.PRETRAINED_FOLD_3_MODEL_PATH}")
        #     nnunet_model_folder = config.PRETRAINED_FOLD_3_MODEL_PATH
        # else: 
        logging.info(f"Starting nnU-Net training for Fold {fold_num}...")
        nnunet_model_folder = nnunet_pipeline.run_nnunet_training_for_fold(
            fold_num=fold_num,
            train_pids=nnunet_train_pids
        )

        logging.info("\n--- Stage 2: Initializing nnU-Net Predictor ---")
        predictor = nnunet_pipeline.initialize_predictor_for_fold(nnunet_model_folder)
        logging.info("\n--- Stage 3: nnU-Net Global Inference ---")
        all_masks_dir = nnunet_pipeline.run_global_inference(
                predictor=predictor,
                fold_num=fold_num
            )
        # # # --- 【核心修改】调用新的全局推理函数 ---
        # if fold_num ==3:
        #     logging.info("\n--- Stage 3: nnU-Net Global Inference ---")
        #     all_masks_dir = nnunet_pipeline.run_global_inference(
        #         predictor=predictor,
        #         fold_num=fold_num
        #     )
        # fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
        # masks_output_dir = os.path.join(fold_dir, "nnunet_inference_output")
        # os.makedirs(masks_output_dir, exist_ok=True)
        # --- 7. 执行下游分类流程（特征提取 -> ViT训练 -> ViT测试）---
        logging.info("\n--- Stage 3: Downstream Classification Pipeline ---")
        # 这个函数内部会自己处理特征提取、训练、验证和测试
        test_metrics = classification_pipeline.run_classification_pipeline_for_fold(
            fold_num=fold_num,
            vit_train_pids=vit_train_pids,
            vit_val_pids=vit_val_pids,
            test_pids=test_pids,
            all_masks_dir=all_masks_dir
        )
        
        all_fold_results.append(test_metrics)
        logging.info(f"#################### FOLD {fold_num} COMPLETED ####################\n")

    # --- 8. 总结所有折的结果 ---
    utils.summarize_results(all_fold_results)

if __name__ == "__main__":
    # 确保在多进程环境中（nnU-Net可能会使用），主逻辑只执行一次
    main()