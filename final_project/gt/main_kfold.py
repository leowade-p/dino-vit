# main_kfold.py

import sys
import os
import logging
from typing import List

# 将项目根目录添加到Python搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入我们创建的所有模块
import config
import utils
# 【修改】现在我们只需要导入 classification_pipeline
from pipelines import classification_pipeline

def main():
    """
    主函数，用于执行完整的5折交叉验证流程。
    此版本使用GT掩码，并在数据加载时进行在线3D数据增强。
    """
    # 1. 初始化设置
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    utils.set_seed(config.RANDOM_SEED)

    # 2. 在病人层面划分数据集为5折 (不变)
    patient_folds: List[List[str]] = utils.create_patient_folds(
        patient_labels=config.ALL_PATIENTS_LABELS,
        num_folds=config.K_FOLDS,
        seed=config.RANDOM_SEED
    )

    all_fold_results = []

    # 3. 主循环，迭代5次 (不变)
    for i in range(config.K_FOLDS):
        fold_num = i + 1
        utils.setup_logging(fold_num)
        logging.info(f"#################### FOLD {fold_num}/{config.K_FOLDS} ####################")

        # --- 4. 数据划分 (不变) ---
        # 遵循您指定的 3-1-1 划分策略
        test_fold_index = i
        # 使用前一折作为验证集 (循环) - 保持与之前讨论的一致
        val_fold_index_for_vit = (i + 1 ) % config.K_FOLDS 
        
        train_fold_indices_for_vit = [
            j for j in range(config.K_FOLDS) 
            if j != test_fold_index and j != val_fold_index_for_vit
        ]

        # 获取对应的病人ID列表
        test_pids = patient_folds[test_fold_index]
        vit_val_pids = patient_folds[val_fold_index_for_vit]
        vit_train_pids = [pid for idx in train_fold_indices_for_vit for pid in patient_folds[idx]]

        logging.info(f"Data Split for Fold {fold_num}:")
        logging.info(f"  - Test Set        : Fold {test_fold_index + 1} ({len(test_pids)} patients)")
        logging.info(f"  - Validation Set  : Fold {val_fold_index_for_vit + 1} ({len(vit_val_pids)} patients)")
        logging.info(f"  - Training Set    : Folds {[x+1 for x in train_fold_indices_for_vit]} ({len(vit_train_pids)} patients)")

        # --- 5. 【核心修改】直接调用分类流程 ---
        
        # 直接指定GT掩码的文件夹路径
        gt_masks_dir = config.MASK_DIRS
        logging.info(f"\nUsing Ground Truth masks from: {gt_masks_dir}")
        
        classifier_model_path = classification_pipeline.run_classification_training(
            fold_num=fold_num,
            train_pids=vit_train_pids,
            val_pids=vit_val_pids,
        )
        
        # 步骤 C: 在测试集上评估
        # 这个函数会加载所有必要的模型，在独立的测试集上进行最终评估
        test_metrics = classification_pipeline.run_classification_testing(
            fold_num=fold_num,
            test_pids=test_pids,
            classifier_model_path=classifier_model_path
        )
        
        all_fold_results.append(test_metrics)
        logging.info(f"#################### FOLD {fold_num} COMPLETED ####################\n")
        
        all_fold_results.append(test_metrics)
        logging.info(f"#################### FOLD {fold_num} COMPLETED ####################\n")

    # --- 6. 总结所有折的结果 (不变) ---
    utils.summarize_results(all_fold_results)

if __name__ == "__main__":
    main()