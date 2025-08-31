# main_kfold.py

import os
import logging
from typing import List

# 导入我们创建的所有模块
import config
import utils
from modules import segmentation_trainer
from modules import classification_pipeline

def main():
    """
    主函数，用于执行完整的5折交叉验证流程。
    """
    # 1. 初始化设置
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    utils.set_seed(config.RANDOM_SEED)

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # 2. 在病人层面划分数据集为5折
    # 这是整个交叉验证的基础，只执行一次
    patient_folds: List[List[str]] = utils.create_patient_folds(
        patient_labels=config.ALL_PATIENTS_LABELS,
        num_folds=config.K_FOLDS,
        seed=config.RANDOM_SEED
    )

    all_fold_results = []

    # 3. 主循环，迭代5次，每次选择一折作为测试集
    for i in range(config.K_FOLDS):
        fold_num = i + 1
        
        # 为当前折设置独立的日志文件
        utils.setup_logging(fold_num)
        logging.info(f"#################### FOLD {fold_num}/{config.K_FOLDS} ####################")

        # 4. 根据您的要求分配数据集角色
        # 测试集 = 当前折 (i)
        # 验证集 = 下一折 ((i + 1) % 5)，实现循环
        # 训练集 = 剩下的3折
        test_fold_index = i
        val_fold_index = (i + 1) % config.K_FOLDS
        
        train_fold_indices = [j for j in range(config.K_FOLDS) if j != test_fold_index and j != val_fold_index]

        # 获取对应的病人ID列表
        test_pids = patient_folds[test_fold_index]
        val_pids = patient_folds[val_fold_index]
        train_pids = [pid for idx in train_fold_indices for pid in patient_folds[idx]]

        logging.info(f"Data Split for Fold {fold_num}:")
        logging.info(f"  - Test Set Fold Index : {test_fold_index + 1} ({len(test_pids)} patients)")
        logging.info(f"  - Validation Set Fold Index : {val_fold_index + 1} ({len(val_pids)} patients)")
        logging.info(f"  - Training Set Fold Indices: {[idx + 1 for idx in train_fold_indices]} ({len(train_pids)} patients)")

        # --- 执行流水线 ---

        # 步骤 A: 训练分割模型
        # 这个函数会处理从数据加载到模型保存的所有事情
        segmentation_model_path = segmentation_trainer.run_segmentation_training(
            fold_num=fold_num,
            train_pids=train_pids,
            val_pids=val_pids,
            test_pids=test_pids
        )
        # fold_dir = os.path.join(config.OUTPUT_DIR, f"fold_{fold_num}")
        # segmentation_model_path = os.path.join(fold_dir, "best_segmentation_model.pt")
        # 步骤 B: 训练分类器
        # 这个函数会加载上一步的模型，提取特征，并训练下游分类器
        classifier_model_path = classification_pipeline.run_classification_training(
            fold_num=fold_num,
            train_pids=train_pids,
            val_pids=val_pids,
            segmentation_model_path=segmentation_model_path
        )
        
        # 步骤 C: 在测试集上评估
        # 这个函数会加载所有必要的模型，在独立的测试集上进行最终评估
        test_metrics = classification_pipeline.run_classification_testing(
            fold_num=fold_num,
            test_pids=test_pids,
            segmentation_model_path=segmentation_model_path,
            classifier_model_path=classifier_model_path
        )
        
        all_fold_results.append(test_metrics)
        logging.info(f"#################### FOLD {fold_num} COMPLETED ####################\n")

    # 5. 总结所有折的结果
    logging.info("All k-fold iterations have completed. Summarizing results.")
    utils.summarize_results(all_fold_results)

if __name__ == "__main__":
    main()