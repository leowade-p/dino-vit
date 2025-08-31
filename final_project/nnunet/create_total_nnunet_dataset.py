# create_total_nnunet_dataset.py
import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from collections import defaultdict
import json
from pathlib import Path

# ==============================================================================
# ---                           【请在这里配置】                           ---
# ==============================================================================

# 1. 原始数据路径
#    假设您的2D .png切片都存放在这些文件夹里
RAW_IMAGE_FOLDERS = [
    "../dino/det_datasets/4v1withouts1s2/train2017",
    "../dino/det_datasets/4v1withouts1s2/val2017",
]
RAW_LABEL_FOLDERS = [
    "../dino/det_datasets/4v1withouts1s2/panoptic_train2017",
    "../dino/det_datasets/4v1withouts1s2/panoptic_val2017", 
]

# 2. 输出的 nnU-Net 数据集配置
#    【重要】确保您的 $nnUNet_raw 环境变量已设置！
if 'nnUNet_raw' not in os.environ:
    raise EnvironmentError("错误: 请先设置 'nnUNet_raw' 环境变量!")

NNUNET_RAW_PATH = os.environ.get('nnUNet_raw')
TARGET_DATASET_ID = 510
TARGET_TASK_NAME = "TumorTotal"

# 3. 图像处理参数
TARGET_SIZE_XY = (392, 392) # 所有切片将被resize到这个XY尺寸

# ==============================================================================
# ---                           脚本主逻辑开始                           ---
# ==============================================================================

def main():
    print("--- Starting Full Dataset Conversion for nnU-Net ---")

    # --- 1. 创建输出目录 ---
    output_folder = Path(NNUNET_RAW_PATH) / f"Dataset{TARGET_DATASET_ID:03d}_{TARGET_TASK_NAME}"
    imagesTr_folder = output_folder / "imagesTr"
    labelsTr_folder = output_folder / "labelsTr"
    imagesTr_folder.mkdir(parents=True, exist_ok=True)
    labelsTr_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_folder}")

    # --- 2. 收集并按病人ID分组所有文件 ---
    patient_files = defaultdict(list)
    print("\nScanning for all image files...")
    for folder in RAW_IMAGE_FOLDERS:
        for f in sorted(os.listdir(folder)):
            if f.endswith(".png"):
                # 过滤掉您不想要的文件
                if 's1' in f or 's2' in f:
                    continue
                pid = f.split('_')[0]
                # 保存文件的完整路径，避免混淆
                patient_files[pid].append(os.path.join(folder, f))
    
    num_patients = len(patient_files)
    print(f"Found {num_patients} unique patients across all folders.")

    # --- 3. 创建并保存全局ID映射 ---
    all_pids_sorted = sorted(patient_files.keys())
    pid_to_nnunet_id_map = {pid: i for i, pid in enumerate(all_pids_sorted)}
    
    map_filename = output_folder / "pid_map_total.json"
    with open(map_filename, 'w') as f:
        json.dump(pid_to_nnunet_id_map, f, indent=4)
    print(f"✅ Patient ID to nnU-Net ID map saved to: {map_filename}")

    # --- 4. 遍历每个病人，创建3D .nii.gz 文件 ---
    print("\nStarting conversion to 3D .nii.gz format...")
    for pid, file_path_list in patient_files.items():
        nnunet_id = pid_to_nnunet_id_map[pid]
        print(f"  Processing patient {pid} (nnU-Net ID: {nnunet_id:03d})...")
        
        # --- a. 处理图像 ---
        image_slices = []
        for f_path in file_path_list:
            img = Image.open(f_path).convert("L")
            img_resized = img.resize(TARGET_SIZE_XY, Image.Resampling.BILINEAR)
            image_slices.append(np.array(img_resized))
        image_3d = np.stack(image_slices, axis=0)
        
        sitk_img = sitk.GetImageFromArray(image_3d)
        # 如果您有体素间距信息，在这里设置: sitk_img.SetSpacing([sp_x, sp_y, sp_z])
        
        output_image_filename = imagesTr_folder / f"{TARGET_TASK_NAME}_{nnunet_id:03d}_0000.nii.gz"
        sitk.WriteImage(sitk_img, str(output_image_filename))

        # --- b. 处理标签 ---
        label_slices = []
        for f_path in file_path_list:
            # 找到对应的标签文件路径
            filename = os.path.basename(f_path)
            label_found = False
            for lbl_folder in RAW_LABEL_FOLDERS:
                lbl_path = os.path.join(lbl_folder, filename)
                if os.path.exists(lbl_path):
                    lbl = Image.open(lbl_path).convert("L")
                    lbl_resized = lbl.resize(TARGET_SIZE_XY, Image.Resampling.NEAREST)
                    label_slices.append(np.array(lbl_resized))
                    label_found = True
                    break
            if not label_found:
                print(f"    - WARNING: Label file not found for {filename}. Creating an empty label.")
                label_slices.append(np.zeros(TARGET_SIZE_XY, dtype=np.uint8))
        
        label_slices_binary = [(s > 0).astype(np.uint8) for s in label_slices]
        label_3d = np.stack(label_slices_binary, axis=0)
        
        sitk_lbl = sitk.GetImageFromArray(label_3d)
        sitk_lbl.CopyInformation(sitk_img)

        output_label_filename = labelsTr_folder / f"{TARGET_TASK_NAME}_{nnunet_id:03d}.nii.gz"
        sitk.WriteImage(sitk_lbl, str(output_label_filename))

    # --- 5. 创建 dataset.json 文件 ---
    json_content = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": num_patients,
        "file_ending": ".nii.gz",
        "name": TARGET_TASK_NAME
    }
    with open(output_folder / "dataset.json", 'w') as f:
        json.dump(json_content, f, indent=4)
        
    print("\n✅ Dataset conversion finished successfully!")


if __name__ == "__main__":
    main()