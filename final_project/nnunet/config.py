# config.py
import torch
from nnunetv2.paths import nnUNet_results, nnUNet_raw
# --- 1. 核心路径配置 ---
# 请根据您的实际环境确认这些路径
# 项目根目录，也就是 finalproject 所在的目录
PROJECT_ROOT = "/home/huyiding/pengdie/tumor_classification" 
# 包含 finalproject, nnUNet_raw 等文件夹的顶级目录

# 交叉验证的所有输出（日志、模型、特征缓存等）将保存在这里
OUTPUT_DIR = f"{PROJECT_ROOT}/finalproject/kfold_results_without_posembed_829_2d"
SKIP_NNUNET_TRAINING_FOR_FOLD_1 = True
PRETRAINED_FOLD_1_MODEL_PATH = "/home/huyiding/pengdie/tumor_classification/nnUNet_results/Dataset901_Fold1TumorSeg/nnUNetTrainer__nnUNetPlans__3d_fullres" 
PRETRAINED_FOLD_2_MODEL_PATH = "/home/huyiding/pengdie/tumor_classification/nnUNet_results/Dataset902_Fold2TumorSeg/nnUNetTrainer__nnUNetPlans__3d_fullres" 
PRETRAINED_FOLD_3_MODEL_PATH = "/home/huyiding/pengdie/tumor_classification/nnUNet_results/Dataset903_Fold3TumorSeg/nnUNetTrainer__nnUNetPlans__3d_fullres" 

# --- 2. 交叉验证与全局配置 ---
K_FOLDS = 5
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NNUNET_DIM = '3d_fullres'
NNUNET_DIM = '2d'  
# (记得改OUTPUT_DIR )

# --- 3. nnU-Net 集成配置 ---
# nnU-Net 环境变量指向的目录
NNUNET_RAW_DATA_DIR = f"{PROJECT_ROOT}/nnUNet_raw"

# 用于训练 nnU-Net 的数据集（包含所有病人）
# 假设任务ID为510，任务名为TumorTotal
BASE_NNUNET_TASK_ID = 510
BASE_NNUNET_TASK_NAME = "TumorTotal"

# 包含所有病人原始3D影像 (.nii.gz) 的文件夹
NNUNET_RAW_IMAGES_DIR = f"{NNUNET_RAW_DATA_DIR}/Dataset{BASE_NNUNET_TASK_ID:03d}_{BASE_NNUNET_TASK_NAME}/imagesTr"
# 包含所有病人原始3D标签 (.nii.gz) 的文件夹 (用于创建临时训练集)
NNUNET_RAW_LABELS_DIR = f"{NNUNET_RAW_DATA_DIR}/Dataset{BASE_NNUNET_TASK_ID:03d}_{BASE_NNUNET_TASK_NAME}/labelsTr"

# PID到Case ID的映射文件路径
PID_TO_CASE_ID_MAP_PATH = f"{NNUNET_RAW_DATA_DIR}/Dataset{BASE_NNUNET_TASK_ID:03d}_{BASE_NNUNET_TASK_NAME}/pid_map_total.json"

# --- 4. 分类流程与模型超参数 ---
# 特征提取骨干网络 (ResNet50)
BACKBONE_OUTPUT_DIM = 512 # ResNet50 layer2 output channels

# RoI Align 参数
ROI_SIZE = 1

# 分类器Transformer参数
MAX_ROIS = 200
CLS_EPOCHS = 300
CLS_LR = 1e-4
CLS_BATCH_SIZE = 32
CLS_HIDDEN_DIM = 128
CLS_N_HEADS = 4
CLS_N_LAYERS = 2
# 使用您最初的局部位置编码方案
CLS_POS_EMBED_LEARNABLE = True 

# --- 5. 全量病人标签数据 ---
# 这个字典是进行K-Fold划分的基础
ALL_PATIENTS_LABELS =  {'2195561.nii': 1, '1918325.nii': 1, '1916513.nii': 1, '1852423.nii': 1, '1917881.nii': 1,
     '1857249.nii': 1, '2120967.nii': 1, '1865172.nii': 1, '1913371.nii': 1, '1926461.nii': 1, 
     '1925578.nii': 1, '2185498.nii': 1, '1862468.nii': 1, '1921315.nii': 1, '1854065.nii': 1, 
     '2176703.nii': 1, '2121132.nii': 1, '1918223.nii': 1, '1850916.nii': 1, '2189875.nii': 1, 
     '1924172.nii': 1, '1936357.nii': 1, '2192392.nii': 1, '2054533.nii': 1, '1933813.nii': 1, 
     '2191704.nii': 1, '1860366.nii': 1, '1914739.nii': 1, '1850947.nii': 1, '1926530.nii': 1, 
     '2149937.nii': 1, '2175352.nii': 1, '1859252.nii': 1, '1876395.nii': 1, '2189476.nii': 1, 
     '1910532.nii': 1, '2146979.nii': 1, '1926016.nii': 1, '2120982.nii': 1, '2173278.nii': 1, 
     '1927130.nii': 1, '1872957.nii': 1, '2173219.nii': 1, '2142355.nii': 1, '1917842.nii': 1, 
     '2021293.nii': 1, '1927747.nii': 1, '2186670.nii': 1, '1918272.nii': 1, '2174289.nii': 1, 
     '2191237.nii': 1, '2192966.nii': 1, '1870372.nii': 1, '1912864.nii': 1, '2101138.nii': 1, 
     '1853371.nii': 1, '2092015.nii': 1, '1879483.nii': 1, '1900454.nii': 1, '2194012.nii': 1, 
     '2173398.nii': 1, '1901411.nii': 1, '1916755.nii': 1, '1918233.nii': 1, '2188185.nii': 1,
     '1927047.nii': 1, '1917103.nii': 1, '1867143.nii': 1, '1916422.nii': 1, '1922289.nii': 1, 
     '1922706.nii': 1, '1859176.nii': 1, '2130175.nii': 1, '2196119.nii': 1, '1847203.nii': 1, 
     '2110127.nii': 1, '2132202.nii': 1, '1927553.nii': 1, '2189573.nii': 1, '2086306.nii': 1, 
     '1914670.nii': 1, '1912351.nii': 1, '2145811.nii': 1, '2124977.nii': 1, '1925054.nii': 1, 
     '1918234.nii': 1, '1860410.nii': 1, '2185872.nii': 1, '2154484.nii': 1, '2141440.nii': 1, 
     '2701357.nii': 0, '2336756.nii': 0, '2093629.nii': 0, '1915238.nii': 0, '2699066.nii': 0, 
     '2701358.nii': 0, '1924774.nii': 0, '1921022.nii': 0, '2027345.nii': 0, '1958973.nii': 0, 
     '1915209.nii': 0, '2698551.nii': 0, '1942564.nii': 0, '2561719.nii': 0, '1851144.nii': 0, 
     '1931525.nii': 0, '2597789.nii': 0, '2945516.nii': 0, '1916092.nii': 0, '1926843.nii': 0, 
     '2701177.nii': 0, '1953114.nii': 0, '2577307.nii': 0, '2061898.nii': 0, '2639834.nii': 0, 
     '2130364.nii': 0, '2648358.nii': 0, '2110802.nii': 0, '3138995.nii': 0, '1132045.nii': 0, 
     '3237293.nii': 0, '2042416.nii': 0, '1915203.nii': 0, '2125006.nii': 0, '2696318.nii': 0, 
     '879720.nii': 0, '1925620.nii': 0, '2336755.nii': 0, '2578670.nii': 0, '2013797.nii': 0, 
     '1926144.nii': 0, '879721.nii': 0, '1882547.nii': 0, '1935102.nii': 0, '1378023.nii': 0, 
     '1851152.nii': 0, '2090498.nii': 0, '1915205.nii': 0, '2125005.nii': 0, '1879428.nii': 0, 
     '2110801.nii': 0, '2597790.nii': 0, '1930335.nii': 0, '2796911.nii': 0, '1931526.nii': 0, 
     '2770800.nii': 0, '1915201.nii': 0, '2951334.nii': 0, '2582916.nii': 0, '1915207.nii': 0,
     '1922914.nii': 0, '2027346.nii': 0, '1925759.nii': 0, '2639836.nii': 0, '1925991.nii': 0, 
     '1401139.nii': 0, '1270467.nii': 0, '2577306.nii': 0, '1865567.nii': 0, '1927969.nii': 0, 
     '1915206.nii': 0, '2093628.nii': 0, '2110800.nii': 0, '1915202.nii': 0, '2027348.nii': 0, 
     '1953115.nii': 0, '2027347.nii': 0, '2770801.nii': 0, '1958972.nii': 0, '2639835.nii': 0, 
     '2122585.nii': 0, '1853634.nii': 0, '1945008.nii': 0, '1850703.nii': 0, '1915208.nii': 0, 
     '2141374.nii': 0, '2639837.nii': 0, '2100216.nii': 0, '1880821.nii': 0, '2125004.nii': 0, 
     '2785765.nii': 0, '1921021.nii': 0, '1864666.nii': 0, '2181815.nii': 1, '1920578.nii': 1, 
     '2188198.nii': 1, '1909134.nii': 1, '1913193.nii': 1, '1912052.nii': 1, '1944853.nii': 1, 
     '1953833.nii': 1, '1929233.nii': 1, '1906015.nii': 1, '2153776.nii': 1, '1947628.nii': 1, 
     '1927558.nii': 1, '1928694.nii': 1, '2165239.nii': 1, '1991716.nii': 0, '1865568.nii': 0, 
     '2760906.nii': 0, '2336754.nii': 0, '2728474.nii': 0, '2336753.nii': 0, '1132044.nii': 0, 
     '1879429.nii': 0, '2736060.nii': 0, '238876.nii': 0, '1915204.nii': 0, '1850473.nii': 0, 
     '2701356.nii': 0, '2593021.nii': 0, '1931527.nii': 0}