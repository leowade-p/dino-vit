# dino-vit
the code for dino-vit

题：判断肿瘤良恶性

数据集： 乳腺CT 良性病例：108例；恶性病例：105例；数据均衡

核心流程： 数据准备: 加载每个病人的多张医学影像切片及其对应的label。 单切片特征提取 (DINOv2-lora): 对每一张切片，使用微调的DINO模型提取框。 ROIAlign：通过框在featuremap做映射得到框对应的feature 多切片特征融合: 为每个病人选择所有的切片特征图，整合成一个张量。 序列构建与填充: 将每个病人的ROI特征向量序列化。由于ROI数量可能不同，将序列填充到固定的最大长度。 序列分类 (Transformer): 使用Transformer模型处理填充后的特征序列，进行最终的分类任务。

详细步骤：

利用DINO对每一张切片预测框 模型选择与加载: 选择一个预训练的DINO模型。 加载预训练权重。

将病人所有切片的的预测框映射到feature map featuremap为backbone中分辨率最大的那张feature map（resnet的最后一层） 得到框后，将其映射回featmap，得到每一个框对应的featmap 真实的标注框坐标通常是（0，1），需要映射回原始图像分辨率，再将这些坐标按比例缩放到特征图 (H, W) 的尺度上。缩放因子是 原始图像尺寸 / 特征图尺寸（例如，224 / 14 = 16）。

3.分别进行 RoIAlign，以每个病人为单位，再拼接起来 对每一张featmap进行ROIALIGN，size为3*3，得到长为9的tensorlist，每一个tensor的feat_dim为1024 （因为resnet的backbone的feat_dim为1024 之后将病人的所有切片对应的tensor list拼接起来

类似VIT ，每一个tensor视为token，添加一个cls token与所有的query做互相关运算 不要忘了加pos_embed 最后把clstoken送到MLP来获得最终的预测
5.实验结果 在以病人为单位，3：1：1划分数据集，五折交叉验证下，auc达到0.958
