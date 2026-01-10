"""
配置文件，包含模型和训练的所有超参数
"""

class Config:
    # 数据相关参数
    BATCH_SIZE = 512
    NUM_WORKERS = 4
    
    # 模型相关参数
    DROPOUT = 0.3
    FEATURE_DIM = 2048  # CNN14特征维度
    USE_PROJECTION_HEAD = True  # 是否使用投影头
    
    # 训练相关参数
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5  # Warmup轮数
    MIN_LR = 1e-6  # 最小学习率
    
    # 损失函数相关参数
    EXP_MODE = 'adaptation'  # 'classification' 或 'adaptation'
    ADAPT_METHOD = 'homm'  # 域适应方法: 'ddc', 'coral', 'log_coral', 'cmd', 'homm', 'geo_adapt'
    USER_LAMDA = 0.1  # 域适应损失的权重
    HIGHEST_MOMENT = 2  # 中心矩匹配的最高阶数
    DIST_METRIC_TYPE = 'hilbert'  # 几何适应的距离度量类型
    DET_THR = 1e-6  # 行列式阈值，用于geo_adapt方法
    
    # 数据集相关参数
    DATASET_JSON = r'E:\代码\data2\dann_dataset_b.json'  # 数据集JSON文件路径
    DATA_ROOT = r'E:\代码\data2\raw'  # 数据根目录
    
    # 其他参数
    LOG_DIR = 'E:\代码\CNN14VAE\output'
    MODEL_SAVE_PATH = 'E:\代码\CNN14VAE\output'
    
    # 预训练权重路径
    PRETRAINED_CHECKPOINT_PATH = r'E:\代码\DANN_audio\Cnn14_16k_mAP=0.438.pth'  # 预训练模型路径

    # ===== DIR offline augmentation =====
    DIR_ENABLE = True
    DIR_PROB = 0.7                 # 训练时抽到 source 样本，用 DIR 版本的概率
    DIR_NUM_VARIANTS = 8           # 每个样本离线生成多少个增强版本（建议 2~8，推荐8）
    DIR_AUG_ROOT = r'E:\代码\data2\raw_dir'  # 存放增强 npy 的根目录
    DIR_APPLY_DOMAIN = 0           # 只对 domain==0（source/device A）启用
    
# 为方便访问，创建一个全局配置实例
config = Config()