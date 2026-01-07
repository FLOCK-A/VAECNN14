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
    USE_PROJECTION_HEAD = False  # 是否使用投影头

    # 训练相关参数
    NUM_EPOCHS = 120
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 5e-5
    WARMUP_EPOCHS = 10  # Warmup轮数
    MIN_LR = 1e-6  # 最小学习率

    # 损失函数相关参数
    EXP_MODE = 'adaptation'  # 'classification' 或 'adaptation'
    ADAPT_METHOD = 'cmd'  # 域适应方法: 'ddc', 'coral', 'log_coral', 'cmd', 'homm', 'geo_adapt'
    USER_LAMDA = 0.10  # 域适应损失的权重
    HIGHEST_MOMENT = 2  # 中心矩匹配的最高阶数
    DIST_METRIC_TYPE = 'hilbert'  # 几何适应的距离度量类型
    DET_THR = 1e-6  # 行列式阈值，用于geo_adapt方法
    
    # 数据集相关参数
    DATASET_JSON = r'E:\代码\data2\dann_dataset_c.json'  # 数据集JSON文件路径
    DATA_ROOT = r'E:\代码\data2\raw'  # 数据根目录
    
    # 其他参数
    LOG_DIR = r'E:\代码\CNN14VAE\output'
    MODEL_SAVE_PATH = r'E:\代码\CNN14VAE\output'
    
    # 预训练权重路径
    PRETRAINED_CHECKPOINT_PATH = r'E:\代码\DANN_audio\Cnn14_16k_mAP=0.438.pth'  # 预训练模型路径

    # ===== Augmentations (spectrogram-level) =====
    NUM_CLASSES = 10
    AUG_ENABLE = True

    # 设备类别数（DCASE Task1 通常为 6：a,b,c,s1,s2,s3；按你的domain编码为准）
    NUM_DEVICES = 6

    # 设备预测损失权重（先给小一点，后面再调）
    LAMBDA_DEV = 0.1

    # 正交/去相关损失权重（建议先小）
    LAMBDA_ORTH = 1e-3

    # 语义-设备独立性损失权重（非对抗，建议先小）
    LAMBDA_INDEP = 1e-2

    # 投影维度（先用和你CMD的adapt_dim一致，后续可调）
    ZS_DIM = 128
    ZD_DIM = 128

    AUG_P_TIME_ROLL = 0.3

    AUG_P_SPECAUG = 0.6
    AUG_FREQ_MASK_RATIO = 0.08
    AUG_TIME_MASK_RATIO = 0.04
    AUG_NUM_FREQ_MASKS = 2
    AUG_NUM_TIME_MASKS = 1

    AUG_P_FILTER = 0.2
    AUG_FILTER_N_BAND = (4, 8)
    AUG_FILTER_DB = (-0.5, 0.5)  # 若你的 mel 已做 mean/std 标准化，可先改成 (-1.0, 1.0)
    AUG_FILTER_MODE = "add"  # log-mel 推荐 add

    AUG_P_FMS = 0.5
    AUG_FMS_ALPHA = 0.5
    AUG_FMS_MIX = "target_donor"

    AUG_P_MIXUP = 0.0
    AUG_MIXUP_ALPHA = 0.0


# 为方便访问，创建一个全局配置实例
config = Config()
