import os

from matplotlib import pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


class Config:
    """
    项目配置文件
    集中管理所有路径和配置参数
    """

    # 基础目录配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0]
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # 数据目录结构
    DATA_RAW_DIR = os.path.join(DATA_DIR, "data_raw")
    DATA_MID_DIR = os.path.join(DATA_DIR, "data_mid")
    DATA_FINAL_DIR = os.path.join(DATA_DIR, "data_final")
    DATA_MODEL_DIR = os.path.join(DATA_DIR, "model")

    MODEL_MISS_ENCODER_DIR = os.path.join(DATA_MODEL_DIR, "model_miss_encoder")
    MODEL_MICE_DIR = os.path.join(DATA_MODEL_DIR, "model_mice")
    MODEL_ENCODER_DIR = os.path.join(DATA_MODEL_DIR, "model_encoder")

    # 文件路径配置
    DATA_RAW_FILE = os.path.join(DATA_RAW_DIR, "检验清洗数据_参考区间调整_20250427.xlsx")
    DATA_RAW_PHENOTYPE_FILE = os.path.join(DATA_RAW_DIR, "原始数据汇总_20260325.xlsx")
    DATA_ZH_EN_FILE = os.path.join(DATA_RAW_DIR, "变量中英文.xlsx")

    DATA_WIDE_FILE = os.path.join(DATA_MID_DIR, "检验清洗数据_l2w_20250325.xlsx")

    # 数据转换配置
    center = 'Hospital'
    label = 'Target'
    gender = 'Gender'
    age = 'Age'
    pid = 'PatientID'
    miss_dir = os.path.join(DATA_DIR, "missing_analysis")
    INDEX_COLS = [center, label, pid, gender, age]

    PHENOTYPE_COLS = ['医院', '分组', '患者ID-新', '患者性别', '患者年龄']
    TEST_COLS = ['医院', '分组', 'patientId', 'gender', 'age']

    test_index_map = dict(zip(TEST_COLS, INDEX_COLS))
    phenotype_index_map = dict(zip(PHENOTYPE_COLS, INDEX_COLS))

    FEATURE_COL = 'feature'
    VALUE_COL = 'value'
    non_order_cols = [center]

    for directory in [DATA_MID_DIR, DATA_FINAL_DIR, MODEL_MICE_DIR, MODEL_ENCODER_DIR,
                      MODEL_MISS_ENCODER_DIR, DATA_MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"目录已创建或已存在: {directory}")


# 全局配置实例
config = Config()
