from src.config import config
from src.data.data_processor import DataProcessor


def main():
    """主函数 - 执行长格式到宽格式的数据转换"""
    print("开始数据转换流程...")

    # 创建数据处理器
    processor = DataProcessor()

    try:
        # 1. 加载数据
        print("\n=== 步骤1: 加载数据 ===")
        processor.load_data()
        processor.load_phen(file_path=config.DATA_RAW_PHENOTYPE_FILE)

        # 2. 预处理数据
        print("\n=== 步骤2: 预处理数据 ===")
        processor.preprocess_data()

        # 3. 转换数据格式
        print("\n=== 步骤3: 长格式转宽格式 ===")
        processor.transform_long_to_wide()

        # 4. 保存结果
        print("\n=== 步骤4: 保存结果 ===")
        processor.save_result()

        # 5. 显示数据信息
        print("\n=== 数据转换完成 ===")
        info = processor.get_data_info()
        print(f"原始数据形状: {info.get('original_shape', 'N/A')}")
        print(f"转换后数据形状: {info.get('wide_shape', 'N/A')}")
        print(f"转换后列数: {len(info.get('wide_columns', []))}")

    except Exception as e:
        print(f"数据转换过程中发生错误: {e}")
        raise


def long2wide(data_raw_file=None):
    """
    兼容旧版本的函数接口

    Args:
        data_raw_file: 数据文件路径，如果为None则使用默认配置
    """
    processor = DataProcessor()
    config.create_directories()

    processor.load_data(data_raw_file)
    processor.preprocess_data()
    processor.transform_long_to_wide()
    processor.save_result()

    return processor.df_wide


if __name__ == "__main__":
    main()
