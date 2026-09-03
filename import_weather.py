import re
import sys
from pathlib import Path

import pandas as pd

from iotdb.Session import Session
from iotdb.utils.Tablet import Tablet
from iotdb.utils.IoTDBConstants import TSDataType


# ==================== 配置 ====================

CSV_PATH = Path(r"E:\ApacheIoTDB\data\weather.csv")

HOST = "127.0.0.1"
PORT = 6667
USERNAME = "root"
PASSWORD = "root"

DATABASE = "root.weather"
DEVICE = "root.weather.station001"

BATCH_SIZE = 1000

MAPPING_OUTPUT = Path(
    r"E:\ApacheIoTDB\outputs\weather_column_mapping.csv"
)


# ==================== 字段名称清理 ====================

def sanitize_measurement_name(name):
    """
    将CSV中的复杂列名转换成适合IoTDB路径的名称。

    例如：
    T (degC)       -> T_degC
    max. wv (m/s)  -> max_wv_m_s
    """
    cleaned = str(name).strip()

    # 将特殊字符替换为下划线
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", cleaned)

    # 合并连续下划线
    cleaned = re.sub(r"_+", "_", cleaned)

    # 删除首尾下划线
    cleaned = cleaned.strip("_")

    if not cleaned:
        cleaned = "sensor"

    # 如果以数字开头，增加前缀
    if cleaned[0].isdigit():
        cleaned = "sensor_" + cleaned

    return cleaned


def create_unique_names(original_columns):
    """
    清理字段名，并确保不会出现重复字段。
    """
    cleaned_columns = []
    used_names = set()

    for original_name in original_columns:
        base_name = sanitize_measurement_name(original_name)
        final_name = base_name
        suffix = 2

        while final_name in used_names:
            final_name = f"{base_name}_{suffix}"
            suffix += 1

        used_names.add(final_name)
        cleaned_columns.append(final_name)

    return cleaned_columns


# ==================== 读取数据 ====================

def load_weather_data():
    print("=" * 70)
    print("1. 读取 Weather 数据集")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{CSV_PATH}\n"
            "请先将 weather.csv 放到 E:\\ApacheIoTDB\\data 中。"
        )

    df = pd.read_csv(CSV_PATH)

    print(f"文件路径：{CSV_PATH}")
    print(f"原始数据形状：{df.shape}")
    print(f"原始字段数量：{len(df.columns)}")
    print(f"原始字段：{df.columns.tolist()}")

    if len(df.columns) < 2:
        raise ValueError("CSV 至少需要一个时间字段和一个传感器字段。")

    # Weather标准数据集的第一列为date
    if "date" in df.columns:
        time_column = "date"
    else:
        time_column = df.columns[0]
        print(
            f"没有找到date字段，将第一列 {time_column!r} "
            "作为时间字段。"
        )

    original_sensor_columns = [
        column
        for column in df.columns
        if column != time_column
    ]

    cleaned_sensor_columns = create_unique_names(
        original_sensor_columns
    )

    rename_mapping = dict(
        zip(
            original_sensor_columns,
            cleaned_sensor_columns
        )
    )

    print()
    print("字段名称映射：")

    for original, cleaned in rename_mapping.items():
        print(f"  {original}  ->  {cleaned}")

    # 保存字段映射，便于报告中解释
    MAPPING_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mapping_df = pd.DataFrame({
        "original_column": original_sensor_columns,
        "iotdb_measurement": cleaned_sensor_columns,
        "iotdb_path": [
            f"{DEVICE}.{column}"
            for column in cleaned_sensor_columns
        ],
    })

    mapping_df.to_csv(
        MAPPING_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n字段映射已保存：{MAPPING_OUTPUT}")

    df = df.rename(columns=rename_mapping)

    # 转换时间
    df[time_column] = pd.to_datetime(
        df[time_column],
        errors="coerce"
    )

    invalid_time_count = int(
        df[time_column].isna().sum()
    )

    if invalid_time_count > 0:
        raise ValueError(
            f"发现 {invalid_time_count} 个无效时间。"
        )

    # 转换所有气象变量为数值
    for column in cleaned_sensor_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    missing_before = int(
        df[cleaned_sensor_columns]
        .isna()
        .sum()
        .sum()
    )

    print(f"数值转换后的缺失值数量：{missing_before}")

    if missing_before > 0:
        print("正在使用线性插值处理缺失值……")

        df[cleaned_sensor_columns] = (
            df[cleaned_sensor_columns]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )

    missing_after = int(
        df[cleaned_sensor_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_after > 0:
        raise ValueError(
            f"插值后仍有 {missing_after} 个缺失值，程序停止。"
        )

    # 按时间排序
    df = df.sort_values(
        time_column
    ).reset_index(drop=True)

    duplicate_count = int(
        df[time_column].duplicated().sum()
    )

    print(f"重复时间戳数量：{duplicate_count}")

    if duplicate_count > 0:
        print("将保留重复时间戳中的最后一条记录。")

        df = df.drop_duplicates(
            subset=[time_column],
            keep="last"
        ).reset_index(drop=True)

    print(f"处理后数据形状：{df.shape}")
    print(f"时间范围：{df[time_column].min()} 至 {df[time_column].max()}")
    print(f"气象变量数量：{len(cleaned_sensor_columns)}")

    print("\n前5行数据：")
    print(df.head())

    return (
        df,
        time_column,
        cleaned_sensor_columns
    )


# ==================== 连接IoTDB ====================

def open_session():
    print()
    print("=" * 70)
    print("2. 连接 IoTDB")
    print("=" * 70)

    session = Session(
        HOST,
        PORT,
        USERNAME,
        PASSWORD
    )

    session.open(False)

    print(f"IoTDB连接成功：{HOST}:{PORT}")

    return session


# ==================== 创建数据库 ====================

def create_database(session):
    print()
    print("=" * 70)
    print("3. 创建或检查数据库")
    print("=" * 70)

    try:
        session.execute_non_query_statement(
            f"CREATE DATABASE {DATABASE}"
        )

        print(f"数据库创建成功：{DATABASE}")

    except Exception as error:
        message = str(error)

        if (
            "already exists" in message.lower()
            or "already been set" in message.lower()
            or "path already exist" in message.lower()
        ):
            print(f"数据库已经存在：{DATABASE}")
        else:
            print(f"数据库创建提示：{message}")
            print("程序将继续尝试写入。")


# ==================== 生成时间戳 ====================

def create_timestamps(df, time_column):
    """
    将pandas纳秒时间转换成IoTDB毫秒时间戳。
    """
    return (
        df[time_column]
        .astype("int64")
        .floordiv(1_000_000)
        .astype("int64")
        .tolist()
    )


# ==================== 批量导入 ====================

def insert_weather_data(
    session,
    df,
    time_column,
    sensor_columns
):
    print()
    print("=" * 70)
    print("4. 批量写入 Weather 数据")
    print("=" * 70)

    timestamps = create_timestamps(
        df,
        time_column
    )

    data_types = [
        TSDataType.DOUBLE
        for _ in sensor_columns
    ]

    total_rows = len(df)

    total_batches = (
        total_rows + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, total_rows, BATCH_SIZE),
        start=1
    ):
        end = min(
            start + BATCH_SIZE,
            total_rows
        )

        batch_df = df.iloc[start:end]

        values = (
            batch_df[sensor_columns]
            .astype(float)
            .values
            .tolist()
        )

        batch_timestamps = timestamps[start:end]

        tablet = Tablet(
            DEVICE,
            sensor_columns,
            data_types,
            values,
            batch_timestamps
        )

        session.insert_tablet(tablet)

        print(
            f"批次 {batch_number:02d}/{total_batches}："
            f"已写入 {end}/{total_rows} 个时间点"
        )

    try:
        session.execute_non_query_statement("FLUSH")
        print("FLUSH执行成功")
    except Exception as error:
        print(f"FLUSH提示：{error}")

    total_points = (
        total_rows * len(sensor_columns)
    )

    print()
    print("Weather数据写入完成")
    print(f"时间点数量：{total_rows}")
    print(f"气象变量数量：{len(sensor_columns)}")
    print(f"数据点总数：{total_points}")


# ==================== 主程序 ====================

def main():
    session = None

    try:
        (
            df,
            time_column,
            sensor_columns
        ) = load_weather_data()

        session = open_session()

        create_database(session)

        insert_weather_data(
            session,
            df,
            time_column,
            sensor_columns
        )

        print()
        print("=" * 70)
        print("Weather数据集已成功导入IoTDB")
        print("=" * 70)
        print(f"数据库：{DATABASE}")
        print(f"设备：{DEVICE}")
        print(f"时间点：{len(df)}")
        print(f"气象变量：{len(sensor_columns)}")

    except FileNotFoundError as error:
        print(f"\n文件错误：{error}")
        sys.exit(1)

    except ConnectionRefusedError:
        print("\n无法连接IoTDB。")
        print("请确认IoTDB已经启动，并且6667端口可用。")
        sys.exit(1)

    except Exception as error:
        print()
        print("=" * 70)
        print("导入失败")
        print("=" * 70)
        print(f"错误类型：{type(error).__name__}")
        print(f"错误信息：{error}")
        raise

    finally:
        if session is not None:
            try:
                session.close()
                print("IoTDB连接已关闭")
            except Exception:
                pass


if __name__ == "__main__":
    main()