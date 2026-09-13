# WeatherDataset 高维时间序列分段与工况聚类

本项目使用 Python 和 Apache IoTDB 对 WeatherDataset 气象多变量时间序列进行清洗、存储、自动分段和工况识别。分段方法包括 PELT 和滑动窗口对称 KL 散度；聚类阶段比较 K-Means 与 GMM，并生成工况统计及四类可视化。

## 1. 提交目录

代码位于项目根目录，已有实验结果集中在 outputs/，按处理阶段分类，每个阶段分别保存 PELT 和 KL 的结果。

~~~text
Big-Data/
├── README.md
├── requirements.txt
├── main.py                         # 一键运行完整流程
├── data_loader.py                  # CSV / IoTDB 数据读取与清洗
├── import_weather.py               # 清洗数据并导入 IoTDB
├── segmentation.py                 # PELT 与滑动窗口 KL 分段
├── feature_extraction.py           # 分段特征提取与标准化
├── clustering.py                   # 聚类比较、工况标签及统计
├── visualization.py                # 四类结果图
├── weather_column_mapping.csv      # 原始列名与 IoTDB 字段名对照
├── data/
│   └── weather.csv                 # 原始气象数据
├── test_code4pelt/                 # PELT 参数实验代码
├── test_code4window/               # 滑动窗口 KL 参数实验代码
└── outputs/
    ├── segmentation/
    │   ├── pelt_segments.csv       # PELT 分段表
    │   └── kl_segments.csv         # KL 分段表
    ├── features/
    │   ├── pelt/                  # PELT 原始特征、标准化特征及标准化器
    │   └── kl/                    # KL 原始特征、标准化特征及标准化器
    ├── clustering/
    │   ├── pelt/                  # PELT 聚类指标、工况、坐标与模型
    │   └── kl/                    # KL 聚类指标、工况、坐标与模型
    ├── figures/
    │   ├── pelt/                  # PELT 四类最终图表
    │   └── kl/                    # KL 四类最终图表
    └── analysis/
        └── pelt/                  # 补充特征统计、相关性及冗余分析
            └── report_assets/     # 报告用 PNG / PDF 和配套数据
~~~

test_code4pelt/ 和 test_code4window/ 保存参数选择过程中的实验脚本，运行主流程不需要先运行它们。weather_column_mapping.csv 由导入程序生成，供核对字段含义。

## 2. 结果文件说明

### 分段表

[PELT 分段表](outputs/segmentation/pelt_segments.csv) 和 [KL 分段表](outputs/segmentation/kl_segments.csv) 保存每段的起止时间、下标和长度。

| 字段 | 含义 |
| --- | --- |
| method | 分段方法 |
| segment_number | 从 1 开始的分段编号 |
| start_index | 起始下标，包含该位置 |
| end_index_exclusive | 结束下标，不包含该位置 |
| end_index_inclusive | 段内最后一个采样点的下标 |
| start_time、end_time | 段内首尾采样时间 |
| length_samples | 采样点数量 |
| length_hours | 按采样间隔换算的分段长度 |

下标采用左闭右开的范围，例如 [0, 144) 包含 144 个采样点。

### 特征与聚类

下表中的 pelt 可以替换为 kl，查看另一种分段方法的对应结果。

| 文件 | 用途 |
| --- | --- |
| outputs/features/pelt/segment_features_raw.csv | 分段元数据及 567 维原始特征 |
| outputs/features/pelt/segment_features_scaled.csv | 标准化后的特征，元数据不参与标准化 |
| outputs/features/pelt/feature_scaler.joblib | 特征标准化器 |
| outputs/clustering/pelt/clustering_metrics.csv | 各算法、各 K 值的评价指标 |
| outputs/clustering/pelt/best_model_selection.txt | 最佳模型选择依据 |
| outputs/clustering/pelt/segment_labels.csv | 分段对应的聚类标签和工况 ID |
| outputs/clustering/pelt/operation_summary.csv | 各工况的分段数和时长统计 |
| outputs/clustering/pelt/pca_2d.csv | 分段的二维 PCA 坐标 |
| outputs/clustering/pelt/cluster_centers_2d.csv | 聚类中心的二维坐标 |
| outputs/clustering/pelt/clustering_preprocessor.joblib | 聚类预处理对象 |
| outputs/clustering/pelt/best_cluster_model.joblib | 所选聚类模型 |

### 图表与补充分析

| 图表 | PELT | KL |
| --- | --- | --- |
| 多通道时序及分段边界 | [查看](outputs/figures/pelt/01_multichannel_segmentation.png) | [查看](outputs/figures/kl/01_multichannel_segmentation.png) |
| 聚类散点与中心 | [查看](outputs/figures/pelt/02_cluster_scatter.png) | [查看](outputs/figures/kl/02_cluster_scatter.png) |
| 代表性分段 | [查看](outputs/figures/pelt/03_representative_segments.png) | [查看](outputs/figures/kl/03_representative_segments.png) |
| 工况时间轴 | [查看](outputs/figures/pelt/04_operation_timeline.png) | [查看](outputs/figures/kl/04_operation_timeline.png) |

outputs/analysis/pelt/ 保存已有的补充分析结果，可从 [完整统计分析](outputs/analysis/pelt/完整统计分析.md) 和 [报告相关性分析](outputs/analysis/pelt/report_assets/report_correlation_analysis.md) 开始阅读。该目录的材料不由 main.py 自动生成，补充分析生成脚本未包含在本提交目录中。

## 3. 数据与实验结果

原始数据共 52,696 行，包含 1 个重复时间戳。稳定排序并保留重复时间戳的最后一条记录后，得到 52,695 个时间点、21 个气象变量。采样间隔为 10 分钟，时间范围为 2020-01-01 00:10:00 至 2021-01-01 00:00:00。

预处理将小于等于 -9990 的哨兵值转为缺失值，再进行插值和有效性检查。分段前对各气象变量进行 Z-score 标准化；每段提取统计、时域形状、趋势及传感器相关性特征，共 567 维。

| 方法 | 最终分段参数 | 变点数 | 分段数 |
| --- | --- | ---: | ---: |
| PELT | c=0.6，min_size=144，jump=6 | 266 | 267 |
| 滑动窗口对称 KL | window=24，阈值系数 m=0.5，峰间距=144，平滑=6 | 260 | 261 |

PELT 使用多维 L2 代价，惩罚值为 c × 数据维度 × ln(时间点数量)。KL 方法比较左右窗口中对角高斯分布的均值和方差，以平滑后的对称 KL 散度峰值识别变点。

聚类默认使用 PCA 保留 95% 方差，比较 K=2～10 的 K-Means 和对角协方差 GMM。已有结果如下：

| 分段方法 | PCA 维度 | 所选算法 | K | Silhouette | CH | DB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| PELT | 64 | K-Means | 2 | 0.161670 | 57.520245 | 2.058445 |
| 滑动窗口 KL | 63 | K-Means | 2 | 0.156124 | 54.901159 | 2.074103 |

两个方法的工况 ID 分别由各自模型生成，同编号不代表同一种工况。二维 PCA 图用于展示，实际聚类使用上表所列的 PCA 维度。

## 4. 环境与一键运行

建议使用 Python 3.10 或更高版本。所有命令均在包含 main.py 的项目根目录执行。

~~~bash
python -m venv .venv
~~~

Windows PowerShell 激活环境：

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Linux / macOS 激活环境：

~~~bash
source .venv/bin/activate
~~~

安装依赖并运行：

~~~bash
python -m pip install -r requirements.txt
python main.py
~~~

默认读取 data/weather.csv，运行两种分段、特征提取、聚类和可视化。新运行的结果写入：

~~~text
outputs/pipeline/
├── cleaned_weather.csv
├── pelt/                       # 分段、特征、聚类、模型和 figures/
└── kl/                         # 分段、特征、聚类、模型和 figures/
~~~

**提交结果按阶段整理，主程序重新运行时按方法输出。** outputs/pipeline/ 是运行后生成的目录，不属于当前已整理的结果目录。每种方法的分段表在新运行目录中命名为 segments.csv。

只运行一种方法，或自定义本次运行的输出根目录：

~~~bash
python main.py --method pelt
python main.py --method kl
python main.py --output-dir outputs/rerun
~~~

更改 --output-dir 只改变输出根目录，内部仍为 pelt/、kl/，不会自动生成 features/、clustering/ 等分类目录。

## 5. 按提交目录单独运行

以下命令使用已有分段表或特征文件，将结果写到当前提交目录的对应位置；同名结果会更新。处理不同数据或新分段时，应同步调整各步骤的输入路径。

### 特征提取

先确认同一时间范围的清洗数据已导入 IoTDB，然后执行：

~~~bash
python feature_extraction.py --segments outputs/segmentation/pelt_segments.csv --output-dir outputs/features/pelt
python feature_extraction.py --segments outputs/segmentation/kl_segments.csv --output-dir outputs/features/kl
~~~

CSV 备用模式要求输入已经清洗的数据。例如，完整运行 main.py 后，可使用其生成的清洗文件；需确保该文件与所用分段表的时间轴一致：

~~~bash
python feature_extraction.py --source csv --data outputs/pipeline/cleaned_weather.csv --segments outputs/segmentation/pelt_segments.csv --output-dir outputs/features/pelt
~~~

### 聚类与可视化

~~~bash
python clustering.py --input outputs/features/pelt/segment_features_scaled.csv --output-dir outputs/clustering/pelt
python clustering.py --input outputs/features/kl/segment_features_scaled.csv --output-dir outputs/clustering/kl

python visualization.py --data data/weather.csv --segments outputs/segmentation/pelt_segments.csv --result-dir outputs/clustering/pelt --output-dir outputs/figures/pelt
python visualization.py --data data/weather.csv --segments outputs/segmentation/kl_segments.csv --result-dir outputs/clustering/kl --output-dir outputs/figures/kl
~~~

--result-dir 指向聚类结果目录，--output-dir 指定图表保存目录。单独调用特征和聚类程序时，请显式传入上述 --output-dir；原代码的默认输出目录不是提交结果的分类目录。

### 单独重新分段

~~~bash
python segmentation.py --source csv --input data/weather.csv
~~~

当前 segmentation.py 没有 --output-dir 参数，会在运行目录生成 pelt_segments.csv 和 kl_segments.csv。确认新结果后，可将它们归档至 outputs/segmentation/；若分段边界改变，需要重新生成对应的特征、聚类和图表。

## 6. IoTDB 与参数实验

IoTDB 默认连接为 127.0.0.1:6667，设备为 root.weather.station001。启动数据库并核对 import_weather.py 开头的连接配置后执行：

~~~bash
python import_weather.py
python main.py --source iotdb --host 127.0.0.1 --port 6667
~~~

导入程序先清洗数据，再删除该设备在 CSV 时间范围内的旧记录并重新写入；中断后需重新运行导入程序。字段对应关系写入项目根目录的 weather_column_mapping.csv。

主流程通过扩大查询范围后按时间轴裁剪来处理时区差异。独立运行 segmentation.py 查询 IoTDB 时，仍需核对首尾时间和行数，避免日期字面量的时区解释导致数据缺失。

参数实验代码的用途如下：

| 文件 | 用途 |
| --- | --- |
| test_code4pelt/pelt_experiment1.py | 比较 PELT 惩罚系数 |
| test_code4pelt/pelt_experiment2_min_size.py | 比较最短分段长度 |
| test_code4pelt/pelt_experiment3_jump.py | 比较候选位置搜索步长 |
| test_code4window/window_kl_experiment.py | 比较滑动窗口大小与 KL 阈值 |
| test_code4window/window_kl_final_candidates.py | 比较最终候选参数及与 PELT 的一致性 |

这些脚本保留原实验配置，路径中仍包含 E:\ApacheIoTDB。重新运行前需检查 CSV_PATH、OUTPUT_DIR、FIGURE_DIR，以及 KL 实验引用的 PELT 检查点路径。它们不接受与 main.py 相同的输出目录参数，也不需要在每次运行主流程前执行。
