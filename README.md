# Big-Data

根据题目中的六项任务、评分权重和提交要求，建议采用“按流水线模块分工”，这样代码边界清晰、三个人可以并行开发。依据《实践题目》中的完整要求制定：:codex-file-citation{path="F:/OneDrive/课程/大数据综合实践/实践题目.docx" purpose="source"}

| 组员                | 负责内容                                                     | 主要文件                                                     | 验收标准                                                     |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 组员A：数据与分段   | 数据集预处理；部署IoTDB；批量导入与查询；实现两种自动分段方法；分段参数调优与对比 | `data_loader.py`、`segmentation.py`                          | IoTDB可正常查询；输出统一DataFrame；至少两种分段算法；包含最小长度约束及参数选择依据 |
| 组员B：特征与聚类   | 分段特征提取及标准化；实现两种聚类算法；选择最优聚类数；生成工况ID和持续时间统计 | `feature_extraction.py`、`clustering.py`                     | 至少覆盖4类特征；至少两种聚类算法；输出Silhouette/CH等指标；生成`OP_001`形式标签 |
| 组员C：可视化与整合 | 完成四类规定图表；编排完整流水线；整理依赖和运行说明；汇总实验报告；组织验收演示 | `visualization.py`、`main.py`、`requirements.txt`、`README.md`、实验报告 | 四类图表齐全；项目可一键运行；报告8～15页；完成端到端演示    |

其中，组员A负责题目权重30%，组员B负责45%，组员C显性任务为25%，但组员C还承担项目整合、报告排版、运行复现和演示准备，因此实际工作量基本平衡。

### 必须提前统一的接口

为了避免最后无法合并，第一天就固定下面四种数据格式：

1. `raw_df`：时间为索引，每个传感器一个列。
2. `segments`：包含`segment_id、start_idx、end_idx、start_time、end_time`。
3. `features`：每行对应一个分段，并保留`segment_id`。
4. `labels`：包含`segment_id、cluster、operation_id、duration`。

建议默认选择ETT数据集：CSV格式、7维、连续时间较长，下载和导入IoTDB都比较方便。

### 报告分工

- 组员A撰写：数据集、预处理、IoTDB、分段算法和参数调优。
- 组员B撰写：特征设计、标准化、聚类算法和评价指标对比。
- 组员C撰写：可视化分析、问题与解决方案、结论，并负责最终合并和排版。
- 每个人提交自己部分的实验参数、结果表格和图，不能只交文字给组员C。

### 推荐进度

- 第1天：共同确定数据集、IoTDB路径和接口格式。
- 第2～3天：三人使用模拟输入并行开发各自模块。
- 第4天：首次合并，跑通`main.py`完整流程。
- 第5天：参数调优、算法对比、生成最终图表。
- 第6天：完成报告、README和演示彩排。

验收演示时，组员A讲数据接入与分段，组员B讲特征与聚类，组员C讲可视化、整体流程和结论。这样每个人都有明确可展示的贡献。

# WeatherDataset 高维时间序列自动分段实验

本项目使用 Apache IoTDB 和 Python 对 WeatherDataset 气象多变量时间序列进行存储、预处理和自动分段，并比较两种高维时间序列变点检测方法：

1. 基于 `ruptures` 库的 PELT（Pruned Exact Linear Time）方法；
2. 基于滑动窗口和对称 KL 散度（Kullback–Leibler Divergence）的方法。

项目不仅提供最终分段程序，还保留了参数选择实验，便于复现实验过程、理解参数影响或重新选择参数。

## 1. 数据集

实验使用 WeatherDataset，主要特点如下：

- 52,696 个时间点；
- 每 10 分钟采样一次；
- 时间范围约为 2020-01-01 至 2021-01-01；
- 21 个气象变量，例如温度、湿度、气压和风速等；
- CSV 格式；
- 原始数据中存在少量缺失值或哨兵值，程序会自动进行线性插值。

请将数据文件命名为 `weather.csv`，并放在项目的 `data` 文件夹中：

```text
项目根目录/
└── data/
    └── weather.csv
```

## 2. 项目结构

```text
项目根目录/
├── data/
│   └── weather.csv                 # WeatherDataset 原始数据
├── test_code4pelt/                 # PELT 参数实验代码与实验结果
├── test_code4window/               # 滑动窗口 KL 参数实验代码与实验结果
├── import_weather.py               # 将 Weather 数据导入 IoTDB
├── segmentation.py                 # 使用最终参数运行两种分段方法
└── README.md
```

其中：

- `test_code4pelt` 用于依次研究 PELT 的惩罚系数、最短分段长度和 `jump`；
- `test_code4window` 用于研究滑动窗口长度和 KL 阈值系数；
- `segmentation.py` 是最终使用文件，一次运行即可生成两种方法的分段表；
- `import_weather.py` 用于完成 WeatherDataset 到 Apache IoTDB 的数据导入。

## 3. 实验流程

整个实验按照以下步骤进行：

1. 部署并启动 Apache IoTDB；
2. 清洗 WeatherDataset，并将气象数据导入 IoTDB；
3. 对所有数值维度进行 Z-score 标准化；
4. 对 PELT 的惩罚系数、最短分段长度和搜索步长进行参数实验；
5. 对滑动窗口 KL 方法的窗口长度和阈值系数进行参数实验；
6. 使用 BIC、相邻参数稳定性、分段长度和两种方法的一致性综合选择参数；
7. 使用最终参数生成两套分段结果。

本实验采用多维联合检测，即同时利用21个气象变量判断变点，而不是分别检测每个维度后再取并集。

## 4. 数据预处理

`segmentation.py` 会自动执行以下处理：

1. 将 CSV 第一列识别为时间列；
2. 将其余列转换为数值；
3. 将小于等于 `-9990` 的哨兵值视为缺失值；
4. 使用线性插值填补缺失值；
5. 按时间从早到晚排序；
6. 对每个气象维度分别进行 Z-score 标准化。

标准化是必要的，因为温度、气压、湿度和风速等变量的单位及数值范围不同。如果直接计算距离，数值较大的变量会对结果产生不合理的主导作用。

## 5. 方法 A：PELT

PELT 通过寻找一组变点，使各分段内部的误差和变点惩罚之和最小。本项目使用多维 L2 代价，因此主要检测气象变量整体均值结构的变化。

最终参数为：

```text
penalty multiplier c = 0.6
min_size = 144
jump = 6
```

参数含义：

- `c=0.6`：控制新增变点所需付出的惩罚；
- `min_size=144`：任意分段至少包含144个采样点，即至少24小时；
- `jump=6`：每隔6个采样点搜索一次候选位置，即1小时的搜索精度。

实际惩罚值按以下公式计算：

```text
penalty = c × 数据维度 × ln(时间点数量)
```

参数实验表明，`jump=6` 与最精细的 `jump=1` 得到的变点结果几乎一致，但运行时间明显更短，因此最终采用 `jump=6`。

最终结果约为：

```text
266 个变点
267 个分段
```

## 6. 方法 B：滑动窗口对称 KL 散度

该方法在每个候选位置建立左右两个滑动窗口，分别估计21维数据的均值和方差，再计算两个对角高斯分布之间的对称 KL 散度。

当左右窗口的气象分布差异较大时，KL 曲线会形成峰值。超过稳健阈值并满足最小间隔要求的峰值被识别为变点。

最终参数为：

```text
window = 24
threshold multiplier m = 0.5
minimum peak distance = 144
smoothing = 6
```

参数含义：

- `window=24`：变点左右各使用24个采样点，即左右各4小时；
- `m=0.5`：阈值为 KL 曲线中位数加 `0.5` 倍稳健尺度；
- `minimum peak distance=144`：相邻变点至少间隔24小时；
- `smoothing=6`：对 KL 曲线进行1小时尺度的平滑。

最终参数是在窗口长度和阈值系数网格实验中，根据均值型 BIC、高斯均值-方差 BIC、阈值邻域稳定性、分段长度约束和 PELT 一致性综合确定的。

最终结果约为：

```text
260 个变点
261 个分段
```

两种方法在允许 ±12 小时时间偏差时具有较高的一致性；允许 ±24 小时时，一致性进一步提高。严格时间容差下的一致性较低是正常的，因为 PELT 主要关注整体均值变化，而 KL 方法同时关注均值和方差变化，且天气过程本身通常是逐渐过渡的。

## 7. 环境安装

建议使用 Python 3.9 或更高版本。在项目根目录打开 PowerShell 或终端。

### 7.1 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux 或 macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 7.2 安装分段程序依赖

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn ruptures
```

如果需要执行 IoTDB 导入程序，还需要安装：

```bash
python -m pip install apache-iotdb
```

## 8. 快速使用最终分段程序

确认目录中存在：

```text
data/weather.csv
segmentation.py
```

然后在项目根目录运行：

```bash
python segmentation.py
```

程序运行结束后，只会在当前文件夹生成两个结果文件：

```text
pelt_segments.csv
kl_segments.csv
```

如果数据不在默认位置，可以使用 `--input` 指定路径：

```bash
python segmentation.py --input "你的路径/weather.csv"
```

例如 Windows：

```powershell
python .\segmentation.py --input ".\data\weather.csv"
```

## 9. 输出分段表说明

`pelt_segments.csv` 和 `kl_segments.csv` 的字段相同：

| 字段                  | 含义                       |
| --------------------- | -------------------------- |
| `method`              | 分段方法名称               |
| `segment_number`      | 分段编号，从1开始          |
| `start_index`         | 分段起始下标               |
| `end_index_exclusive` | 分段结束下标，不包含该位置 |
| `end_index_inclusive` | 分段内最后一个数据点下标   |
| `start_time`          | 分段起始时间               |
| `end_time`            | 分段结束时间               |
| `length_samples`      | 分段包含的采样点数量       |
| `length_hours`        | 分段持续时间，单位为小时   |

本项目主要采用 Python 常用的左闭右开边界形式：

```text
[start_index, end_index_exclusive)
```

例如：

```text
start_index = 0
end_index_exclusive = 144
```

表示该分段包含下标 `0` 至 `143`，总计144个采样点。

## 10. 将数据导入 Apache IoTDB

这一步用于复现实验中的数据库存储部分，不是运行最终 `segmentation.py` 的必要条件。最终分段程序直接读取 CSV，因此没有安装 IoTDB 的同学也可以运行。

首先确保 IoTDB 服务已经启动，并确认默认连接信息：

```text
host: 127.0.0.1
port: 6667
username: root
password: root
```

然后执行：

```bash
python import_weather.py
```

本实验使用的时间序列设备路径为：

```text
root.weather.station001
```

可以在 IoTDB CLI 中验证：

```sql
SHOW TIMESERIES root.weather.**;
```

```sql
SELECT * FROM root.weather.station001 LIMIT 10;
```

如果其他同学的 CSV 路径、IoTDB 地址、用户名或密码不同，请先检查并修改 `import_weather.py` 开头的配置。

## 11. 参数实验代码

最终复现只需要运行 `segmentation.py`。如果需要重新研究参数，可以查看：

```text
test_code4pelt/
test_code4window/
```

PELT 参数实验主要包括：

1. 惩罚系数实验；
2. 最短分段长度实验；
3. `jump` 搜索步长实验。

滑动窗口 KL 参数实验主要包括：

1. 窗口长度实验；
2. 阈值系数实验；
3. 最终候选参数比较；
4. 与 PELT 在不同时间容差下的一致性比较。

需要注意，BIC 越小通常表示拟合效果与模型复杂度之间的权衡越好，但不应仅依靠 BIC 机械选择参数。最短分段长度具有明确的业务含义，最终选择还应考虑分段长度是否合理、参数是否稳定以及变点是否具有可解释性。

## 12. 常见问题

### 找不到 `data/weather.csv`

确认运行命令时所在目录是项目根目录，或者使用：

```bash
python segmentation.py --input "weather.csv 的实际路径"
```

### 出现 `ModuleNotFoundError`

重新安装依赖：

```bash
python -m pip install numpy pandas scipy scikit-learn ruptures
```

### Windows PowerShell 无法激活虚拟环境

可以执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新运行：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 运行时间较长

PELT 需要处理52,696行、21维数据。最终脚本已经使用 `jump=6` 降低计算量，请等待程序输出结果，不建议直接结束进程。

### 结果数量与本文略有不同

请确认：

- 使用的是同一版本的 WeatherDataset；
- CSV 行数和字段一致；
- 缺失值没有被提前以其他方式处理；
- 未修改 `segmentation.py` 中的最终参数。

## 13. 说明

本项目用于高维时间序列存储、变点检测与自动分段实验。两种方法产生的分段结果可继续用于分段特征提取、聚类、工况识别和可视化分析。

