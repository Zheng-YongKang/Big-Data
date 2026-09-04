# WeatherDataset 高维时间序列自动分段实验

本项目使用 Apache IoTDB 和 Python 对 WeatherDataset 气象多变量时间序列进行存储、预处理和自动分段，并比较两种高维时间序列变点检测方法：

1. 基于 `ruptures` 库的 PELT（Pruned Exact Linear Time）方法；
2. 基于滑动窗口和对称 KL 散度（Kullback–Leibler Divergence）的方法。

项目不仅提供最终分段程序，还保留了参数选择实验，便于复现实验过程、理解参数影响或重新选择参数。

## 1. 数据集

实验使用 WeatherDataset，主要特点如下：

- 原始 CSV 共 52,696 行，其中包含1个重复时间戳；
- 按统一规则去重后得到 52,695 个唯一时间点；
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

`segmentation.py` 默认先按指定时间范围从 IoTDB 查询数据，并通过官方
`SessionDataSet.todf()` 方法转换成 `pandas.DataFrame`，然后自动执行以下处理：

`import_weather.py` 和 `segmentation.py` 使用完全相同的预处理顺序：

1. 识别并解析时间列；
2. 将气象变量转换为数值；
3. 使用稳定排序按时间从早到晚排列；
4. 删除重复时间戳，并统一保留重复记录中的最后一条；
5. 将小于等于 `-9990` 的81个哨兵值转换为缺失值；
6. 使用线性插值填补缺失值；
7. 检查时间戳唯一性、时间顺序、哨兵值和缺失值；
8. 在分段前对每个气象维度分别进行 Z-score 标准化。

经过处理后，CSV 模式和 IoTDB 模式都使用同一条包含52,695个时间点的
时间轴，保证分段下标与数据库查询结果严格对应。

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

### 7.2 安装程序依赖

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn ruptures apache-iotdb
```

## 8. 快速使用最终分段程序

首先确认 IoTDB 已启动，并使用当前版本的导入脚本重新导入数据：

```bash
python import_weather.py
```

修正后的导入程序会在写入前处理重复时间戳和 `-9999` 哨兵值。它会使用
清洗后的数值覆盖相同时间戳上的旧值，因此通常不需要删除原数据库。

导入后的信息为：

```text
设备路径：root.weather.station001
默认地址：127.0.0.1:6667
```

然后在项目根目录运行。程序默认查询
`2020-01-01 00:10:00` 至 `2021-01-01 00:00:00`：

```bash
python segmentation.py
```

也可以自定义查询时间范围：

```bash
python segmentation.py --start "2020-03-01 00:00:00" --end "2020-06-01 00:00:00"
```

自定义 IoTDB 连接信息和设备路径：

```bash
python segmentation.py --host 127.0.0.1 --port 6667 --user root --password root --device root.weather.station001
```

程序运行结束后，只会在当前文件夹生成两个结果文件：

```text
pelt_segments.csv
kl_segments.csv
```

如果暂时没有启动 IoTDB，可以使用 CSV 备用模式：

```bash
python segmentation.py --source csv --input "你的路径/weather.csv"
```

例如 Windows：

```powershell
python .\segmentation.py --source csv --input ".\data\weather.csv"
```

## 9. 输出分段表说明

`pelt_segments.csv` 和 `kl_segments.csv` 的字段相同：

| 字段 | 含义 |
| --- | --- |
| `method` | 分段方法名称 |
| `segment_number` | 分段编号，从1开始 |
| `start_index` | 分段起始下标 |
| `end_index_exclusive` | 分段结束下标，不包含该位置 |
| `end_index_inclusive` | 分段内最后一个数据点下标 |
| `start_time` | 分段起始时间 |
| `end_time` | 分段结束时间 |
| `length_samples` | 分段包含的采样点数量 |
| `length_hours` | 分段持续时间，单位为小时 |

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

这一步用于复现实验中的数据库存储和查询过程。最终 `segmentation.py` 默认从
IoTDB 查询并返回 `pandas.DataFrame`；没有安装 IoTDB 的同学仍可使用
`--source csv` 备用模式运行。

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
python segmentation.py --source csv --input "weather.csv 的实际路径"
```

### 出现 `ModuleNotFoundError`

重新安装依赖：

```bash
python -m pip install numpy pandas scipy scikit-learn ruptures apache-iotdb
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

### 行数不是 52,695

请检查终端输出。标准数据应满足：

```text
Rows: 52,695
Remaining duplicate/sentinel/missing values: 0
```

如果 IoTDB 返回的行数不同，请重新运行修正后的 `import_weather.py`，并确认
导入和查询使用的设备路径都是 `root.weather.station001`。

### 运行时间较长

PELT 需要处理52,695行、21维数据。最终脚本已经使用 `jump=6` 降低计算量，请等待程序输出结果，不建议直接结束进程。

### 结果数量与本文略有不同

请确认：

- 使用的是同一版本的 WeatherDataset；
- CSV 行数和字段一致；
- 缺失值没有被提前以其他方式处理；
- 未修改 `segmentation.py` 中的最终参数。

## 13. 说明

本项目用于高维时间序列存储、变点检测与自动分段实验。两种方法产生的分段结果可继续用于分段特征提取、聚类、工况识别和可视化分析。
