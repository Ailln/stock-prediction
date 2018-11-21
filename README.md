# stock-prediction

股票预测。

## 1 数据简介

1. 非时间序列类型指标
    - 代表股票的一些基本特性。
    - 其中有部分连续指标做过正态化，也包含部分不连续指标（离散或缺失）。
    - `flag` 一列带买哦股票的分类属性，属于类别指标。

2. 时间序列类型指标
    - {t0, t1, t2..., t20} 代表股票的某一个属性。
    - 并未全部做过正态化处理。

3. 股票未来收益
    - 存储在 `y.csv`

### train_data 训练数据

- 2013/02 - 2017/03

```
├── 20130201 数据选取时间
│   ├── non_ts.csv 非时间序列类型指标
│   ├── ts_1.csv 时间序列类型指标
│   ├── ts_2.csv 时间序列类型指标
│   ├── ts_3.csv 时间序列类型指标
│   ├── ts_4.csv 时间序列类型指标
│   ├── ts_5.csv 时间序列类型指标
│   └── y.csv 股票未来收益
├── 20130204
...
```

### test_data 测试数据

- 2017/04 - 2018/09

```
├── 20170330 数据选取时间
│   ├── non_ts.csv 非时间序列类型指标
│   ├── ts_1.csv 时间序列类型指标
│   ├── ts_2.csv 时间序列类型指标
│   ├── ts_3.csv 时间序列类型指标
│   ├── ts_4.csv 时间序列类型指标
│   └── ts_5.csv 时间序列类型指标
├── 20170331
...
```

## 2 使用方法

```bash
git clone https://github.com/kinggreenhall/stock-prediction.git

cd stock-prediction

# 安装依赖
pip install -r requirements.txt

# 数据预处理
python -m utils.data_utils
```