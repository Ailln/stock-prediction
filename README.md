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

本项目使用的 Python 版本必须大于 3.6.0，环境配置参考[这里](https://www.v2ai.cn/linux/2018/04/29/LX-2.html)。

```bash
git clone https://github.com/kinggreenhall/stock-prediction.git

cd stock-prediction

# 安装依赖
pip install -r requirements.txt

# 修改配置
vi config.yaml

# 每次运行需要手动替换配置中的 model_name
python -m run.sklearn
```

## 3 结果

| 序号 | 模型类型 | 模型名称 | MSE | 日期 |
| :-: | :- | :- | :- | :- |
| 1 | linear | LinearRegression | <font color=orange>0.0003111997484069403</font> | 2018.11.25 23:23 |
| 2 | linear | HuberRegressor | 0.00031623554288487003 | 2018.11.26 01:35 |
| 3 | linear | SGDRegressor | 0.0003137441739156157 | 2018.11.26 01:36 |
| 4 | ensemble | AdaBoostRegressor | 0.0006151271633934587 | 2018.11.26 01:20 |
| 5 | ensemble | GradientBoostingRegressor | <font color=red>0.0003110028137168617</font> | 2018.11.26 01:21 |
| 6 | ensemble | RandomForestRegressor | 0.00036124471790637946 | 2018.11.26 01:33 |
| 7 | ensemble | BaggingRegressor | 0.00036106425975312523 | 2018.11.26 01:48 |
| 8 | ensemble | ExtraTreesRegressor | 0.00035946545498117974 | 2018.11.26 01:34 |

## 4 其他

Q: 数据在哪里？

A: 好问题！别急，先等等。
