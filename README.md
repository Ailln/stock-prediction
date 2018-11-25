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

# 自定义你的配置
vi config.yaml

# 每次运行需要手动替换配置中的 $model_name
python -m run.sklearn
```

## 3 结果

| 序号 | 模型类型 | 模型名称 | MSE(10**-5) | 日期 |
| :-: | :- | :- | :- | :- |
| 1 | linear | SGDRegressor | **31.3744** | 2018.11.26 01:36 |
| 2 | linear | HuberRegressor | 31.6236 | 2018.11.26 01:35 |
| 3 | linear | LinearRegression | **31.1200** | 2018.11.25 23:23 |
| 4 | svm | SVR | 87.8340 | 2018.11.26 04:25 |
| 5 | svm | LinearSVR | 37.6120 | 2018.11.26 02:36 |
| 6 | ensemble | BaggingRegressor | 36.1064 | 2018.11.26 01:48 |
| 7 | ensemble | AdaBoostRegressor | 61.5127 | 2018.11.26 01:20 |
| 8 | ensemble | ExtraTreesRegressor | 35.9465 | 2018.11.26 01:34 |
| 9 | ensemble | RandomForestRegressor | 36.1245 | 2018.11.26 01:33 |
| 10 | ensemble | GradientBoostingRegressor | **31.1003** | 2018.11.26 01:21 |

注：加粗的 MSE 是前 3 名。

## 4 其他

Q: 数据在哪里？

A: 好问题！别急，先等等。
