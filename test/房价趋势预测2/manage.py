#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import numpy as np
import pandas as pd
import pylab
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', '房价趋势预测2.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    # 查看数据
    data.head()
    # 查看数据集形状
    data.shape
    # 查看数据集数据类型
    data.dtypes

    # 2. 分析目标变量 在分析之前，首先要了解目标，查看目标类型、分布，有无异常值。根据不同的类型选择不同的模型。
    data['SalePrice'].describe()
    # 再看看目标变量分布：
    sns.distplot(data['SalePrice'])
    plt.show()

    # 3. 挑选最佳特征 从80个特征中选出与目标变量SalePrice相关的特征。
    # a. 针对连续型变量，可以使用“皮尔逊相关系数”找出与目标变量最相关的特征
    # 1） 一楼面积
    sns.jointplot(x='1stFlrSF', y='SalePrice', data=data)
    plt.show()
    # 2）房屋面积
    sns.jointplot(x='GrLivArea', y='SalePrice', data=data)
    sns.lmplot(x='GrLivArea', y='SalePrice', data=data)
    # 3）泳池面积
    sns.jointplot(x='PoolArea', y='SalePrice', data=data)

    # b. 针对分类变量，无法使用皮尔逊相关系数，可以通过观察每个分类值上目标变量的变化程度来查看相关性，通常来说，在不同值上数据范围变化较大，两变量相关性较大。
    # 盒须图
    # 1) 房屋材料与质量
    sns.boxplot(x='OverallQual', y='SalePrice', data=data)
    # 2）建造年代
    sns.boxplot(x='YearBuilt', y='SalePrice', data=data)
    # 柱状图
    # 使用groupby将价格按照特征分类，再去平均值，使用柱状图展示
    grouped = data.groupby('OverallQual')
    g1 = grouped['SalePrice'].mean().reset_index('OverallQual')
    sns.barplot(x='OverallQual', y='SalePrice', data=g1)

    # c. 以上两种分析都是针对单个特征与目标变量逐一分析，这种方法非常耗时繁琐，下面介绍一种系统性分析特征与目标变量相关性的方法，通过对数据集整体特征（数值型数据）进行分析，来找出最佳特征。
    # 热力图  sns.heatmap()
    # 设置图幅大小
    pylab.rcParams['figure.figsize'] = (15, 10)
    # 计算相关系数
    corrmatrix = data.corr()
    # 绘制热力图，热力图横纵坐标分别是data的index/column,vmax/vmin设置热力图颜色标识上下限，center显示颜色标识中心位置，cmap颜色标识颜色设置
    sns.heatmap(corrmatrix, square=True, vmax=1, vmin=-1, center=0.0, cmap='coolwarm')

    # 特征较多，且相关性不大的特征可以忽略，选取相关性排前十的特征：
    # 取相关性前10的特征
    k = 10
    # data.nlargest(k, 'target')在data中取‘target'列值排前十的行
    # cols为排前十的行的index,在本例中即为与’SalePrice‘相关性最大的前十个特征名
    cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(data[cols].values.T)
    # data[cols].values.T
    # 设置坐标轴字体大小
    sns.set(font_scale=1.25)
    # sns.heatmap() cbar是否显示颜色条，默认是；cmap显示颜色；annot是否显示每个值，默认不显示；
    # square是否正方形方框，默认为False,fmt当显示annotate时annot的格式；annot_kws为annot设置格式
    # yticklabels为Y轴刻度标签值，xticklabels为X轴刻度标签值
    hm = sns.heatmap(cm, cmap='RdPu', annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)

    # 上例提供了求相关系数另一种方法，也可以直接用data.corr(),更方便
    cm1 = data[cols].corr()
    hm2 = sns.heatmap(cm1, square=True, annot=True, cmap='RdPu', fmt='.2f', annot_kws={'size': 10})

    cols1 = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd',
             'YearBuilt']
    sns.pairplot(data[cols1], size=2.5)

    # 4.数据处理
    # a.Missingvalue
    # isnull() boolean, isnull().sum()统计所有缺失值的个数
    # isnull().count()统计所有项个数（包括缺失值和非缺失值），.count()统计所有非缺失值个数
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    # pd.concat() axis=0 index,axis=1 column, keys列名
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)

    # 缺失值超过80%的有特征“PoolQC”, "MiscFeature", "Alley", "Fence"，可以认定这些特征无效，可以剔除。
    # 处理缺失值，将含缺失值的整列剔除
    data1 = data.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
    # 由于特征Electrical只有一个缺失值，故只需删除该行即可
    data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)
    # 检查缺失值数量
    data2.isnull().sum().max()

    # 5. 建立模型 首先划分数据集，使用sklearn里面的train_test_split()可以将数据集划分为训练集和测试集。
    feature_data = data2.drop(['SalePrice'], axis=1)
    target_data = data2['SalePrice']
    # 将数据集划分为训练集和测试集
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.3)

    # a. 线性回归模型  线性回归模型是最简单的模型，实际应用中已经很少用到了，作为基础知识练习
    from statsmodels.formula.api import ols
    from statsmodels.sandbox.regression.predstd import wls_prediction_std

    df_train = pd.concat([X_train, y_train], axis=1)
    # ols("target~feature+C(feature)", data=data
    # C(feature)表示这个特征为分类特征category
    lr_model = ols("SalePrice~C(OverallQual)+GrLivArea+C(GarageCars)+TotalBsmtSF+C(FullBath)+YearBuilt",
                   data=df_train).fit()
    print(lr_model.summary())

    # 预测测试集
    lr_model.predict(X_test)
    #判定系数R2表示，房价“SalePrice”的变异性的79%，能用该多元线性回归方程解释；

    #在该多元线性回归方程中，也有很多特征的P_value大于0.05，说明这些特征对y值影响非常小，可以剔除。
    # prstd为标准方差，iv_l为置信区间下限，iv_u为置信区间上限
    prstd, iv_l, iv_u = wls_prediction_std(lr_model, alpha = 0.05)
    # lr_model.predict()为训练集的预测值
    predict_low_upper = pd.DataFrame([lr_model.predict(),iv_l, iv_u],index=['PredictSalePrice','iv_l','iv_u']).T
    predict_low_upper.plot(kind='hist',alpha=0.4)