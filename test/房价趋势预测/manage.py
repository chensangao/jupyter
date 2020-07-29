#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# 导入一些数据分析和数据挖掘常用的包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', '房价趋势预测.settings')
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
    # 加载一下数据，并打印部分数据，查看一下数据的情况
    data_train = pd.read_csv("house_train.csv")
    data_test = pd.read_csv("house_test.csv")
    # print(data_train.head())
    # print(data_test.head())

    # 查看数据的列名和每列的数据格式，方便后面对数据进行处理
    data_train.columns
    # data_tarin.info
    data_train_dtypes = data_train.dtypes
    # print(data_train_dtypes)

    # 描述统计因变量的基础情况
    # print(data_train['SalePrice'].describe())
    # 查看因变量数值情况
    # print(data_train['SalePrice'].value_counts())

    # 查看因变量价格的情况，进行基础分析
    sns.distplot(data_train['SalePrice'])
    plt.show()

    # 对房屋价格金额的数值图形化，查看一下
    sns.set(style="darkgrid")
    titanic = pd.DataFrame(data_train['SalePrice'].value_counts())
    titanic.columns = ['SalePrice_count']
    ax = sns.countplot(x="SalePrice_count", data=titanic)
    plt.show()

    # 从房屋价格的正态分布情况，查看房屋价格的峰度和偏度情况
    print('房屋价格偏度：%f' % (data_train['SalePrice'].skew()))
    print('房屋价格峰度：%f' % (data_train['SalePrice'].kurt()))

    # 在进行图形分析之前，先分析一下数据中缺失值的情况
    miss_data = data_train.isnull().sum().sort_values(ascending=False)  # 缺失值数量
    total = data_train.isnull().count()  # 总数量
    miss_data_tmp = miss_data / total.sort_values(ascending=False)  # 缺失值占比


    # 添加百分号
    def precent(X):
        X = '%.2f%%' % (X * 100)
        return X


    miss_precent = miss_data_tmp.map(precent)
    # 根据缺失值占比倒序排序
    miss_data_precent = pd.concat([total, miss_precent, miss_data_tmp], axis=1, keys=[
        'total', 'Percent', 'Percent_tmp']).sort_values(by='Percent_tmp', ascending=False)
    # 有缺失值的变量打印出来
    print(miss_data_precent[miss_data_precent['Percent'] != '0.00%'])

    # 将缺失值比例大于15%的数据全部删除，数值型变量用众数填充、类别型变量用None填充
    drop_columns = miss_data_precent[miss_data_precent['Percent_tmp'] > 0.15].index
    data_train = data_train.drop(drop_columns, axis=1)
    data_test = data_test.drop(drop_columns, axis=1)
    # 类别型变量
    class_variable = [
        col for col in data_train.columns if data_train[col].dtypes == 'O']
    # 数值型变量
    numerical_variable = [
        col for col in data_train.columns if data_train[col].dtypes != 'O']  # 大写o
    print('类别型变量:%s' % class_variable, '数值型变量:%s' % numerical_variable)
    # 数值型变量用中位数填充，test集中最后一列为预测价格，所以不可以填充
    from sklearn.preprocessing import Imputer

    padding = Imputer(strategy='median')
    data_train[numerical_variable] = padding.fit_transform(
        data_train[numerical_variable])
    data_test[numerical_variable[:-1]
    ] = padding.fit_transform(data_test[numerical_variable[:-1]])
    # 类别变量用None填充
    data_train[class_variable] = data_train[class_variable].fillna('None')
    data_test[class_variable] = data_test[class_variable].fillna('None')

    # 根据变量，选择相关变量查看与因变量之间的关系
    # CentralAir 中央空调
    data = pd.concat([data_train['SalePrice'], data_train['CentralAir']], axis=1)
    fig = sns.boxplot(x='CentralAir', y="SalePrice", data=data)
    plt.title('CentralAir')
    plt.show()
    # 有中央空调的房价更高

    # MSSubClass 房屋类型
    data = pd.concat([data_train['SalePrice'], data_train['MSSubClass']], axis=1)
    fig = sns.boxplot(x='MSSubClass', y="SalePrice", data=data)
    plt.title('MSSubClass')
    plt.show()
    # 房屋类型方面和房屋价格之间关联度不大

    # MSZoning 房屋区域，例如有高密度住宅、低密度住宅，猜想这个变量和房屋价格之间的关系比较密切
    data = pd.concat([data_train['SalePrice'], data_train['MSZoning']], axis=1)
    fig = sns.boxplot(x='MSZoning', y="SalePrice", data=data)
    plt.title('MSZoning')
    plt.show()
    # 实际结果显示和房屋价格的关系不大

    # LotArea 这个变量猜想会与房屋价格有直接的关系
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=data_train['SalePrice'], y=data_train['LotArea'])
    plt.xlabel('SalePrice')
    plt.ylabel('LotArea')
    plt.title('LotArea')
    plt.show()
    # 地块面积与房屋价格有较高相关性

    # Street
    data = pd.concat([data_train['SalePrice'], data_train['Street']], axis=1)
    fig = sns.boxplot(x='Street', y="SalePrice", data=data)
    plt.title('Street')
    plt.show()
    # 街道类型和房屋价格有较高相关性

    # LotShape
    data = pd.concat([data_train['SalePrice'], data_train['LotShape']], axis=1)
    fig = sns.boxplot(x='LotShape', y="SalePrice", data=data)
    plt.title('LotShape')
    plt.show()
    # 房屋形状和房屋价格相关不明显

    # LandContour
    data = pd.concat([data_train['SalePrice'], data_train['LandContour']], axis=1)
    fig = sns.boxplot(x='LandContour', y="SalePrice", data=data)
    plt.title('LandContour')
    plt.show()
    # 房屋所在地是否平整和房屋价格关系较弱

    # Utilities
    data = pd.concat([data_train['SalePrice'], data_train['Utilities']], axis=1)
    fig = sns.boxplot(x='Utilities', y="SalePrice", data=data)
    plt.title('Utilities')
    plt.show()
    # 公共设施和房屋价格基本无关系

    # LotConfig
    data = pd.concat([data_train['SalePrice'], data_train['LotConfig']], axis=1)
    fig = sns.boxplot(x='LotConfig', y="SalePrice", data=data)
    plt.title('LotConfig')
    plt.show()

    # 绘制热力图，查看一下数值型变量之间的关系
    corrmat = data_train[numerical_variable].corr('spearman')
    f, ax = plt.subplots(figsize=(12, 9))
    ax.set_xticklabels(corrmat, rotation='horizontal')
    sns.heatmap(np.fabs(corrmat), square=False, center=1)
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=90)
    plt.show()

    # 计算变量之间的相关性
    numerical_variable_corr = data_train[numerical_variable].corr('spearman')
    numerical_corr = numerical_variable_corr[
        numerical_variable_corr['SalePrice'] > 0.5]['SalePrice']
    print(numerical_corr.sort_values(ascending=False))
    index0 = numerical_corr.sort_values(ascending=False).index
    # 结合考虑两两变量之间的相关性
    print(data_train[index0].corr('spearman'))

    # 结合上述情况，选择出相关性大于0.5的变量，在这个基础上再考虑变量之间的多重共线性
    new_numerical = ['OverallQual', 'GrLivArea', 'GarageCars',
                     'YearBuilt', 'FullBath', 'TotalBsmtSF', 'YearRemodAdd', 'Fireplaces']
    X = np.matrix(data_train[new_numerical])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    VIF_list
    # 可以明显看到数据有很强的多重共线性，对数据进行标准化和降维
    Scaler = StandardScaler()
    data_train_numerical = Scaler.fit_transform(data_train[new_numerical])
    pca = PCA(n_components=7)
    newData_train = pca.fit_transform(data_train_numerical)
    newData_train

    Scaler = StandardScaler()
    data_test_numerical = Scaler.fit_transform(data_test[new_numerical])
    pca = PCA(n_components=7)
    newData_test = pca.fit_transform(data_test_numerical)
    newData_test

    newData_train = pd.DataFrame(newData_train)
    # newData
    y = np.matrix(newData_train)
    VIF_list = [variance_inflation_factor(y, i) for i in range(y.shape[1])]
    print(newData_train, VIF_list)

    # 从上面的数据标准化和降维之后，已经消除了多重共线性了。接下来处理类别数据
    # 单因素方差分析
    # 我们需要看的是单个自变量对因变量SalePrice的影响，因此这里使用单因素方差分析。
    # 分析结果中 P_values（PR(>F)）越小，说明该变量对目标变量的影响越大。
    # 通常我们只选择 P_values 小于 0.05 的变量

    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    a = '+'.join(class_variable)
    formula = 'SalePrice~ %s' % a
    anova_results = anova_lm(ols(formula, data_train).fit())
    print(anova_results.sort_values(by='PR(>F)'))
    # 从变量列表和数据中剔除 P 值大于 0.05 的变量
    del_var = list(anova_results[anova_results['PR(>F)'] > 0.05].index)
    del_var
    # 移除变量
    for each in del_var:
        class_variable.remove(each)
    # 移除变量数据
    data_train = data_train.drop(del_var, axis=1)
    data_test = data_test.drop(del_var, axis=1)


    # 对类别型变量进行编码
    def factor_encode(data):
        map_dict = {}
        for each in data.columns[:-1]:
            piv = pd.pivot_table(data, values='SalePrice',
                                 index=each, aggfunc='mean')
            piv = piv.sort_values(by='SalePrice')
            piv['rank'] = np.arange(1, piv.shape[0] + 1)
            map_dict[each] = piv['rank'].to_dict()
        return map_dict


    class_variable.append('SalePrice')
    # 调用上面的函数，对名义特征进行编码转换
    # class_variable.append('SalePrice')
    map_dict = factor_encode(data_train[class_variable])
    for each_fea in class_variable[:-1]:
        data_train[each_fea] = data_train[each_fea].replace(map_dict[each_fea])
        data_test[each_fea] = data_test[each_fea].replace(map_dict[each_fea])

    # 因为上面已经完成编码，这里我们再根据相关性判断和选择变量
    class_coding_corr = data_train[class_variable].corr('spearman')['SalePrice'].sort_values(ascending=False)
    print(class_coding_corr[class_coding_corr > 0.5])
    class_0 = class_coding_corr[class_coding_corr > 0.5].index
    data_train[class_0].corr('spearman')

    # 查找两两之间的共线性之后，我们保留如下变量
    # Neighborhood，ExterQual，BsmtQual，GarageFinish，GarageType，GarageType；
    # 接下来尝试查看多重共线性
    class_variable = ['Neighborhood', 'ExterQual', 'BsmtQual',
                      'GarageFinish', 'GarageType', 'Foundation']
    X = np.matrix(data_train[class_variable])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    VIF_list

    Scaler = StandardScaler()
    data_train_class = Scaler.fit_transform(data_train[class_variable])
    pca = PCA(n_components=3)
    newData_train_class = pca.fit_transform(data_train_class)
    newData_train_class

    Scaler = StandardScaler()
    data_test_class = Scaler.fit_transform(data_test[class_variable])
    pca = PCA(n_components=3)
    newData_test_class = pca.fit_transform(data_test_class)
    newData_test_class

    newData_train_class = pd.DataFrame(newData_train_class)
    y = np.matrix(newData_train_class)
    VIF_list = [variance_inflation_factor(y, i) for i in range(y.shape[1])]
    print(VIF_list)

    # 训练集
    newData_train_class = pd.DataFrame(newData_train_class)
    newData_train_class.columns = ['降维后类别A', '降维后类别B', '降维后类别C']
    newData_train = pd.DataFrame(newData_train)
    newData_train.columns = ['降维后数值A', '降维后数值B', '降维后数值C', '降维后数值D', '降维后数值E', '降维后数值F', '降维后数值G']
    target = data_train['SalePrice']
    target = pd.DataFrame(target)
    train = pd.concat([newData_train_class, newData_train], axis=1, ignore_index=True)

    # 测试集
    newData_test_class = pd.DataFrame(newData_test_class)
    newData_test_class.columns = ['降维后类别A', '降维后类别B', '降维后类别C']
    newData_test = pd.DataFrame(newData_test)
    newData_test.columns = ['降维后数值A', '降维后数值B', '降维后数值C', '降维后数值D', '降维后数值E', '降维后数值F', '降维后数值G']
    test = pd.concat([newData_test_class, newData_test], axis=1, ignore_index=True)

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm

    train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=0)
    # 当前参数为默认参数
    m = RandomForestRegressor()
    m.fit(train_data, train_target)
    from sklearn.metrics import r2_score

    score = r2_score(test_target, m.predict(test_data))
    print(score)

    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(train_data, train_target)
    from sklearn.metrics import r2_score

    score = r2_score(test_target, lr.predict(test_data))
    print(score)

    clf = svm.SVC(kernel='poly')
    clf.fit(train_data, train_target)
    score = r2_score(test_target, clf.predict(test_data))
    print(score)

    # 结论就是逻辑回归等模型性能比较差，即使已经经过了正则化、PCA降维、去除多重共线性等。
    # 下面尝试使用一下网格搜索的方式看能否提高一下随机森林的性能。
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline

    param_grid = {'n_estimators': [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200],
                  'max_features': ('auto', 'sqrt', 'log2')}
    m = GridSearchCV(RandomForestRegressor(), param_grid)
    m = m.fit(train_data, train_target.values.ravel())
    print(m.best_score_)
    print(m.best_params_)
    n_estimators = m.best_params_['n_estimators']
    max_features = m.best_params_['max_features']

    # 通过网格搜索找到最佳的参数后，代入模型，模型完成
    m = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features)
    m.fit(train_data, train_target.values.ravel())
    predict = m.predict(test)
    test = pd.read_csv('house_test.csv')['Id']
    sub = pd.DataFrame()
    sub['Id'] = test
    sub['SalePrice'] = pd.Series(predict)
    sub.to_csv('house_pre.csv', index=False)
