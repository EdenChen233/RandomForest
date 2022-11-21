import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息

# 数据处理清洗包
import pandas as pd
import numpy as np

# 可视化包
import seaborn as sns
import matplotlib.pyplot as plt

# 机器学习算法相关包
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz
import pydotplus

# 数据导入
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]  # 合并数据
test_df.isnull().sum()

# 针对Pclass和Survived进行分类汇总
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# pd.DataFrame(train_df.groupby('Pclass', as_index=False)['Survived'].mean()).sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')  # FacetGrid(data, row, col, hue, height, aspect, palette, ...)
g.map(plt.hist, 'Age', bins=20)

# 删除无用特征
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# 使用正则表达式提取Title特征
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex']).sort_values(by='female', ascending=False)  # pd.crosstab列联表

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# 将分类标题转换为序数
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)  # 将序列中的每一个元素，输入函数，最后将映射后的每个值返回合并，得到一个迭代器
    dataset['Title'] = dataset['Title'].fillna(0)

# 现在可以从训练和测试数据集中删除Name特征以及训练集中的PassengerId 特征
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# 转换分类特征Sex
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)  # 男性赋值为0，女性赋值为1，并转换为整型数据

# 创建空数组
guess_ages = np.zeros((2, 3))

# 遍历 Sex (0 或 1) 和 Pclass (1, 2, 3) 来计算六种组合的 Age 猜测值
for dataset in combine:
    # 第一个for循环计算每一个分组的Age预测值
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # 将随机年龄浮点数转换为最接近的 0.5 年龄（四舍五入）
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    # 第二个for循环对空值进行赋值
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# 创建年龄段,并确定其与Survived的相关性
# 一般在建立分类模型时，需要对连续变量离散化，特征离散化后，模型会更稳定，降低了模型过拟合的风险
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)  # 将年龄分割为5段,等距分箱
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# 将这些年龄区间替换为序数
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)  # 删除训练集中的AgeBand特征
combine = [train_df, test_df]

# 创建一个新特征FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# 创建新特征IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# 舍弃这些特征，因为isalone更能反应与survived的相关性
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# 创建Age*Pclass特征以此用来结合Age和Pclass变量
for dataset in combine:
    dataset['Age*Pclass'] = dataset.Age * dataset.Pclass

# 填补分类特征embarked
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# 同样转换分类特征为序数
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 测试集中Fare有一个缺失值，用中位数进行填补
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)  # 根据样本分位数进行分箱，等频分箱
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# 准备训练模型
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

random_forest = RandomForestClassifier(n_estimators=100)
# 训练随机森林
random_forest.fit(X_train, Y_train)
# 使用随机森林进行预测
Y_pred = random_forest.predict(X_test)
# 计算模型的置信度得分
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})
submission.to_csv('./predict.csv', index=False)
# print(acc_random_forest)
# 可视化随机森林中的决策树
# m=0
# for per_estimator in random_forest.estimators_:
#     dot_data = tree.export_graphviz(per_estimator, out_file=None,
#                              feature_names=X_train.columns,
#                              class_names=['0','1'],
#                              filled=True, rounded=True,
#                              special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     m=m+1
#     graph.write_pdf("./TreeGraph/"+(str(m)+"DTtree.pdf"))

# 逻辑回归模型
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)  # logreg.predict_proba(X_test)[:,1]
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# 朴素贝叶斯分类器
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

models = pd.DataFrame({
    'Model': ['Logistic Regression',
              'Random Forest', 'Naive Bayes'],
    'Score': [acc_log,
              acc_random_forest,
              acc_gaussian]})
result = models.sort_values(by='Score', ascending=False)
print(result)
