import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import string
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data_path = '/Users/seuk/Google 드라이브/Private/01.Data/dacon8/'
data_list = glob.glob(data_path + '*.csv')
for i in data_list:
    if 'train' in i:
        data = pd.read_csv(i, index_col=[0])
    elif 'test' in i:
        test = pd.read_csv(i, index_col=[0])
    else:
        submission = pd.read_csv(i, index_col=[0])

# 결측치 처리
# null로 우선 변환
def make_null(data):
    data.education[data.education == 0] = None
    data.engnat[data.engnat == 0] = None
    data.hand[data.hand == 0] = None
    data.urban[data.urban  == 0] = None
    data.loc[:,'tp01':'tp10'] = data.loc[:,'tp01':'tp10'].applymap(lambda x : None if x==7 else x+1)
    return data

data = make_null(data)
test = make_null(test)

null_list = data.isnull().sum()
print(null_list[null_list > 0])
data.describe() # time관련 및 familysize의 max값들을 보면 outlier존재 확인가능

# 무응답 합
"""
#data['notp_sum'] = data.loc[:,'tp01':'tp10'].isnull().sum(axis=1)
#test['notp_sum'] = test.loc[:,'tp01':'tp10'].isnull().sum(axis=1)
data['noans_sum'] = data.isnull().sum(axis=1)
test['noans_sum'] = test.isnull().sum(axis=1)
"""
# sns.distplot(data.loc[(data.voted==1) & (data.notp_sum >= 1), 'notp_sum'])
# sns.distplot(data.loc[(data.voted==2) & (data.notp_sum >= 1), 'notp_sum'])

# 정답 합(결측치 대치에 미리 사용하면 좋을것같음)
def wrf_sum_fe(data):
    #data['wr_sum'] = data.loc[:,['wr_' + str(i).zfill(2) for i in list(range(1, 14))]].apply(sum,axis=1)
    #data['wf_sum'] = data.loc[:,['wf_' + str(i).zfill(2) for i in list(range(1, 3))]].apply(sum,axis=1)
    data['wrf_sum'] = data.loc[:,['wr_' + str(i).zfill(2) for i in list(range(1, 14))] + ['wf_' + str(i).zfill(2) for i in list(range(1, 4))]].apply(sum,axis=1)
    #sns.boxplot(data=data, x='voted', y='wrf_sum')
    return data

data = wrf_sum_fe(data)
test = wrf_sum_fe(test)

# 결측치 대치(tp01:tp10 제외)
def dropna(data, test):
    # education
    # voted를 제외하면 married가 상관성이 제일 높음
    # 다만 확인해보면 이혼가정이 의미있다는 결과가 나오므로 아니라고 판단.
    # boxplot을 그려봤을 때 wrf_sum이 education별로 확실히 차이가 남을 느낄 수 있음(똑똑할수록 단어의 정의를 많이 안다.)
    # sns.violinplot(data=data.loc[data.education.notnull(), :], x='education', y='wrf_sum')

    # education을 softmax로 작업해야할지 수치형으로 봐야할지 애매함, boxplot을 봤을 때는 확실히 1>2>3>4 순이 보이기 때문
    # 우선은 룰베이스로 작업(나중에 고민하자)
    mean_list = data.loc[data.education.notnull(),['education','wrf_sum']].groupby('education')['wrf_sum'].mean()

    data.loc[data.education.isnull() & (data.wrf_sum < (mean_list.values[0] + mean_list.values[1])/2), 'education'] = 1
    data.loc[data.education.isnull() & (data.wrf_sum < (mean_list.values[1] + mean_list.values[2])/2), 'education'] = 2
    data.loc[data.education.isnull() & (data.wrf_sum < (mean_list.values[2] + mean_list.values[3])/2), 'education'] = 3
    data.loc[data.education.isnull(), 'education'] = 4
    data.education = data.education.astype('int')

    test.loc[test.education.isnull() & (test.wrf_sum < (mean_list.values[0] + mean_list.values[1])/2), 'education'] = 1
    test.loc[test.education.isnull() & (test.wrf_sum < (mean_list.values[1] + mean_list.values[2])/2), 'education'] = 2
    test.loc[test.education.isnull() & (test.wrf_sum < (mean_list.values[2] + mean_list.values[3])/2), 'education'] = 3
    test.loc[test.education.isnull(), 'education'] = 4
    test.education = test.education.astype('int')

    # engnat
    # 나중에 분류 트리 하나 돌려도될듯 wrf_sum간의 차이가 보임
    # sns.violinplot(data=data.loc[data.engnat.notnull(), :], x='engnat', y='wrf_sum')
    mean_list = data.loc[data.education.notnull(), ['engnat', 'wrf_sum']].groupby('engnat')['wrf_sum'].mean().mean()
    data.loc[data.engnat.isnull() & (data.wrf_sum < mean_list), 'engnat'] = 2
    data.loc[data.engnat.isnull(), 'engnat'] = 1
    data.engnat = data.engnat.astype('int')

    test.loc[test.engnat.isnull() & (test.wrf_sum < mean_list), 'engnat'] = 2
    test.loc[test.engnat.isnull(), 'engnat'] = 1
    test.engnat = test.engnat.astype('int')

    # hand
    # 손잡이에 따라 다를만한 것이 없다고 판단,
    # 오른손잡이가 대부분이므로 그냥 오른손으로 변경
    # data.loc[data.hand.notnull(), :].groupby('hand')['wrf_sum'].count()
    data.hand[data.hand.isnull()] = 1
    data.hand = data.hand.astype('int')
    test.hand[test.hand.isnull()] = 1
    test.hand = test.hand.astype('int')

    # urban
    # 1이 많아서 1로 대치
    data.urban[data.urban.isnull()] = 1
    data.urban = data.urban.astype('int')
    test.urban[test.urban.isnull()] = 1
    test.urban = test.urban.astype('int')
    return data, test

data, test = dropna(data, test)

# 결측치 대치(tp01:tp10)
def drop_tpna(data):
    # reverse 성향의 답이 있는 경우
    data.tp01[data.tp01.isnull() & data.tp06.notnull()] = 8 - data.tp06[data.tp01.isnull() & data.tp06.notnull()]
    data.tp02[data.tp02.isnull() & data.tp07.notnull()] = 8 - data.tp07[data.tp02.isnull() & data.tp07.notnull()]
    data.tp03[data.tp03.isnull() & data.tp08.notnull()] = 8 - data.tp08[data.tp03.isnull() & data.tp08.notnull()]
    data.tp04[data.tp04.isnull() & data.tp09.notnull()] = 8 - data.tp09[data.tp04.isnull() & data.tp09.notnull()]
    data.tp05[data.tp05.isnull() & data.tp10.notnull()] = 8 - data.tp10[data.tp05.isnull() & data.tp10.notnull()]
    data.tp06[data.tp06.isnull() & data.tp01.notnull()] = 8 - data.tp01[data.tp06.isnull() & data.tp01.notnull()]
    data.tp07[data.tp07.isnull() & data.tp02.notnull()] = 8 - data.tp02[data.tp07.isnull() & data.tp02.notnull()]
    data.tp08[data.tp08.isnull() & data.tp03.notnull()] = 8 - data.tp03[data.tp08.isnull() & data.tp03.notnull()]
    data.tp09[data.tp09.isnull() & data.tp04.notnull()] = 8 - data.tp04[data.tp09.isnull() & data.tp04.notnull()]
    data.tp10[data.tp10.isnull() & data.tp05.notnull()] = 8 - data.tp05[data.tp10.isnull() & data.tp05.notnull()]

    # 없는 경우
    # 평균으로 대치
    data.loc[:, 'tp01':'tp10'] = data.loc[:, 'tp01':'tp10'].fillna(4)
    return data

data = drop_tpna(data)
test = drop_tpna(test)

# 이상치 제거
def fam_outlier(data, outlier=None, type='train'):
    # data['familysize'].quantile(0.5)
    # IQR 변형하여 대치 quantile(0.99) + 1.5IQR_adj(0.99-0.01로 변경)
    if type == 'train':
        IQR_adj = data['familysize'].quantile(0.99) - data['familysize'].quantile(0.01)
        outlier = data['familysize'].quantile(0.99) + (1.5 * IQR_adj)
    len_outlier = len(data['familysize'][data['familysize'] >= outlier])
    data['familysize'][data['familysize'] >= outlier] = data['familysize'].median()
    print("familysize | {}개 이상치 평균으로 대치".format(len_outlier))
    if type == 'train':
        return data, outlier
    else:
        return data

data, outlier = fam_outlier(data)
test = fam_outlier(test, outlier, type='test')

def outlier_adj(train, test):
    # IQR 변형하여 대치 quantile(0.99) + 1.5IQR_adj(0.99-0.01로 변경)
    # max로 대치
    outlier_list = ['QaE','QbE','QcE','QdE','QeE','QfE','QgE','QhE','QiE','QjE',
                    'QkE','QlE','QmE','QnE','QoE','QpE','QqE','QrE','QsE','QtE']
    for i in range(0, len(outlier_list)):
        IQR_adj = train[outlier_list[i]].quantile(0.99) - train[outlier_list[i]].quantile(0.01)
        outlier = train[outlier_list[i]].quantile(0.99) + (1.5 * IQR_adj)

        # train
        len_outlier_train = len(train[outlier_list[i]][train[outlier_list[i]] >= outlier])
        train[outlier_list[i]][train[outlier_list[i]] >= outlier] = outlier # train[outlier_list[i]].median()

        # test
        len_outlier_test = len(test[outlier_list[i]][test[outlier_list[i]] >= outlier])
        test[outlier_list[i]][test[outlier_list[i]] >= outlier] = outlier # test[outlier_list[i]].median()

        print("{} | train :{}, test: {}개 이상치 평균으로 대치".format(outlier_list[i], len_outlier_train, len_outlier_test))
    return train, test

data, test = outlier_adj(data, test)

# sns.distplot(data.QaE)
# 마키아벨라즘 테스트 피쳐 생성
# 시크릿 처리된 피쳐에 대해 스피어만 상관 계수를 사용하여 상관관계가 높은 방향으로 피쳐 변경
# 스피어만 상관계수를 사용한 이유는 순서형 척도 지표이기 때문에 사용
# 모든 피쳐의 평균을 구함

def machia_fe(data):
    Answers = ['QaA', 'QbA', 'QcA', 'QdA', 'QeA', 'QfA', 'QgA', 'QhA', 'QiA', 'QjA',
                 'QkA', 'QlA', 'QmA', 'QnA', 'QoA', 'QpA', 'QqA', 'QrA', 'QsA', 'QtA']
    flipping_columns = ["QeA", "QfA", "QkA", "QqA", "QrA", "QaA", "QdA", "QgA", "QiA", "QnA"]
    for flip in flipping_columns:
        data[flip] = 6 - data[flip]
    data['machia'] = data[Answers].mean(axis=1)
    return data['machia']

data['machia'] = machia_fe(data)
# sns.distplot(data.loc[data.voted==1,"machia"])
# sns.distplot(data.machia)
test['machia'] = machia_fe(test)

# Q?E는 답변을 하는 데 걸리는 시간

def time_adj_fe(data):
    log_list = ['QaE','QbE','QcE','QdE','QeE','QfE','QgE','QhE','QiE','QjE',
                'QkE','QlE','QmE','QnE','QoE','QpE','QqE','QrE','QsE','QtE']
    data.loc[:,log_list] = np.log(data.loc[:,log_list] + 1e-25)
    data['time_sum'] = data.loc[:,log_list].apply(sum,axis=1)
    #data['time_mean'] = data.loc[:,log_list].mean(axis=1)

    # 3에 대해서만 실행해보자
    for i in range(0, len(log_list)):
        data['Q' + string.ascii_lowercase[i] + 'A_adj1'] = data['Q' + string.ascii_lowercase[i] + 'A'].map(lambda x : 1 if x == 1 else 0) * data['Q' + string.ascii_lowercase[i] + 'E']
        data['Q' + string.ascii_lowercase[i] + 'A_adj2'] = data['Q' + string.ascii_lowercase[i] + 'A'].map(lambda x : 1 if x == 2 else 0) * data['Q' + string.ascii_lowercase[i] + 'E']
        data['Q' + string.ascii_lowercase[i] + 'A_adj3'] = data['Q' + string.ascii_lowercase[i] + 'A'].map(lambda x : 1 if x == 3 else 0) * data['Q' + string.ascii_lowercase[i] + 'E']
        data['Q' + string.ascii_lowercase[i] + 'A_adj4'] = data['Q' + string.ascii_lowercase[i] + 'A'].map(lambda x: 1 if x == 4 else 0) * data['Q' + string.ascii_lowercase[i] + 'E']
        data['Q' + string.ascii_lowercase[i] + 'A_adj5'] = data['Q' + string.ascii_lowercase[i] + 'A'].map(lambda x: 1 if x == 5 else 0) * data['Q' + string.ascii_lowercase[i] + 'E']
    return data

data = time_adj_fe(data)
test = time_adj_fe(test)
print("{}, {}".format(len(data.columns), len(test.columns)))

# tp01:tp10

def tipi_fe(data):
    data['tp06'] = 8 - data['tp06']
    data['tp07'] = 8 - data['tp07']
    data['tp08'] = 8 - data['tp08']
    data['tp09'] = 8 - data['tp09']
    data['tp10'] = 8 - data['tp10']
    #data['tp01'] = data.loc[:,['tp01','tp06']].mean(axis=1)
    #data['tp02'] = data.loc[:,['tp02','tp07']].mean(axis=1)
    #data['tp03'] = data.loc[:,['tp03','tp08']].mean(axis=1)
    #data['tp04'] = data.loc[:,['tp04','tp09']].mean(axis=1)
    #data['tp05'] = data.loc[:,['tp05','tp10']].mean(axis=1)
    #data['tp_mean'] = data.loc[:,['tp01':'tp05']].mean(axis=1)
    #data['tp_2_mean'] = data.loc[:,['tp02','tp07']].mean(axis=1)
    #data['tp_3_mean'] = data.loc[:,['tp03','tp08']].mean(axis=1)
    #data['tp_4_mean'] = data.loc[:,['tp04','tp09']].mean(axis=1)
    #data['tp_5_mean'] = data.loc[:,['tp05','tp10']].mean(axis=1)
    #data = data.drop(['tp' + str(i).zfill(2) for i in list(range(6, 11))], axis=1)
    return data

data = tipi_fe(data)
test = tipi_fe(test)
print("{}, {}".format(len(data.columns), len(test.columns)))

"""
sns.heatmap(data.loc[:, 'tp01':'tp05'].corr().abs())
sns.heatmap(data.loc[data.voted==1,['tp01','tp06']].corr().abs()) #
sns.heatmap(data.loc[:,['tp02','tp07']].corr().abs()) #
sns.distplot(data.loc[:,"tp01"])
sns.violinplot(data=data, y='tp01', x='voted')
sns.distplot(data.loc[data.voted==1,"tp03"])
sns.distplot(data.loc[data.voted==2,"tp03"])
"""

# 범주형 변수 체크
# sns.countplot(data=data, x='education', hue='voted') # 1 / 2 / 3,4 로 나눠야 할 듯
# sns.countplot(data=data, x='familysize', hue='voted')
# sns.boxplot(data=data, x='age_group', y='QaE', hue='voted') # 10대와 나머지로 나눠야 할듯
# sns.countplot(data=data, x='engnat', hue='voted')
# sns.countplot(data=data, x='gender', hue='voted')
# sns.countplot(data=data, x='hand', hue='voted')
# sns.countplot(data=data, x='married', hue='voted') # 1과 나머지로 나눠야 할 듯
# sns.countplot(data=data, x='race', hue='voted') # White와 나머지로 나눠야 할 듯
# sns.countplot(data=data, x='religion', hue='voted') # Christian_Protestant, Jewish와 나머지로 나눠야 할 듯
# sns.countplot(data=data, x='urban', hue='voted') # 1과 2,3으로 나눠야 할 듯
# sns.countplot(data=data, x='wr_01', hue='voted')
# sns.countplot(data=data, x='wr_03', hue='voted')
# sns.countplot(data=data, x='wr_06', hue='voted')
# sns.countplot(data=data, x='wr_09', hue='voted')
# sns.countplot(data=data, x='wr_11', hue='voted')
# sns.countplot(data=data, x='wf_03', hue='voted')
# 유의미한 차이가 나 보이는 0~3을 그룹핑
"""
def education_func(x):
    if x == 1:
        x = 1
    elif x == 2:
        x = 2
    else:
        x = 3
    return x

def group_fe(data):
    data['age_group'] = data['age_group'].map(lambda x : 1 if x == '10s' else 'else')
    data['married'] = data['married'].map(lambda x: 1 if x == 2 else 0)
    #data['religion'] = data['religion'].map(lambda x: 1 if x == ('Christian_Protestant') or (x == 'Jewish') else 0)
    data['race'] = data['race'].map(lambda x: 1 if x == ('White') or (x == 'Asian') else 0)
    #data['engnat'] = data['engnat'].map(lambda x: 1 if x == 2 else 0)
    return data

data = group_fe(data)
test = group_fe(test)
"""
# 범주형 변수 선택
cat_feature = ['age_group', 'education', 'married', 'religion', 'race']
cat_feature_all = ['age_group', 'education', 'married', 'religion', 'race', 'wr_01', 'wr_03', 'wr_06', 'wr_09', 'wr_11', 'wf_03']


# 전체합 / 전체시간 컬럼 추가
data['machia_time'] = data.machia / data.time_sum
test['machia_time'] = test.machia / test.time_sum
print("{}, {}".format(len(data.columns), len(test.columns)))
# sns.distplot(data.loc[data.voted==1, 'machia_time'])
# sns.distplot(data.loc[data.voted==2, 'machia_time'])

# scaling
num_col = list(data.loc[:, 'QaA':'QtE'].columns) + ['tp' + str(i).zfill(2) for i in list(range(1, 11))] + list(data.loc[:, 'wrf_sum':'time_sum'].columns) + ['machia_time']
scaler = RobustScaler()
data.loc[:,num_col] = scaler.fit_transform(data.loc[:,num_col])
test.loc[:, num_col] = scaler.transform(test.loc[:, num_col])

num_col2 = list(data.loc[:, 'QaA_adj1':'QtA_adj5'].columns)
scaler = StandardScaler()
data.loc[:,num_col2] = scaler.fit_transform(data.loc[:,num_col2])
test.loc[:, num_col2] = scaler.transform(test.loc[:, num_col2])
print("{}, {}".format(len(data.columns), len(test.columns)))

# Q?A 선제거하자
# 모델 돌려보니 성능 제일 안좋은 변수임
#drop_list = ['QaA', 'QbA', 'QcA', 'QdA', 'QeA', 'QfA', 'QgA', 'QhA', 'QiA', 'QjA',
#           'QkA', 'QlA', 'QmA', 'QnA', 'QoA', 'QpA', 'QqA', 'QrA', 'QsA', 'QtA']
#data = data.drop(drop_list, axis=1)
#test = test.drop(drop_list, axis=1)
drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE', 'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
            'QkE', 'QlE', 'QmE', 'QnE', 'QoE', 'QpE', 'QqE', 'QrE', 'QsE', 'QtE']
data = data.drop(drop_list, axis=1)
test = test.drop(drop_list, axis=1)
data = data.drop(data.loc[:, 'QaA_adj1':'QtA_adj5'].columns, axis=1)
test = test.drop(test.loc[:, 'QaA_adj1':'QtA_adj5'].columns, axis=1)
#drop_list = data.loc[:,'wf_01':'wr_13'].columns
#data = data.drop(drop_list, axis=1)
#test = test.drop(drop_list, axis=1)
#data = data.drop(drop_list, axis=1)
#test = test.drop(drop_list, axis=1)
# + list(data.loc[:, 'tp01':'tp05'].columns) + list(data.loc[:, 'notp_sum':'machia_time'].columns)
# data = data.loc[:, cat_feature + num_feature + ['voted']]
# test = test.loc[:, cat_feature + num_feature]

"""
# wrf_sum과 곱한 one-hot 실험
def mul_wrfsum(data, cat_feature):
    dd = pd.get_dummies(data=pd.get_dummies(data.loc[:,cat_feature]), columns = ['education', 'married'])
    for i in range(0, len(dd.columns)):
        dd.iloc[:,i] = dd.iloc[:,i] * data['wrf_sum']
    lendd = len(dd.columns)
    coldd = ['dd' + str(i).zfill(2) for i in range(0, lendd)]
    dd.columns = coldd
    return pd.concat([data, dd], axis=1)

data = mul_wrfsum(data, cat_feature_all)
test = mul_wrfsum(test, cat_feature_all)
print("{}, {}".format(len(data.columns), len(test.columns)))
"""

# encoding
data = pd.get_dummies(data, columns = cat_feature)
test = pd.get_dummies(test, columns = cat_feature)

# 전체로 할 땐 인코딩 추가
data = pd.get_dummies(data, columns = ['engnat','gender','hand'])
test = pd.get_dummies(test, columns = ['engnat','gender','hand'])
data = data.drop(['machia'], axis=1)
test = test.drop(['machia'], axis=1)

# 7780 / 3603 : age_group_10s
# 10063 / 1320 : education_1

if True:
    x_tr, x_te, y_tr, y_te = \
        train_test_split(data.drop('voted', axis=1), data['voted'], test_size=0.2, random_state=20) # 43


    from catboost import CatBoostClassifier
    model = CatBoostClassifier(n_estimators = 3000, learning_rate = 0.02, early_stopping_rounds = 500,
                               eval_metric = 'AUC', reg_lambda=0.3, random_state=143, subsample=0.5, colsample_bylevel=0.8)
    model.fit(x_tr, y_tr, use_best_model=True, eval_set=[(x_tr, y_tr), (x_te, y_te)], plot=False, silent=False)
    model.best_score_['validation_1']['AUC']
    aa = pd.DataFrame({'imp' : model.get_feature_importance(), 'name' : x_tr.columns}).sort_values('imp', ascending=False)

    #list_b = list(aa.name[0:-15])
    #data = data.loc[:, list_b + ['voted']]
    #test = test.loc[:, list_b]
    #    x_tr, x_te, y_tr, y_te = \
    #    train_test_split(data.drop('voted', axis=1), data['voted'], test_size=0.2, random_state=143) # 43

    #
    submission['voted'] = model.predict_proba(test)[:, 1]
    submission.to_csv('sub_proba2.csv')

    model2 = LGBMClassifier(boosting_type='gbdt', n_estimators=3000, learning_rate = 0.02, random_state=143,
                            reg_lambda=0.3, colsample_bytree=0.5, subsample=0.5, metric='AUC')
    model2.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_te, y_te)], early_stopping_rounds= 1000, eval_metric='AUC')
    submission['voted'] = model2.predict_proba(test)[:, 1]
    submission.to_csv('sub_proba3.csv')

    submission['voted'] = model.predict_proba(test)[:, 1]/2 + model2.predict_proba(test)[:, 1]/2
    submission.to_csv('sub_proba4.csv')

if False:
    from pycaret.classification import *
    clf = setup(data = data, target='voted', categorical_features=data.loc[:,'age_group_+70s':'hand_3'].columns)
    best_3 = compare_models(sort='AUC', n_select=3)
    blended = blend_models(estimator_list=best_3, fold=5, method='soft')
    pred_holdout = predict_model(blended)
    final_model = finalize_model(blended)

    predictions = predict_model(final_model, data=test)
    submission['voted'] = predictions['Score']

    submission.to_csv('sub_proba.csv')
