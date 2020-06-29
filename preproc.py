##############################################################################
##############################################################################
# 월간 데이콘 5 | dyddl1993@naver.com
#
# 제출점수 MAE 1.07, 최종 66등인데
#
# 최종제출 등록을 안하면 맨 첫번째 스코어가 제출되버려서 200등까지 떨어짐. ㅜㅜ
# 시각화 위주로 공부했고, 처음으로 catboost를 사용해보았고, 마지막에 시간이 없어서
# 적용은 못했지만 베이지안 옵티마이저도 공부할 수 있는 기회였음.
##############################################################################
##############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings ; warnings.filterwarnings('ignore')
import time
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

##############################################################################
##############################################################################
# 전처리
##############################################################################
##############################################################################

data_path = 'C:\\Users\\waikorea\\PycharmProjects\\untitled\\data\\dacon5\\'
#data_path = 'C:\\Users\\SEUK\\PycharmProjects\\root\\data\\dacon5\\'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')

src_list = list(map(lambda x : str(x) + '_src',list(range(650,1000,10))))
dst_list = list(map(lambda x : str(x) + '_dst',list(range(650,1000,10))))

train_dst = train.loc[:,dst_list]
test_dst = test.loc[:,dst_list]

# 선형보간법 사용, 보간해도 남은 데이터는 추가 전처리
for i in tqdm(train_dst.index):
    train_dst.loc[i] = train_dst.loc[i].interpolate()

for i in tqdm(test_dst.index):
    test_dst.loc[i] = test_dst.loc[i].interpolate()

# 단순대치, 회귀를 통해 대치하는것보다 효율이 좋음....
if True:
    train_dst.loc[train_dst['700_dst'].isnull(),'700_dst']=train_dst.loc[train_dst['700_dst'].isnull(),'710_dst']
    train_dst.loc[train_dst['690_dst'].isnull(),'690_dst']=train_dst.loc[train_dst['690_dst'].isnull(),'700_dst']
    train_dst.loc[train_dst['680_dst'].isnull(),'680_dst']=train_dst.loc[train_dst['680_dst'].isnull(),'690_dst']
    train_dst.loc[train_dst['670_dst'].isnull(),'670_dst']=train_dst.loc[train_dst['670_dst'].isnull(),'680_dst']
    train_dst.loc[train_dst['660_dst'].isnull(),'660_dst']=train_dst.loc[train_dst['660_dst'].isnull(),'670_dst']
    train_dst.loc[train_dst['650_dst'].isnull(),'650_dst']=train_dst.loc[train_dst['650_dst'].isnull(),'660_dst']

    test_dst.loc[test_dst['700_dst'].isnull(),'700_dst']=test_dst.loc[test_dst['700_dst'].isnull(),'710_dst']
    test_dst.loc[test_dst['690_dst'].isnull(),'690_dst']=test_dst.loc[test_dst['690_dst'].isnull(),'700_dst']
    test_dst.loc[test_dst['680_dst'].isnull(),'680_dst']=test_dst.loc[test_dst['680_dst'].isnull(),'690_dst']
    test_dst.loc[test_dst['670_dst'].isnull(),'670_dst']=test_dst.loc[test_dst['670_dst'].isnull(),'680_dst']
    test_dst.loc[test_dst['660_dst'].isnull(),'660_dst']=test_dst.loc[test_dst['660_dst'].isnull(),'670_dst']
    test_dst.loc[test_dst['650_dst'].isnull(),'650_dst']=test_dst.loc[test_dst['650_dst'].isnull(),'660_dst']

    train.loc[:,dst_list] = train_dst
    test.loc[:,dst_list] = test_dst

    train.to_csv('train2.csv', index=False)
    test.to_csv('test2.csv', index=False)

##############################################################################
##############################################################################
# EDA
##############################################################################
##############################################################################

train = pd.read_csv('train2.csv')
test = pd.read_csv('test2.csv')

src_list = list(map(lambda x : str(x) + '_src',list(range(650,1000,10))))
dst_list = list(map(lambda x : str(x) + '_dst',list(range(650,1000,10))))
src_log_list = list(map(lambda x : str(x) + '_src_log',list(range(650,1000,10))))
dst_log_list = list(map(lambda x : str(x) + '_dst_log',list(range(650,1000,10))))
hup_list = list(map(lambda x : str(x) + '_hup',list(range(650,1000,10))))
ratio_list = list(map(lambda x : str(x) + '_ratio',list(range(650,1000,10))))
src_fft_list = list(map(lambda x : str(x) + '_src_fft',list(range(650,1000,10))))
dst_fft_list = list(map(lambda x : str(x) + '_dst_fft',list(range(650,1000,10))))

if True:

    def log_append(train, dst_list, dst_log_list):
        dst_log = (-np.log(train.loc[:, dst_list] + 1e-30)).replace(float('-inf'),np.NaN).replace(float('inf'),np.NaN).replace(float('inf'),np.NaN).astype('float')
        dst_log.columns = dst_log_list
        train = pd.concat([train, dst_log], axis=1)
        return train

    train = log_append(train, dst_list, dst_log_list)
    test = log_append(test, dst_list, dst_log_list)

    def hup_append(train, src_list, dst_list, hup_list):
        gg_data = pd.DataFrame(np.array(train.loc[:,dst_list])/(np.array(train.loc[:,src_list])),
                               columns = hup_list)
        train = pd.concat([train, gg_data], axis=1)
        train.loc[:,hup_list] = (-np.log(train.loc[:,hup_list])).replace(float('-inf'),np.NaN).replace(float('inf'),np.NaN).astype('float')
        return train

    train = hup_append(train, src_list, dst_list, hup_list)
    test = hup_append(test, src_list, dst_list, hup_list)

    def ratio_append(train, src_list, dst_list, ratio_list):
        gg_data = pd.DataFrame(np.array(train.loc[:,dst_list])/(np.array(train.loc[:,src_list] + 1e-10)),
                               columns = ratio_list)
        train = pd.concat([train, gg_data], axis=1)
        return train

    train = ratio_append(train, src_list, dst_list, ratio_list)
    test = ratio_append(test, src_list, dst_list, ratio_list)

    def make_max(data, data_list):
        l1 = []
        data2 = data.loc[:, data_list]
        for i in range(0, len(data2)):
            l1.append(data2.columns[data2.iloc[i, :] == data2.iloc[i, :].max()][0].split("_")[0])
        return l1

    train['max_hup'] = make_max(train, hup_list)
    test['max_hup'] = make_max(test, hup_list)

    train['max_dst'] = make_max(train, dst_list)
    test['max_dst'] = make_max(test, dst_list)

    def make_max2(data, data_list, colname = 'max_hup_val'):
        data[colname] = data.loc[:,data_list].apply(max, axis=1)
        return data

    train = make_max2(train, hup_list)
    test = make_max2(test, hup_list)

    train = make_max2(train, dst_list, colname='max_dst_val')
    test = make_max2(test, dst_list, colname='max_dst_val')

    def fft_adj(train, src_list, dst_list):

        fs = 5
        # sampling frequency
        fmax = 25
        # sampling period
        dt = 1 / fs
        # length of signal
        N = 75

        df = fmax / N
        f = np.arange(0, N) * df

        signals1 = []
        signals2 = []
        for i in tqdm(train.id):
            xf1 = np.fft.fft(train.loc[train.id == i, src_list].values) * dt
            xf2 = np.fft.fft(train.loc[train.id == i, dst_list].values) * dt

            signals1.append(np.concatenate([np.abs(xf1[0:int(N / 2 + 1)])]))
            signals2.append(np.concatenate([np.abs(xf2[0:int(N / 2 + 1)])]))

        s1 = pd.DataFrame({})
        s2 = pd.DataFrame({})
        for i in range(0, len(signals1)):
            s1 = pd.concat([s1, pd.DataFrame(signals1[i], columns=list(map(lambda x : str(x) + '_src_fft',list(range(650,1000,10)))))], axis=0)
            s2 = pd.concat([s2, pd.DataFrame(signals2[i], columns=list(map(lambda x : str(x) + '_dst_fft',list(range(650,1000,10)))))], axis=0)

        s1 = s1.reset_index(drop=True)
        s2 = s2.reset_index(drop=True)

        train = pd.concat([train, s1], axis=1)
        train = pd.concat([train, s2], axis=1)

        return train

    train = fft_adj(train, src_list, dst_list)
    test = fft_adj(test, src_list, dst_list)

    def count_fft(train, dst_fft_list):
        l1 = []
        for i in train.index:
            count = ((train.loc[i,dst_fft_list] - train.loc[i,dst_fft_list].shift(1) > 0) \
             & (train.loc[i,dst_fft_list] - train.loc[i,dst_fft_list].shift(-1) > 0))[1:-1].sum()

            if train.loc[i,'650_dst_fft'] > train.loc[i,'660_dst_fft']:
                count += 1

            if train.loc[i,'990_dst_fft'] > train.loc[i,'980_dst_fft']:
                count += 1

            l1.append(count)
        return l1

    train['count_fft'] = count_fft(train, dst_fft_list)
    test['count_fft'] = count_fft(test, dst_fft_list)

    train['count_dst'] = count_fft(train, dst_list)
    test['count_dst'] = count_fft(test, dst_list)

    train['max_dst'] = train['max_dst'].astype('int')
    test['max_dst'] = test['max_dst'].astype('int')

    def make_gap(train, src_list, dst_list):
        train_gap = np.array(train.loc[:, src_list]) - np.array(train.loc[:, dst_list])
        train_gap = pd.DataFrame(data=train_gap, columns=list(map(lambda x : str(x) + '_gap',list(range(650,1000,10)))))
        train = pd.concat([train, train_gap], axis=1)
        return train

    train = make_gap(train, src_list, dst_list)
    test = make_gap(test, src_list, dst_list)

    #
    def real_imag(train, dst_list):
        alpha_real = train[dst_list]
        alpha_imag = train[dst_list]

        for i in tqdm(alpha_real.index):
            alpha_real.loc[i] = alpha_real.loc[i] - alpha_real.loc[i].mean()
            alpha_imag.loc[i] = alpha_imag.loc[i] - alpha_real.loc[i].mean()

            alpha_real.loc[i] = np.fft.fft(alpha_real.loc[i], norm='ortho').real
            alpha_imag.loc[i] = np.fft.fft(alpha_imag.loc[i], norm='ortho').imag

        fft_real = list(map(lambda x: str(x) + '_fft_real', list(range(650, 1000, 10))))
        fft_imag = list(map(lambda x: str(x) + '_fft_imag', list(range(650, 1000, 10))))
        alpha_real.columns = fft_real
        alpha_imag.columns = fft_imag
        alpha = pd.concat([alpha_real, alpha_imag], axis=1)
        train = pd.concat([train, alpha], axis=1)
        return train

    train = real_imag(train, dst_list)
    test = real_imag(test, dst_list)

    #
    def bogan(data_list, train, test):
        train_dst = train.loc[:, data_list]
        test_dst = test.loc[:, data_list]

        # 선형보간법 사용, 보간해도 남은 데이터는 추가 전처리
        for i in tqdm(train_dst.index):
            train_dst.loc[i] = train_dst.loc[i].interpolate()

        for i in tqdm(test_dst.index):
            test_dst.loc[i] = test_dst.loc[i].interpolate()
        return train_dst, test_dst

    train_dst_log, test_dst_log = bogan(dst_log_list, train, test)
    train_hup, test_hup = bogan(hup_list, train, test)

    train.loc[:, dst_log_list] = train_dst_log
    test.loc[:, dst_log_list] = test_dst_log
    train.loc[:, hup_list] = train_hup
    test.loc[:, hup_list] = test_hup

    #
    x_train = train.drop(['id','hhb','hbo2','ca','na','count_fft'], axis=1)
    x_train = x_train.loc[:, '650_dst_log':]
    x_train['rho'] = train.rho
    x_test = test.drop(['id','count_fft'], axis=1)
    x_test = x_test.loc[:, '650_dst_log':]
    x_test['rho'] = test.rho
    y_train = train.loc[:,'hhb':'na']
    y_train['all'] = y_train.apply(sum, axis=1)

# 시각화
# rho : 범주형 자료로 사용하자

# log : rho별로 데이터의 크기가 너무 달라서 스케일링하기가 아려움. 그나마 할만한 로그 스케일링 수행
# 널값 대치가 문제
sns.scatterplot(data=train, x='800_dst_log', y='hhb', hue='rho')
train.loc[0, dst_log_list].T.plot()

# hup : 흡광도를 표현 -log(반사빛/비춘빛)
# 약간의 헛도는 값을 제외하면 일단 rho별로 나눠짐 na에는 효과가 없어 보임
# NaN 값을 일단 나뒀음 어떻게 처리할지가 문제
sns.scatterplot(data=train, x='800_hup', y='hhb', hue='rho')
sns.scatterplot(data=train, x='800_hup', y='ca', hue='rho')
train.loc[1, ratio_list].T.plot()

# max : 각 스펙트럼들을 보면 최대값이 하나 존재함. 효과가 있을 것으로 생각하여 최대값이 존재하는 컬럼명를 뽑음
# 값별로 차이가 유의미하게 난다고 판단.
# max_dst별로는 차이가 많이 나고, ratio는 중간 컬럼만 차이가 존재
train.loc[0, dst_list].T.plot()
sns.boxplot(data=train, x='max_dst', y='hhb', hue='rho')
sns.boxplot(data=train, x='max_ratio', y='hhb', hue='rho')

# max_val : 최대값이 어떤 컬럼인지에 따라 차이가 발생하므로 그 실제값도 추가함
sns.scatterplot(data=train, x='max_ratio_val', y='hhb', hue='rho')

# fft 관련 : 푸리에 변환 식을 적용하면 효과가 있을까 싶어 추가

# count 관련 : 관련 자료를 읽어보던 중 진동의 수가 영향이 있다는 글을 읽어서 추가함
# 푸리에 변환한 값과 기존 측정 스펙트럼의 count를 세봄
# count_fft는 별로이고, dst도 애매하지만 fft보단 나아보임
sns.scatterplot(data=train, x='count_fft', y='hhb', hue='rho')
sns.scatterplot(data=train, x='count_dst', y='hhb', hue='rho')
sns.boxplot(data=train, x='count_fft', y='hhb', hue='rho')
sns.boxplot(data=train, x='count_dst', y='hhb', hue='rho')

##############################################################################
##############################################################################
# 모델링
##############################################################################
##############################################################################

# y값에 all을 추가해서 ca를 대체할 수 있을까 했으니 의미없었음

# rho 카테고리화
if False:

    model = CatBoostRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        early_stopping_rounds=100,
        eval_metric='MAE',
        l2_leaf_reg=100
        )
        #,cat_features=['rho','max_dst'])

    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                random_state=0)
    a1 = []
    a2 = []
    for i in range(0, 5):
        model.fit(x_tr, y_tr.iloc[:, i], use_best_model=True, eval_set=(x_val, y_val.iloc[:, i]), plot=False, silent=True)
        a1.append(list(model.predict(x_test)))
        a2.append([model.best_score_['validation']['MAE'], model.best_iteration_])

        f_imp = pd.DataFrame({'feature': x_tr.columns, 'fe': model.feature_importances_})
        f_imp = f_imp.sort_values('fe', ascending=False)
        print(f_imp.head(3))

    a1 = pd.DataFrame(a1).T
    a1.columns = ['hhb', 'hbo2', 'ca', 'na', 'all']
    a2 = pd.DataFrame(a2).T
    a2.columns = ['hhb', 'hbo2', 'ca', 'na', 'all']
    a2.index = ['MAE', 'Best_iter']
    print(a2)

    #a1['id'] = submission['id']
    #a1[['id', 'hhb', 'hbo2', 'ca', 'na']].to_csv("sub4.csv", index=False)

    print('최종 점수 : {}'.format(a2.loc['MAE',:][:-1].mean()))

    #a1['ca'] = a1['all'] - a1['hhb'] - a1['hbo2'] - a1['na']
    #a1[['id', 'hhb', 'hbo2', 'ca', 'na']].to_csv("sub4_2.csv", index=False)

# rho가 상당히 유의미한 변수인 만큼 rho별로 모델링을 해 봄
if True:

    model = CatBoostRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        early_stopping_rounds=100,
        eval_metric='MAE',
        l2_leaf_reg=100
    )
    # ,cat_features=['rho','max_dst'])

    a1_df = pd.DataFrame({})
    a2_df = pd.DataFrame({})
    for j in [10, 15, 20, 25]:
        train_x = x_train.loc[x_train.rho == j, :]
        train_y = y_train.loc[x_train.rho == j, :]
        test_x = x_test.loc[x_test.rho == j, :]

        """    if j == 15:
            train_x = x_train.loc[x_train.rho != 10, :]
            train_y = y_train.loc[x_train.rho != 10, :]
            test_x = x_test.loc[x_test.rho != 10, :]"""

        x_tr, x_val, y_tr, y_val = train_test_split(train_x, train_y, test_size=0.2,
                                                    random_state=0)
        a1 = []
        a2 = []
        for i in range(0, 5):
            model.fit(x_tr, y_tr.iloc[:, i], use_best_model=True, eval_set=(x_val, y_val.iloc[:, i]), plot=False,
                      silent=True)
            a1.append(list(model.predict(test_x)))
            a2.append([model.best_score_['validation']['MAE'], model.best_iteration_])

            f_imp = pd.DataFrame({'feature': x_tr.columns, 'fe': model.feature_importances_})
            f_imp = f_imp.sort_values('fe', ascending=False)
            print(f_imp.head(3))

        a1 = pd.DataFrame(a1).T
        a1.columns = ['hhb', 'hbo2', 'ca', 'na', 'all']
        a1.index = test_x.index
        a2 = pd.DataFrame(a2).T
        a2.columns = ['hhb', 'hbo2', 'ca', 'na', 'all']
        a2.index = ['MAE', 'Best_iter']
        print(a2)

        a1_df = pd.concat([a1_df, a1], axis=0).sort_index()
        a2_df = pd.concat([a2_df, a2], axis=0)

    # a1_df['id'] = submission['id']
    # a1_df[['id', 'hhb', 'hbo2', 'ca', 'na']].to_csv("sub4.csv", index=False)

    print(a2_df.loc['MAE', :].mean())
    print('최종 점수 : {}'.format(a2_df.loc['MAE', :][:-1].mean().mean()))

    # a1_df['ca'] = a1_df['all'] - a1_df['hhb'] - a1_df['hbo2'] - a1_df['na']
    # a1_df[['id', 'hhb', 'hbo2', 'ca', 'na']].to_csv("sub4_2.csv", index=False)

##############################################################################
##############################################################################
# AutoML 연습
##############################################################################
##############################################################################

if False:
    from bayes_opt import BayesianOptimization
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    import math
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
    from sklearn.metrics import mean_absolute_error
    from lightgbm import LGBMRegressor
    import warnings;

    warnings.filterwarnings('ignore')


    def lgb_evaluate(numLeaves, maxDepth, subsample, colSam, learning_rate, n_estimators, reg_alpha):
        reg = LGBMRegressor(num_leaves=int(numLeaves), max_depth=int(maxDepth),
                            subsample=subsample, colsample_bytree=colSam,
                            learning_rate=learning_rate, n_estimators=int(n_estimators),
                            reg_alpha=int(reg_alpha), boosting_type='gbdt')
        scores = cross_val_score(reg, train_x, train_y, cv=5, scoring='neg_mean_absolute_error')
        return np.mean(scores)


    def bayesOpt(train_x, train_y):
        lgbBO = BayesianOptimization(lgb_evaluate, {'numLeaves': (3, 10), 'maxDepth': (2, 10),
                                                    'subsample': (0.4, 1), 'colSam': (0.4, 1),
                                                    'learning_rate': (0.01, 0.1), 'n_estimators': (500, 1500),
                                                    'reg_alpha': (3, 10)})
        lgbBO.maximize(init_points=5, n_iter=50)
        score = lgbBO.max['target']
        params = lgbBO.max['params']
        return params, score


    pred_final = pd.DataFrame({})
    pred_list = pd.DataFrame({})
    score_list = []

    for i in range(0, 5):
        params, score = bayesOpt(train_x=x_train, train_y=y_train.loc[:, i])
        print("{}, rho = {} model, score is {}".format(i, j, score))

        score_list.append([i, j, score])

        globals()['param_{}_{}'.format(i, j)] = params

        model = LGBMRegressor(num_leaves=int(params['numLeaves']),
                              max_depth=int(params['maxDepth']),
                              # scale_pos_weight=params['scaleWeight'],
                              # min_child_weight=params['minChildWeight'],
                              subsample=params['subsample'],
                              colsample_bytree=params['colSam'],
                              learning_rate=params['subsample'],
                              n_estimators=int(params['n_estimators']),
                              reg_alpha=int(params['reg_alpha']),
                              boosting_type='gbdt')
        model.fit()
        pred = pd.DataFrame(model.predict(test_x), index=test_x.index, columns=[i])
        pred_list = pd.concat([pred_list, pred], axis=0).sort_index()

    pred_final = pd.concat([pred_final, pred_list], axis=1)
    pred_list = pd.DataFrame({})

    pred_final['id'] = submission['id']
    # pred_final[['id', 'hhb', 'hbo2', 'ca', 'na']].to_csv(data_path + "lgb_sub.csv", index=False)