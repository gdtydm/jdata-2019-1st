
import lightgbm as lgb
import pandas as pd
from datetime import datetime, date, timedelta
from preprocess import reduce_mem_usage
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pickle
from tools import reduce_mem_usage
import os
import gc
from cate_base_feature import extract_cate_base_feature
from user_base_feature import extract_user_base_feature
from user_action_feature import extract_user_action_feature
from user_cate_action_feature import extract_user_cate_feature
from config import *
from sklearn.metrics import auc, log_loss, roc_auc_score


"""
train_times， test_times, dev_times：list: list 目标区间开始日期
befor_days: 以目标区间前多少天构建训练数据集
candidate_set："以构建区间所有出现过的 user_id + cate 组合构建数据集" if ”all“ else ”以购买过的 user_id + cate 组合构建训练集" 就f11而言 all 效果更好。
windows: 特征提取窗口
"""

def create_date(target_days:list, schema, windows=[1,3,7,14,30], befor_days=None, candidate_set="all"):
    if not befor_days:
        befor_days = 39
    load_action_table = False
    datas = []
    labels = []
    for target_day in target_days:
        print(f"load {str(target_day)} data...")
        if os.path.exists(cache_path + f"{str(target_day)}_{befor_days}_{candidate_set}.pickle"):
            with open(cache_path + f"{str(target_day)}_{befor_days}_{candidate_set}.pickle", "rb") as f:
                user_cate,label_area = pickle.load(f)
        else:
            if not load_action_table:
                action_table = pd.read_hdf(base_file_path + "jdata_action.h5")
                action_table['action_time'] = action_table['action_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
                product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
                action_table = pd.merge(action_table, product_table, how="left", on="sku_id")
                action_table = action_table[['user_id', 'cate', "action_time", "type"]]
                del product_table
                gc.collect()
                load_action_table = True

            user_cate = action_table[(action_table['action_time'] < target_day) & (action_table['action_time'] >= (target_day - timedelta(befor_days)))]
            
            if candidate_set == "all":
                user_cate = user_cate[['user_id', "cate"]].drop_duplicates() # 购买过的组合作为候选集
            else:
                user_cate = user_cate[user_cate['type'] == 2][['user_id', "cate"]].drop_duplicates() # 购买过的组合作为候选集
            user_cate.dropna(inplace=True)
            user_cate_len = len(user_cate)

            if schema == "train":
                target_end = target_day + timedelta(6)
                label_area = action_table[(action_table['action_time'] >= target_day)&(action_table['action_time']<=target_end)]
                label_area = label_area[label_area['type']==2][['user_id', "cate"]].drop_duplicates() # 目标区间购买作为测试集
                label_area['label'] = 1
                label_area.dropna(inplace=True)
                user_cate = pd.merge(user_cate, label_area, how="left", on=['user_id', "cate"])
                user_cate.fillna(0, inplace=True)
                label_area['time_area'] = str(target_day)
                print(f"create target: {str(target_day)} to {str(target_end)}")
                print(user_cate['label'].value_counts())
            else:
                label_area = pd.DataFrame()
                user_cate['label'] = 0
            print(f"samples :{len(user_cate)}")
            user_cate['time_area'] = str(target_day)
            print("load base user feature...")
            base_user_feature_path = feature_path + f"base_user_feature_{str(target_day)}.h5"
            if not os.path.exists(base_user_feature_path):
                extract_user_base_feature(target_day)
            base_user_feature = pd.read_hdf(base_user_feature_path)
            user_cate = pd.merge(user_cate, base_user_feature, how="left", on="user_id")
            print(user_cate.shape)
            del base_user_feature

            print("load base cate feature...")
            base_cate_feature_path = feature_path + f"base_cate_feature_{str(target_day)}.h5"
            if not os.path.exists(base_cate_feature_path):
                extract_cate_base_feature(target_day)
            base_cate_feature = pd.read_hdf(base_cate_feature_path)
            user_cate = pd.merge(user_cate, base_cate_feature, how="left", on="cate")
            print(user_cate.shape)

            del base_cate_feature

            print("load user cate feature...")
            user_cate_feature_path = feature_path + f"user_cate_feature_{str(target_day)}.h5"
            if not os.path.exists(user_cate_feature_path):
                extract_user_cate_feature(target_day)
            user_cate_feature = pd.read_hdf(user_cate_feature_path)
            user_cate = pd.merge(user_cate, user_cate_feature, how="left", on=['user_id', "cate"])
            print(user_cate.shape)
            del user_cate_feature

            print("load user feature...")
            user_action_feature_path = feature_path + f"user_feature_{str(target_day)}.h5"
            if not os.path.exists(user_action_feature_path):
                extract_user_action_feature(target_day)
            user_feature = pd.read_hdf(user_action_feature_path)
            user_cate = pd.merge(user_cate, user_feature, how="left", on="user_id")
            print(user_cate.shape)

            del user_feature


            windows_path = [feature_path + f"move_window_user_feature_{str(target_day)}_{window}.h5" for window in windows] + \
                [feature_path + f"move_window_cate_feature_{str(target_day)}_{window}.h5" for window in windows] + \
                    [feature_path + f"move_window_user_cate_feature_{str(target_day)}_{window}.h5" for window in windows]
            
            for i in windows_path:
                if not os.path.exists(i):
                    print(f"file {i} is not exists...\n start_extract_feature...")
                    extract_user_cate_feature(target_day, windows)
                    break

            print("load user window feature...")
            for window in windows:
                window_user_feat = pd.read_hdf(feature_path + f"move_window_user_feature_{str(target_day)}_{window}.h5")
                user_cate = pd.merge(user_cate, window_user_feat, how="left", on="user_id")
            print(user_cate.shape)

            print("load cate window feature...")
            for window in windows:
                window_user_feat = pd.read_hdf(feature_path + f"move_window_cate_feature_{str(target_day)}_{window}.h5")
                user_cate = pd.merge(user_cate, window_user_feat, how="left", on="cate")
            print(user_cate.shape)
            
            print("load user_cate window feature...")
            for window in windows:
                window_user_feat = pd.read_hdf(feature_path + f"move_window_user_cate_feature_{str(target_day)}_{window}.h5")
                user_cate = pd.merge(user_cate, window_user_feat, how="left", on=['user_id', "cate"])
            print(user_cate.shape)
               
            user_cate = reduce_mem_usage(user_cate)
            assert user_cate_len == len(user_cate)
            with open(cache_path + f"{str(target_day)}_{befor_days}_{candidate_set}.pickle", "wb") as f:
                    pickle.dump((user_cate, label_area), f)
            
        labels.append(label_area)
        datas.append(user_cate)
    print(len(labels), len(datas))
    assert len(labels) == len(datas)
    user_cate = pd.concat(datas, axis=0)
    labels = pd.concat(labels, axis=0)
    return (user_cate, labels)

"""
f1 的评测方式计算：
A: 提交的文件中 user_id cate 组合数(去重)
B：目标区间发生交易的 user_id cate 组合数(去重)
正确率：A正确的数量 / A数量
召回率：A正取的数量 / B数量

从公式来看准确率更重要。
(3 * 正确率 * 召回率) / (2 * 召回率 + 正确率)
"""
def metrice_f11(train_submit, train_labels):
    cache_table = pd.merge(train_labels, train_submit[['user_id', "cate", "time_area", "pre_label"]], how="left", on=['user_id', "cate", "time_area"])
    cache_table.fillna(0, inplace=True)
    pre_s = np.sum(cache_table['pre_label']) / len(train_submit[train_submit["pre_label"]==1])
    rec_s = np.sum(cache_table['pre_label']) / len(train_labels)
    now_score = (3 * pre_s * rec_s) / (2 * rec_s + pre_s)
    return now_score

def get_threshold_metrice_f11(train_submit, train_labels):
    threshold = 0
    score = 0
    for i in np.arange(np.min(train_submit["pre"].values), np.max(train_submit["pre"].values),0.01):
        train_submit["pre_label"] = np.where(train_submit["pre"].values >= i,1,0)
        now_score = metrice_f11(train_submit, train_labels)
        # now_score = metric_f11(pro, train_submit['label'].values)
        if now_score > score:
            score = now_score
            threshold = i
            print("update threshold...")
            print(f"now score:{now_score}\nbest score:{score}\nthreshold :{threshold}\nall samples: {len(train_submit)}\ntrue samples: {len(train_submit[train_submit['pre_label']==1])}")
            print("########" * 5)
    return threshold


"""
训练 user-cate 模型
train_times， test_times, dev_times：list: list 目标区间开始日期
befor_days: 以目标区间前多少天构建训练数据集
candidate_set："以所有出现过的 user_id + cate 组合构建数据集" if ”all“ else ”以购买过的 user_id + cate 组合构建训练集" 就f11而言 all 效果更好。
cv: if True, use lightgbm cv
mark: str，如果不填输出的文件会以训练时间结尾，如果填了以 mark 结尾
params: dict 会重载函数中模型定义的模型参数
ouput_feature_important：是否输出特征重要性
windows: [1, 3, 7 ,15, 30] 特征提取窗口
"""
def train_cate_model(train_times,
         test_times=[date(2018, 4, 16)],
         dev_times=[],
         befor_days=15,
         candidate_set="all",
         mark="",
         windows=[1,3,7,14,30],
         params={},
         cv=False,
         ouput_feature_important=True,
         num_boost_round=100000,
         early_stopping_rounds=100):

    if not mark:
        mark = datetime.now().strftime('%m-%d_%H-%M')

    train_date, train_labels = create_date(
        train_times, schema="train", befor_days=befor_days, candidate_set=candidate_set, windows=windows)
    print(f"train x shape : {train_date.shape}")
    print(f"train labels shape : {train_labels.shape}")
    
    test_date,test_labels  = create_date(
        test_times, schema="test", befor_days=befor_days, candidate_set=candidate_set, windows=windows)
    print(f"test shape : {test_date.shape}")
    if dev_times:
        dev_date, dev_labels = create_date(
            dev_times, schema="train", befor_days=befor_days, candidate_set=candidate_set, windows=windows)
        print(f"dev shape : {dev_date.shape}")
        print(f"dev labels shape : {dev_labels.shape}")
        dev_submit = dev_date[['user_id', "cate", "label", "time_area"]]

    

    train_submit = train_date[['user_id', "cate", "label", "time_area"]]
    test_submit = test_date[['user_id', "cate", "time_area", "label"]]
    # train_d = lgb.Dataset(train_date.drop(['user_id', 'cate', "label"], axis=1), train_date['label'])
    # test_d = lgb.Dataset(test_date.drop(['user_id', 'cate', "label"], axis=1))
    # # del train_date, test_date

    # model = SBBTree(params=params,\
    #                       stacking_num=0,\
    #                       bagging_num=4,\
    #                       bagging_test_size=0.2,\
    #                       num_boost_round=10000,\
    #                       early_stopping_rounds=50,
    #                       verbose_eval=10)

    class_feature = ["age", "sex", "city_level","cate", "province","cate", "city","county"]
    params_1 = {
        "objective": "binary",
        "boosting": "gbdt",
        "num_leaves": 30,
        "max_depth": 5,
        'min_child_samples': 50,
        'subsample_freq': 5,
        'subsample': 0.9,
        'colsample_bytree': 0.6,
        "early_stopping_round": 50,
        "min_gain_to_split": 0.01,
        "max_bin": 250,
        'metric': 'auc',
        "is_unbalance": True,
        "lambda_l1": 0.2,
        "lambda_l2": 0.0,
        "learning_rate": 0.01
    }

    if params:
        for k, v in params.items():
            params_1[k] = v

    lgb_train = lgb.Dataset(train_date.drop(
        ['user_id', "label", "time_area"], axis=1), train_date['label'].values)
    if cv:
        print("Start cv")
        cv_res = lgb.cv(params_1, lgb_train, num_boost_round=num_boost_round, nfold=5, stratified=True,
                        shuffle=True, early_stopping_rounds=early_stopping_rounds, verbose_eval=20)
        print(cv_res)
        return

    if dev_times:
        lgb_eval = lgb.Dataset(dev_date.drop(
            ['user_id', "label", "time_area"], axis=1), dev_date['label'].values)
        model = lgb.train(params_1,
                          train_set=lgb_train,
                          valid_sets=[lgb_train, lgb_eval],
                          valid_names=['train', 'eval'],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          categorical_feature= class_feature,
                          verbose_eval=10)
    else:
        model = lgb.train(params_1,
                          train_set=lgb_train,
                          valid_sets=[lgb_train],
                          valid_names=['train'],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=10)

    print("start predict...")

    train_submit['pre'] = model.predict(train_date.drop(
        ['user_id', "label", "time_area"], axis=1).values,  num_iteration=model.best_iteration)
    if dev_times:
        dev_submit['pre'] = model.predict(dev_date.drop(
            ['user_id', "label", "time_area"], axis=1).values,  num_iteration=model.best_iteration)

    test_submit["pre"] = model.predict(test_date.drop(
        ['user_id', "label", "time_area"], axis=1).values,  num_iteration=model.best_iteration)

    # new_train_metrice = pd.merge(train_labels, train_submit[['user_id', "cate", "time_area", "pre"]], how="left", on=['user_id', "cate", "time_area"])
    # del train_labels

    # new_train_metrice.fillna(0, inplace=True)
    print("select threshold...")
    threshold = get_threshold_metrice_f11(train_submit, train_labels)
    print(threshold)
    train_submit['pre_label'] = np.where(train_submit["pre"] > threshold, 1, 0)
    if dev_times:
        dev_submit["pre_label"] = np.where(dev_submit['pre'] > threshold, 1, 0)
        print(f"dev true sample: {len(dev_submit[dev_submit['pre_label']==1])}")
        dev_score = metrice_f11(dev_submit, dev_labels)
        print(f"dev score : {dev_score}")
    test_submit['pre_label'] = np.where(test_submit["pre"] > threshold, 1, 0)

    print(f"写入结果数据 to {result_path}")

    train_submit[['user_id', 'cate', "pre_label", "pre", "time_area"]].to_csv(
        result_path + f"train_submit_cate_{'_and_'.join([str(i) for i in train_times])}_{mark}.csv", index=False)
    test_submit[['user_id', 'cate', "pre_label", "pre", "time_area"]].to_csv(
        result_path + f"test_submit_cate_{'_and_'.join([str(i) for i in test_times])}_{mark}.csv", index=False)
    if dev_times:
        dev_submit[['user_id', 'cate', "pre_label", "pre", "time_area"]].to_csv(
            result_path + f"dev_submit_cate_{'_and_'.join([str(i) for i in dev_times])}_{mark}.csv", index=False)

    if ouput_feature_important:
        with open(feature_improtant_path + f"/feature_important_{mark}.pickle", "wb") as f:
            feature_importance = model.feature_importance()
            feature_dict = {k: v for k, v in zip(train_date.drop(
                ['user_id', "cate", "label", "time_area"], axis=1).columns, feature_importance)}
            feature_dict = sorted(feature_dict.items(),
                                  key=lambda s: s[1], reverse=True)
            print("输出特征重要性。。。")
            print(feature_dict[:20])
            pickle.dump(feature_dict, f)
    # cv_res = lgb.cv(params, train_d, nfold=10, num_boost_round=2000,early_stopping_rounds=50, seed=2019)
    # print(cv_res)
    # train_res = model.predict(train_d, num_iteration=model.best_iteration)

    # score = 0
    # final_threshold = 0
    # for threshold in range(0.1, 1, 0.1):
    #     now_score = f_score_cate(np.where(train_res > threshold, 1,0), train_d.get_label())
    #     if now_score > score:
    #         score = now_score
    #         final_threshold = threshold

    # test_res = model.predict(test_d, num_iteration=model.best_iteration)
    # submit["label"] = np.where(test_res > )
def cv(train_x, train_y, test_x, n_split):
    params_1 = {
        "objective": "binary",
        "boosting": "gbdt",
        "num_leaves": 30,
        "max_depth": 5,
        'min_child_samples': 50,
        'subsample_freq': 5,
        'subsample': 0.9,
        'colsample_bytree': 0.6,
        "early_stopping_round": 50,
        "min_gain_to_split": 0.01,
        "max_bin": 250,
        'metric': ["binary_logloss", 'auc'],
        "is_unbalance": True,
        "lambda_l1": 0.2,
        "lambda_l2": 0.0,
        "learning_rate": 0.01,
        "seed":2019
    }
    sf = StratifiedKFold(n_splits=n_split,random_state=0,shuffle=False)
    dev_res = np.zeros((train_x.shape[0], 1))
    test_res = np.zeros((test_x.shape[0], 1))
    dev_loss_score = []
    dev_auc_score = []
    for train_index, test_index in sf.split(train_x,train_y):
        X_train, y_train = train_x[train_index,:], train_y[train_index,:]
        X_test, y_test = train_x[test_index,:], train_y[test_index,:]
        
        dtrain = lgb.Dataset(X_train, y_train)
        ddev = lgb.Dataset(X_test, y_test, reference=dtrain)
        dtest = lgb.Dataset(test_x)
        model = lgb.train(params_1,
                          train_set=dtrain,
                          valid_sets=[dtrain, ddev],
                          valid_names=['train', 'eval'],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=10)
        
        dev_res = model.predict(ddev, num_iteration=model.best_iteration)
        train_res = model.predict(dtrain, num_iteration=model.best_iteration)
        dev_loss_score.append(log_loss(y_test, dev_res))
        dev_auc_score.append(roc_auc_score(y_train, train_res))
        print("预测结果")
        print(f"train logloss: {log_loss(y_train, train_res)} auc: {roc_auc_score(y_train, train_res)}\n\
            test logloss: {log_loss(y_test, dev_res)} auc: {roc_auc_score(y_test, dev_res)}")
        t_res = model.predict(dtest)
        dev_res[test_index,0] = res
        test_res += t_res / n_split
    print(f"dev loss：mean: {np.mean(dev_loss_score)} std: {np.std(dev_loss_score)}")
    print(f"dev auc：mean: {np.mean(dev_auc_score)} std {np.std(dev_auc_score)}")



if __name__ == "__main__":
    train_times1 = [date(2018, 4, 2), date(2018, 4, 9)]
    train_times2 = [date(2018, 4, 2)]
    dev_times = [date(2018, 4, 9)]
    test_times = [date(2018, 4, 16)]

    # train_cate_model(train_times2, test_times, dev_times=dev_times, params={
    #      "learning_rate": 0.01}, befor_days=30, cv=False, num_boost_round=1300, early_stopping_rounds=50)
    train_date, train_labels = create_date(
        train_times1, schema="train", befor_days=30, candidate_set="all", windows=[1, 3, 7, 14, 30])
    
    test_date,test_labels  = create_date(
        test_times, schema="test", befor_days=30, candidate_set="all", windows=[1, 3, 7, 14, 30])


    print(f"test shape : {test_date.shape}")
    print(f"train x shape : {train_date.shape}")
    print(f"train labels shape : {train_labels.shape}")

    cv(train_date.drop(['user_id', 'cate', "label", "time_area"], axis=1).values,
    train_date['label'].values, test_date.drop(['user_id', 'cate', "label", "time_area"], axis=1).values, n_split=5)




