#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'user_cate_section'))
	print(os.getcwd())
except:
	pass

#%%
import lightgbm as lgb
import pandas as pd
from datetime import datetime, date, timedelta
from preprocess import reduce_mem_usage
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
import pickle
from tools import reduce_mem_usage
import os
import gc
from cate_base_feature import extract_cate_base_feature
from user_base_feature import extract_user_base_feature
from user_action_feature import extract_user_action_feature
from user_cate_action_feature import extract_user_cate_feature
from config import *
from shop_feature import extract_shop_info
from user_shop_feature import extract_user_shop_feature
from metrics import get_threshold_and_metrice_score, f11_f12_weighted_score, f11_score, f12_score
from cate_model import *
import warnings
warnings.filterwarnings("ignore")


def get_label_user_cate_shop(action_table, target_time):
    end_time = target_time + timedelta(7)
    action_table = action_table[action_table['type'] == 2]
    action_table = action_table[(action_table['action_time'] >= target_time)&(action_table['action_time'] < end_time)]
    user_cate_shop = action_table[['user_id', "cate", "shop_id"]].drop_duplicates().dropna()
    user_cate_shop['time_area'] = str(target_time)
    return user_cate_shop

def get_user_cate_data(action_table, target_day, target_info=None, windows=[1,3,7,14,30], before_days=30,model="train", candidate_set="all"):
    
    if os.path.exists(cache_path + f"{str(target_day)}_{before_days}_{candidate_set}_{'_'.join([str(i) for i in windows])}.pickle"):
        with open(cache_path + f"{str(target_day)}_{before_days}_{candidate_set}_{'_'.join([str(i) for i in windows])}.pickle", "rb") as f:
            user_cate = pickle.load(f)
            return user_cate
    print("构建 {} 的 user-cate 训练集".format(str(target_day)))
    if model == "train":
        if target_info is None:
            target_info = get_label_user_cate_shop(action_table, target_day)
    start_day = target_day - timedelta(before_days)
    if candidate_set != "all":
        action_table = action_table[action_table['type'] == 2]
    action_table = action_table[(action_table['action_time'] >= start_day) & (action_table['action_time'] < target_day)]
    user_cate = action_table[['user_id', "cate"]].drop_duplicates().dropna()
    if model == "train":
        target_info = target_info[['user_id', "cate", "time_area"]].drop_duplicates()
        target_info['label'] = 1
        user_cate_len = len(user_cate)
        user_cate['time_area'] = str(target_day)
        user_cate = pd.merge(user_cate, target_info, how="left", on=['user_id', "cate", "time_area"])
        assert len(user_cate) == user_cate_len
        user_cate.fillna({"label":0}, inplace=True)
        print("样本比例：{}".format(user_cate['label'].value_counts()))
    else:
        user_cate['time_area'] = str(target_day)
        user_cate['label'] = 0
    user_cate_len = len(user_cate)

    print("加载基础用户特征。。")
    base_user_feature_path = feature_path + f"base_user_feature_{str(target_day)}.h5"
    base_user_feature = pd.read_hdf(base_user_feature_path)
    user_cate = pd.merge(user_cate, base_user_feature, how="left", on="user_id")
    assert len(user_cate) == user_cate_len

    print("加载基础类别特征。。")
    base_cate_feature_path = feature_path + f"base_cate_feature_{str(target_day)}.h5"
    base_cate_feature = pd.read_hdf(base_cate_feature_path)
    user_cate = pd.merge(user_cate, base_cate_feature, how="left", on="cate")
    assert len(user_cate) == user_cate_len
    
    print("加载用户类别交互特征。。")
    user_cate_feature_path = feature_path + f"user_cate_feature_{str(target_day)}.h5"
    user_cate_feature = pd.read_hdf(user_cate_feature_path)
    user_cate = pd.merge(user_cate, user_cate_feature, how="left", on=['user_id', "cate"])
    assert len(user_cate) == user_cate_len
    
    print("加载用户行为特征。。")
    user_action_feature_path = feature_path + f"user_feature_{str(target_day)}.h5"
    user_feature = pd.read_hdf(user_action_feature_path)
    user_cate = pd.merge(user_cate, user_feature, how="left", on="user_id")
    assert len(user_cate) == user_cate_len
    
    print("加载用户划窗特征。。")
    for window in windows:
        window_user_feat = pd.read_hdf(feature_path + f"move_window_user_feature_{str(target_day)}_{window}.h5")
        user_cate = pd.merge(user_cate, window_user_feat, how="left", on="user_id")
    assert len(user_cate) == user_cate_len
    
    print("加载类别划窗特征。。")
    for window in windows:
        window_user_feat = pd.read_hdf(feature_path + f"move_window_cate_feature_{str(target_day)}_{window}.h5")
        user_cate = pd.merge(user_cate, window_user_feat, how="left", on="cate")
    assert len(user_cate) == user_cate_len
    
    print("加载用户类别交互划窗特征。。")
    for window in windows:
        window_user_feat = pd.read_hdf(feature_path + f"move_window_user_cate_feature_{str(target_day)}_{window}.h5")
        user_cate = pd.merge(user_cate, window_user_feat, how="left", on=['user_id', "cate"])
    assert len(user_cate) == user_cate_len    
    
    user_cate = reduce_mem_usage(user_cate)
                  
    with open(cache_path + f"{str(target_day)}_{before_days}_{candidate_set}_{'_'.join([str(i) for i in windows])}.pickle", "wb") as f:
        pickle.dump(user_cate, f)
    assert len(user_cate) == len(user_cate[['user_id',"cate"]].drop_duplicates())    
    return user_cate

def get_user_cate_shop_data(action_table, target_day, user_cate=None, target_info=None,model="train", windows=[1,3,7,14,30], before_days=30, candidate_set="all"):
    if model == "train":
        if target_info is None:
            target_info = get_label_user_cate_shop(action_table, target_day)
    
    if user_cate is None:
        user_cate = get_user_cate_data(action_table, target_day, target_info=target_info,model=model, windows=windows, before_days=before_days, candidate_set=candidate_set)
    start_day = target_day - timedelta(before_days)
    if candidate_set != "all":
        action_table = action_table[action_table['type'] == 2]
    action_table = action_table[(action_table['action_time'] >= start_day) & (action_table['action_time'] < target_day)]
    
    user_cate_shop = action_table[['user_id', "cate", "shop_id"]].drop_duplicates().dropna()
    user_cate_shop['time_area'] = str(target_day)
    del action_table
    gc.collect()
    if "label" in user_cate.columns:
        user_cate.drop("label", axis=1, inplace=True)
    if model =="train":
        target_info = target_info[['user_id', "cate", "shop_id", "time_area"]].drop_duplicates()
        assert len(target_info) == len(target_info.dropna())
        target_info['label'] = 1
        user_cate_shop_len = len(user_cate_shop)
        user_cate_shop = pd.merge(user_cate_shop, target_info, how="left", on=['user_id', "cate", "shop_id", "time_area"])
        assert len(user_cate_shop) == user_cate_shop_len
        user_cate_shop.fillna({"label":0}, inplace=True)
        print("样本比例：{}".format(user_cate_shop['label'].value_counts()))
        assert len(user_cate_shop) == len(user_cate_shop[['user_id',"cate", "shop_id"]].drop_duplicates())    
    else:
        user_cate_shop['label'] = 0
    user_cate_shop = pd.merge(user_cate, user_cate_shop, how="left", on=['user_id', "cate", "time_area"])
    print("召回后样本比例：{}".format(user_cate_shop['label'].value_counts()))
    assert len(user_cate_shop) == len(user_cate_shop[['user_id',"cate", "shop_id"]].drop_duplicates())   
    user_cate_shop_len = len(user_cate_shop)
    print("加载店铺特征。。")
    shop_feature_path = feature_path +  f"shop_info_{str(target_day)}.h5"
    shop_feature = pd.read_hdf(shop_feature_path)
    user_cate_shop = pd.merge(user_cate_shop, shop_feature, how="left", on="shop_id")
    assert len(user_cate_shop) == user_cate_shop_len
    
    print("加载用户店铺特征。。")
    user_shop_feature_path = feature_path +  f"user_shop_feature_{str(target_day)}.h5"
    user_shop_feature = pd.read_hdf(user_shop_feature_path)
    user_cate_shop = pd.merge(user_cate_shop, user_shop_feature, how="left", on=['user_id', "shop_id"])
    assert len(user_cate_shop) == user_cate_shop_len
    
    print("加载类别店铺特征。。")
    cate_shop_feature_path = feature_path +  f"cate_shop_feature_{str(target_day)}.h5"
    cate_shop_feature = pd.read_hdf(cate_shop_feature_path)
    user_cate_shop = pd.merge(user_cate_shop, cate_shop_feature, how="left", on=['cate', "shop_id"])
    assert len(user_cate_shop) == user_cate_shop_len
    
    print("加载店铺窗口特征。。")
    for window in windows:
        window_shop_feat = pd.read_hdf(feature_path + f"move_window_shop_feature_{str(target_day)}_{window}.h5")
        user_cate_shop = pd.merge(user_cate_shop, window_shop_feat, how="left", on="shop_id")
        del window_shop_feat
        gc.collect()
    assert len(user_cate_shop) == user_cate_shop_len
    
    print("加载用户店铺窗口特征...")
    for window in windows:
        window_user_shop_feat = pd.read_hdf(feature_path + f"move_window_user_shop_feature_{str(target_day)}_{window}.h5")
        user_cate_shop = pd.merge(user_cate_shop, window_user_shop_feat, how="left", on=['user_id', "shop_id"])
        del window_user_shop_feat
        gc.collect()
    
    assert len(user_cate_shop) == len(user_cate_shop[['user_id',"cate", "shop_id"]].drop_duplicates())    
    return user_cate_shop


#%%
params_1 = {
    "objective": "binary",
    "boosting": "gbdt",
    "num_leaves": 21,
    "max_depth": 5,
    'min_child_samples': 50,
    'subsample_freq': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    "early_stopping_round": 50,
    "min_gain_to_split": 0.01,
    "max_bin": 250,
    'metric': ["binary_logloss", 'auc'],
    "lambda_l1": 0.7,
    "lambda_l2": 0.0,
    "learning_rate": 0.01,
    "seed":2019,
    "verbose_eval":20
}


#%%
class CvModel(object):
    def __init__(self, params, n_splits, num_boost_round, early_stopping_rounds, cate_feature, mark="cv"):
        self.params = params
        self.n_splits = n_splits
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.cate_feature = cate_feature
        self.mark = mark
        self.model = []
        self.aucs = []
        self.logloss = []

    def fit(self, X, y):
        layer_train = np.zeros((X.shape[0], 1))
        self.SK = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1)
        for k,(train_index, test_index) in enumerate(self.SK.split(X, y)):
            print(f"######## cv {k} #########")
            X_train = X.iloc[train_index,:]
            y_train = y.iloc[train_index,:].values.reshape(-1)
            X_test = X.iloc[test_index,:]
            y_test = y.iloc[test_index,:].values.reshape(-1)

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            gbm = lgb.train(self.params,
                        train_set=lgb_train,
                        valid_sets=[lgb_train, lgb_eval],
                        valid_names= ['train', 'eval'],
                        num_boost_round=self.num_boost_round,
                        early_stopping_rounds=self.early_stopping_rounds,
                           categorical_feature=self.cate_feature)
            self.model.append(gbm)
            pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            layer_train[test_index, 0] = pred_y
            self.aucs.append(roc_auc_score(y_test, pred_y))
            self.logloss.append(log_loss(y_test, pred_y))
    def predict(self, X):
        assert len(self.model) == self.n_splits
        test_pred = np.zeros((X.shape[0], 1))
        for sn,gbm in enumerate(self.model):
            pred = gbm.predict(X, num_iteration=gbm.best_iteration)
            test_pred[:, 0] += pred/self.n_splits
        return test_pred
    
    def save_model(self):
        for sn, gbm in enumerate(self.model):
            gbm.save_model(f"../input/model/best_cv_model_f11_{self.mark}_{sn}.bin")
        print(f"{self.mark} save complete...")

    def load_model(self):
        for i in range(self.n_splits):
            if os.path.exists(f"../input/model/best_cv_model_f11_{self.mark}_{i}.bin"):   
                print(f"加载模型：../input/model/best_cv_model_f11_{self.mark}_{i}.bin")
                _model =  lgb.Booster(model_file=f"../input/model/best_cv_model_f11_{self.mark}_{i}.bin")
                self.model.append(_model)
            else:
                print(f"模型地址不存在： ../input/model/best_cv_model_f11_{self.mark}_{i}.bin")
                self.model =  []
                return False
        return True

#%%
def get_f11_threshold(train_sub, train_label):
    score = 0
    threshold = 0
    for i in np.arange(min(train_sub['pre']), max(train_sub['pre']), 0.01):
        i = round(i, 2)
        train_sub['pre_label'] = np.where(train_sub['pre'] >= i, 1, 0)
        f11score = f11_score(train_sub, train_label)
        if f11score > score:
            score = f11score
            threshold = i
    print("f11最高分数: {}".format(score))
    print("当前阈值: {}".format(threshold))
    train_sub['pre_label'] = np.where(train_sub['pre'] >= threshold, 1, 0)
    print("当前提交样本数: {}".format(sum(train_sub['pre_label'])))
    print(">>>>>>>>>>>>> ")
    return threshold


#%%
# 初始化变量
lr = 0.01
train_time = date(2018,4,2)
dev_time = date(2018,4,9)
test_time = date(2018,4,16)
windows=[1,3,7,14,30]
before_days = 30
candidate_set = "all"
class_feature = ["age", "sex", "city_level", "province", "city","county", "cate"]


#%%
# 加载训练集合
action_table = pd.read_hdf(base_file_path + "jdata_action.h5")
action_table['action_time'] = action_table['action_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
action_table = pd.merge(action_table, product_table, how="left", on="sku_id")
action_table = action_table[['user_id', 'cate', "shop_id","action_time", "type"]]
del product_table
gc.collect()

#%%
# 加载目标区间购买的组合
train_label = get_label_user_cate_shop(action_table, train_time)
dev_label = get_label_user_cate_shop(action_table, dev_time)

#%%
# 加载user-cate 特征
# user_cate 的训练集
# user_id， cate， user_id-cate
# feature1 = 你的函数
# pd.merge 
# 2 9 
# 购物车
# 2号的特征数量少， 9号多
# 2.columns
# 9.columns
# 并集
# for col in 2.columns:
    # if col not in 并集:
    #     2[col] = np.NaN
# 2 he 9
# 2 = 2[并集]
# 9 = 9[并集]

train = get_user_cate_data(action_table, train_time, target_info=train_label, windows=windows, before_days=before_days, candidate_set=candidate_set)
dev = get_user_cate_data(action_table, dev_time, target_info=dev_label, windows=windows, before_days=before_days, candidate_set=candidate_set)
test = get_user_cate_data(action_table, test_time,model="test",target_info=None, windows=windows, before_days=before_days, candidate_set=candidate_set)


params_pre = {
    "objective": "binary",
    "boosting": "gbdt",
    "num_leaves": 21,
    "max_depth": 5,
    'min_child_samples': 60,
    'subsample_freq': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    "min_gain_to_split": 0.01,
    "max_bin": 250,
    'metric': ['auc', "binary_logloss"],
    "lambda_l1": 0.7,
    "lambda_l2": 0.0,
    "learning_rate": lr,
    "seed":0
}

if os.path.exists("../input/model/best_model_f11_2_4_19.bin"):
    model = lgb.Booster(model_file="../input/model/best_model_f11_2_4_19.bin")
else:
    lgb_train = lgb.Dataset(
        train.drop(['user_id', "label", "time_area"], axis=1),
        train['label'].values)


    lgb_eval = lgb.Dataset(
        dev.drop(['user_id', "label", "time_area"], axis=1),
        dev['label'].values)

    print("开始训练召回模型...")
    model = lgb.train(params_1,
                    train_set=lgb_train,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', "dev"],
                    num_boost_round=5000,
                    categorical_feature=class_feature,
                    verbose_eval=50)

    print("保存 f11 2号 模型。。.")
    model.save_model("../input/model/best_model_f11_2_4_19.bin")

# train_sub['pre'] =  model.predict(train.drop(['user_id', "label", "time_area"], axis=1), num_iteration=model.best_iteration)
dev['model2tpre'] = model.predict(dev.drop(['user_id', "label", "time_area"], axis=1), num_iteration=model.best_iteration)
test['model2tpre'] = model.predict(test.drop(['user_id', "label", "time_area"], axis=1), num_iteration=model.best_iteration)    


# f11
#%%
model = CvModel(params_1, 5, 10000, 50, class_feature, mark="best_model_cv_f11_9")


#%%
if not  model.load_model():
    model.fit(dev.drop(['user_id', "label", "time_area"], axis=1), dev[["label"]])
    model.save_model()
# for index, model in enumerate(model.model):
#     model.save_model(f'../input/result/cv_f11model_29t_5_17_{index}.bin', num_iteration=model.best_iteration)

#%%
dev_sub =  dev[['user_id', "cate", "time_area"]]
test_sub = test[['user_id', "cate", "time_area"]]


#%%
dev_sub['pre'] = model.predict(dev.drop(['user_id', "label", "time_area"], axis=1))
test_sub['pre'] = model.predict(test.drop(['user_id', "label", "time_area"], axis=1))


#%%
dev_threshold = get_f11_threshold(dev_sub, dev_label)


#%%
threshold = dev_threshold


#%%
dev_sub['pre_label'] = np.where(dev_sub['pre'] >= threshold, 1, 0)
test_sub['pre_label'] = np.where(test_sub['pre'] >= threshold, 1, 0)
print("当前阈值的验证集得分：{}".format(f11_score(dev_sub, dev_label)))


#%%[2, 9] f12, 训练f11时候的特征

dev = dev[dev_sub['pre_label'] == 1]
test = test[test_sub['pre_label'] == 1]

# 拼接 shop 特征
#%% 
dev = get_user_cate_shop_data(action_table, dev_time, user_cate=dev, target_info=dev_label, windows=[1,3,7,14,30], before_days=30, candidate_set="all")
test = get_user_cate_shop_data(action_table, test_time, user_cate=test,model="test",target_info=None, windows=[1,3,7,14,30], before_days=30, candidate_set="all")


#%%
del action_table
gc.collect()


#%%
# 直接将召回的类别中所有看过的店铺作为结果。得分。
dev_sub = dev[['user_id', "cate", "shop_id"]]
dev_sub['pre_label'] = 1
print(f11_f12_weighted_score(dev_sub, dev_label))




#%%
model = CvModel(params_1, 5, 10000, 50, class_feature, "best_model_cv_f12_9")
if not model.load_model():
    model.fit(dev.drop(['user_id', "shop_id", "label", "time_area", "shop_cate"], axis=1), dev[["label"]])
    model.save_model()

#%%
dev_sub =  dev[['user_id', "cate", "shop_id", "time_area"]]
test_sub = test[['user_id', "cate","shop_id", "time_area"]]


#%%
dev_sub['pre'] = model.predict(dev.drop(['user_id', "shop_id", "label", "time_area", "shop_cate"], axis=1))
test_sub['pre'] = model.predict(test.drop(['user_id', "shop_id", "label", "time_area", "shop_cate"], axis=1))


#%%
dev_threshold = get_threshold_and_metrice_score(dev_sub, dev_label)

#%%
threshold = dev_threshold

#%%
dev_sub['pre_label'] = np.where(dev_sub['pre'] >= threshold, 1, 0)
test_sub['pre_label'] = np.where(test_sub['pre'] >= threshold, 1, 0)
print("当前阈值的验证集得分：{}".format(f11_f12_weighted_score(dev_sub, dev_label)))

print("保存 untitled-3 结果")
dev_sub.to_csv("../input/result/untitled_3_dev_result.csv", index=False)
test_sub.to_csv("../input/result/untitled_3_test_result.csv", index=False)


#%%
test_sub_num = sum(test_sub['pre'] >= threshold)
test_sub = test_sub[test_sub['pre_label'] == 1]
assert test_sub_num == len(test_sub)
test_sub['user_id'] = test_sub['user_id'].astype("int")
test_sub['cate'] = test_sub['cate'].astype("int")
test_sub['shop_id'] = test_sub['shop_id'].astype("int")
test_sub[['user_id',"cate","shop_id"]].drop_duplicates().to_csv("../input/result/untitled_3_submit.csv", index=False)


