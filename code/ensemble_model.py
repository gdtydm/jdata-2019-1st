import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from metrics import *
from sklearn import linear_model
from config import *
import gc
template = "../input/result/untitled_{num}_{time_area}_result.csv"


def get_action_table():
    action_table = pd.read_hdf(base_file_path + "jdata_action.h5")
    # action_table = pd.read_csv(base_file_path + "jdata_action.csv")
    action_table['action_time'] = action_table['action_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
    # product_table = pd.read_csv(base_file_path + "jdata_product.csv")
    action_table = pd.merge(action_table, product_table, how="left", on="sku_id")
    action_table = action_table[['user_id', 'cate', "shop_id","action_time", "type"]]
    del product_table
    gc.collect()
    return action_table

def get_label_user_cate_shop(action_table, target_time=date(2018,4,9)):
    end_time = target_time + timedelta(7)
    action_table = action_table[action_table['type'] == 2]
    action_table = action_table[(action_table['action_time'] >= target_time)&(action_table['action_time'] < end_time)]
    user_cate_shop = action_table[['user_id', "cate", "shop_id"]].drop_duplicates().dropna()
    user_cate_shop['time_area'] = str(target_time)
    return user_cate_shop


def manual_weight_ensemble(weight_dict, dev_label):
    print("########　使用加权融合 ########")
    combine_dev = []
    combine_test = []
    for idx, (k, v) in enumerate(weight_dict.items()):
        dev_path = template.format(num=k, time_area="dev")
        test_path = template.format(num=k, time_area="test")

        dev = pd.read_csv(dev_path)[['user_id', 'cate', 'shop_id', 'pre', "pre_label"]]
        print(f"验证集: {k}, 提交样本数：{np.sum(dev['pre_label']==1)}, 得分: {f11_f12_weighted_score(dev, dev_label)}")
        dev['pre'] = dev['pre'] * v
        # dev.rename(columns={"pre_label":f"pre_label_{idx}"}, inplace=True)

        test = pd.read_csv(test_path)[['user_id', 'cate', 'shop_id', 'pre', "pre_label"]]
        print(f"测试集: {k}, 提交样本数：{np.sum(test['pre_label']==1)}")

        test['pre'] = test['pre'] * v
        # test.rename(columns={"pre_label":f"pre_label_{idx}"}, inplace=True)
        combine_dev.append(dev)
        combine_test.append(test)
    
    test = pd.concat(combine_test)
    dev = pd.concat(combine_dev)


    dev = dev.groupby(['user_id', 'cate', 'shop_id']).sum().reset_index()
    test = test.groupby(['user_id', 'cate', 'shop_id']).sum().reset_index()

    dev_threshold = get_threshold_and_metrice_score(dev, dev_label)
    dev['pre_label'] = np.where(dev['pre'] >= dev_threshold, 1, 0)
    print(f"融合后验证集提交样本数：{np.sum(dev['pre_label']==1)}, 得分: {f11_f12_weighted_score(dev, dev_label)}")
    test['pre_label'] = np.where(test['pre'] >= dev_threshold, 1, 0)

    test = test[test['pre_label'] == 1]
    test['user_id'] = test['user_id'].astype("int")
    test['cate'] = test['cate'].astype("int")
    test['shop_id'] = test['shop_id'].astype("int")
    print("保存 manual_weight 结果")
    print(f"提交样本量：{len(test)}")
    test[['user_id',"cate","shop_id"]].drop_duplicates().to_csv("../input/result/manual_weight_ensemble.csv",index=False)
    


# def bagging_ensemble(weight_dict, dev_label):
#     combine_dev = None
#     combine_test = None
#     for idx, (k, v) in enumerate(weight_dict.items()):
#         dev_path = template.format(num=k, time_area="dev")
#         test_path = template.format(num=k, time_area="test")

#         dev = pd.read_csv(dev_path)[['user_id', 'cate', 'shop_id', 'pre_label']]

#         print(f"{k}提交样本数：{len(dev[dev['pre_label']==1])}")

#         dev.rename(columns={"pre_label":f"pre_label_{idx}"}, inplace=True)

#         test = pd.read_csv(test_path)[['user_id', 'cate', 'shop_id', 'pre_label']]
#         test.rename(columns={"pre_label":f"pre_label_{idx}"}, inplace=True)
#         if not idx:
#             combine_dev = dev
#             combine_test = test
#         else:
#             combine_dev = pd.merge(combine_dev, dev, how="outer", on=['user_id', 'cate', 'shop_id'])
#             combine_test = pd.merge(combine_test, test, how="outer", on=['user_id', 'cate', 'shop_id'])
#     print(combine_dev.info())

#     # combine_dev.fillna(0, inplace=True)
#     # combine_test.fillna(0, inplace=True)

#     combine_dev['pre_label'] = combine_dev[[f"pre_label_{idx}" for i in range(len(weight_dict))]].sum(axis=1)
#     combine_test['pre_label'] = combine_test[[f"pre_label_{idx}" for i in range(len(weight_dict))]].sum(axis=1)
#     print(combine_dev.head())
#     print(np.sum(combine_dev['pre_label'] >=2))
#     combine_test['pre_label'] = np.where(combine_test['pre_label'] >=2, 1, 0)
#     combine_dev['pre_label'] = np.where(combine_dev['pre_label'] >=2, 1, 0)

#     print(f"bagging 后验证集提交样本数：{np.sum([combine_dev['pre_label']==1])}")
#     print(f"bagging 后测试集提交样本数：{np.sum([combine_test['pre_label']==1])}")
#     print("bagging ensemble 验证集得分：{}".format(f11_f12_weighted_score(combine_dev, dev_label)))

#     combine_test = combine_test[combine_test['pre_label'] == 1]
#     combine_test['user_id'] = combine_test['user_id'].astype("int")
#     combine_test['cate'] = combine_test['cate'].astype("int")
#     combine_test['shop_id'] = combine_test['shop_id'].astype("int")
#     print("保存bagging 结果")
#     print(f"提交样本量：{len(combine_test)}")
#     combine_test[['user_id',"cate","shop_id"]].drop_duplicates().to_csv("../input/result/bagging_ensemble.csv",index=False)
    


# def manual_weight_ensemble(weight_dict, dev_label):
#     combine_dev = None
#     combine_test = None
#     for idx, (k, v) in enumerate(weight_dict.items()):
#         dev_path = template.format(num=k, time_area="dev")
#         test_path = template.format(num=k, time_area="test")
#         dev = pd.read_csv(dev_path)[['user_id', 'cate', 'shop_id', 'pre']]
#         dev['pre'] = dev['pre'] * v
#         dev.rename(columns={"pre":f"pre_{idx}"}, inplace=True)
#         test = pd.read_csv(test_path)[['user_id', 'cate', 'shop_id', 'pre']]
#         test['pre'] = test['pre'] * v
#         test.rename(columns={"pre":f"pre_{idx}"}, inplace=True)
#         if not idx:
#             combine_dev = dev
#             combine_test = test
#         else:
#             combine_dev = pd.merge(combine_dev, dev, how="inner", on=['user_id', 'cate', 'shop_id'])
#             combine_test = pd.merge(combine_test, test, how="inner", on=['user_id', 'cate', 'shop_id'])
#     print(combine_dev.info())
#     print(combine_dev.head())
#     combine_dev['pre'] = combine_dev[[f"pre_{idx}" for i in range(len(weight_dict))]].sum(axis=1)
#     combine_test['pre'] = combine_test[[f"pre_{idx}" for i in range(len(weight_dict))]].sum(axis=1)
#     dev_threshold = get_threshold_and_metrice_score(combine_dev, dev_label)
#     combine_test['pre_label'] = np.where(combine_test['pre'] >= dev_threshold, 1, 0)

#     combine_test = combine_test[combine_test['pre_label'] == 1]
#     combine_test['user_id'] = combine_test['user_id'].astype("int")
#     combine_test['cate'] = combine_test['cate'].astype("int")
#     combine_test['shop_id'] = combine_test['shop_id'].astype("int")
#     print("保存 manual_weight 结果")
#     print(f"提交样本量：{len(combine_test)}")
#     combine_test[['user_id',"cate","shop_id"]].drop_duplicates().to_csv("../input/result/manual_weight_ensemble.csv",index=False)
    


def bagging_ensemble(weight_dict, dev_label):
    print("#########　bagging 融合　##########")
    combine_dev = []
    combine_test = []
    for idx, (k, v) in enumerate(weight_dict.items()):
        dev_path = template.format(num=k, time_area="dev")
        test_path = template.format(num=k, time_area="test")

        dev = pd.read_csv(dev_path)[['user_id', 'cate', 'shop_id', 'pre_label']]

        print(f"验证集: {k}, 提交样本数：{np.sum(dev['pre_label']==1)}, 得分: {f11_f12_weighted_score(dev, dev_label)}")

        # dev.rename(columns={"pre_label":f"pre_label_{idx}"}, inplace=True)

        test = pd.read_csv(test_path)[['user_id', 'cate', 'shop_id', 'pre_label']]
        print(f"测试集: {k}, 提交样本数：{np.sum(test['pre_label']==1)}")
        # test.rename(columns={"pre_label":f"pre_label_{idx}"}, inplace=True)
        combine_dev.append(dev)
        combine_test.append(test)
    
    test = pd.concat(combine_test)
    dev = pd.concat(combine_dev)


    dev = dev.groupby(['user_id', 'cate', 'shop_id']).sum().reset_index()
    test = test.groupby(['user_id', 'cate', 'shop_id']).sum().reset_index()

    # combine_dev.fillna(0, inplace=True)
    # combine_test.fillna(0, inplace=True)

    # combine_dev['pre_label'] = combine_dev[[f"pre_label_{idx}" for i in range(len(weight_dict))]].sum(axis=1)
    # combine_test['pre_label'] = combine_test[[f"pre_label_{idx}" for i in range(len(weight_dict))]].sum(axis=1)
    # print(combine_dev.head())
    test['pre_label'] = np.where(test['pre_label'] >=2, 1, 0)
    dev['pre_label'] = np.where(dev['pre_label'] >=2, 1, 0)
    print(f"融合后验证集提交样本数：{np.sum(dev['pre_label']==1)}, 得分: {f11_f12_weighted_score(dev, dev_label)}")

    test = test[test['pre_label'] == 1]
    test['user_id'] = test['user_id'].astype("int")
    test['cate'] = test['cate'].astype("int")
    test['shop_id'] = test['shop_id'].astype("int")
    print("保存bagging 结果")
    print(f"提交样本量：{len(test)}")
    test[['user_id',"cate","shop_id"]].drop_duplicates().to_csv("../input/result/bagging_ensemble.csv",index=False)
    
if __name__ == "__main__":
    action_table = get_action_table()
    dev_label = get_label_user_cate_shop(action_table)
    weight_dict = {"3":0.4, "35":0.4, "4":0.2}
    manual_weight_ensemble(weight_dict, dev_label)
    # print("\n\n\n")
    # print("###### bagging_ensemble #######")
    # bagging_ensemble(weight_dict, dev_label)
    # dev1 = pd.read_csv("../input/result/untitled_35_dev_result.csv")[['user_id', 'cate', 'shop_id', 'pre_label']]
    # dev1.rename(columns={"pre_label":"pre_label_0"}, inplace=True)
    # dev2 = pd.read_csv("../input/result/untitled_3_dev_result.csv")[['user_id', 'cate', 'shop_id', 'pre_label']]
    # dev2.rename(columns={"pre_label":"pre_label_1"}, inplace=True)
    # dev3 = pd.read_csv("../input/result/untitled_4_dev_result.csv")[['user_id', 'cate', 'shop_id', 'pre_label']]
    # dev3.rename(columns={"pre_label":"pre_label_2"}, inplace=True)

    # dev = pd.merge(dev1, dev2, on=['user_id', "cate", "shop_id"], how="outer")
    # dev = pd.merge(dev, dev3, on=['user_id', "cate", "shop_id"], how="outer")

    # dev['pre_label'] = dev[['pre_label_0', "pre_label_1", "pre_label_2"]].sum(axis=1)
    # dev['pre_label'] = np.where(dev['pre_label'] >=2, 1, 0)
    # print("bagging ensemble 验证集得分：{}".format(f11_f12_weighted_score(dev, dev_label)))
