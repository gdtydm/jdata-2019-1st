from sklearn.metrics import precision_score, recall_score
import numpy as np
from datetime import datetime, date
import pandas as pd


def f11_score(pred, label_date):
    cate_cache_sub_table = pred[pred['pre_label'] == 1][['user_id', "cate", "pre_label"]]
    cate_cache_sub_table = cate_cache_sub_table.groupby(['user_id', "cate"], as_index=False).max()
    
    cate_cache_label_date = label_date[['user_id',"cate"]].drop_duplicates()
    cate_cache_label_date = pd.merge(cate_cache_label_date, cate_cache_sub_table, how="left", on=['user_id', "cate"])

    cate_prec = np.sum(cate_cache_label_date['pre_label']) / len(cate_cache_sub_table)
    cate_recall = np.sum(cate_cache_label_date['pre_label']) / len(cate_cache_label_date)

    cate_score = (3 * cate_prec * cate_recall) / (2 * cate_recall + cate_prec)
    return cate_score

def f12_score(pred, label_date):
    shop_cache_sub_table = pred[pred['pre_label'] == 1][['user_id', "cate", "shop_id", "pre_label"]]
    shop_cache_sub_table = shop_cache_sub_table.groupby(['user_id', "cate", "shop_id"], as_index=False).max()

    shop_cache_label_date = label_date[['user_id',"cate", "shop_id"]].drop_duplicates()
    shop_cache_label_date = pd.merge(shop_cache_label_date, shop_cache_sub_table, how="left", on=['user_id', "cate", "shop_id"])

    shop_prec = np.sum(shop_cache_label_date['pre_label']) / len(shop_cache_sub_table)
    shop_recall = np.sum(shop_cache_label_date['pre_label']) / len(shop_cache_label_date)

    shop_score = (5 * shop_prec * shop_recall) / (2 * shop_recall + 3 * shop_prec)
    return shop_score

def f11_f12_weighted_score(pred, label_date):
    f11 = f11_score(pred, label_date)
    f12 = f12_score(pred, label_date)
    all_score = 0.4 * f11 + 0.6 * f12
    return all_score, f11, f12

def get_threshold_and_metrice_score(pred, label_date):
    score = 0
    threshold = 0
    for i in np.arange(np.min(pred['pre']), np.max(pred['pre']), 0.01):
        pred['pre_label'] = np.where(pred['pre'] >= i, 1, 0)
        all_score, cate_score, shop_score  = f11_f12_weighted_score(pred, label_date)
        if all_score > score:
            score = all_score
            threshold = i
    print(f"best all score: {score}\nf11 score: {cate_score}\n f12 score: {shop_score}\n threshold: {threshold} \n sub samples: {len(pred[pred['pre_label']==1])}")
    return threshold


# def score(sub_df):
#     best_score = 0
#     for i in range(0.1, 1, 0.1):
#         sub_df[""] = sub_df.grouby(['user_id', 'cate'])['label'].max()
#
#         sub_df["pre_label"] = np.where(sub_df['pre'] > i, 1,0)
#
#
#     s1 = f_score_cate(predict, target)
#     s2 = f_score_shop(predict, target)
#     return 0.4 * f_score_cate + 0.6 * f_score_shop


if __name__ == "__main__":
    target_start = date(2018, 4, 9)
    target_end = date(2018, 4, 16)

    action_table = pd.read_hdf("../input/jdata_action.h5")
    action_table['action_time'] = action_table['action_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    product_table = pd.read_hdf("../input/jdata_product.h5")
    action_table = pd.merge(action_table, product_table, how="left", on="sku_id")
    action_table = action_table[action_table['type'] == 2]
    action_table = action_table[(action_table['action_time'] >= target_start)&(action_table['action_time'] < target_end)]

    action_table = action_table[['user_id', "cate", "shop_id"]].drop_duplicates()

    pre = pd.read_csv("../input/result/new_train_alll_pre.csv")
    threshold = get_threshold_and_metrice_score(pre, action_table)