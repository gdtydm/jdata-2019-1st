import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
from tools import bys_smooth, reduce_mem_usage
from config import feature_path, base_file_path
from functools import partial
from sklearn.preprocessing import minmax_scale


merge_feature = partial(pd.merge, on=['user_id', "shop_id"], how="left")

def str_to_day(df, column):
    df[column].fillna(df[column].mode().values[0], inplace=True)
    df[column] = df[column].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    return df


def get_statistic_info(df, groupby_columns, columns:str, name:str, agg_dict={}):
    default_agg_dict = {
        f"{name}_mean":np.mean,
        f"{name}_max":np.max,
        f"{name}_median":np.median,
        f"{name}_min":np.min,
        f"{name}_std":np.std
    }
    for k,v in agg_dict.items():
        default_agg_dict[f"{name}_{k}"] = v
    statistic_info = df.groupby(groupby_columns)[columns].agg(
        default_agg_dict
    ).reset_index()

    return statistic_info


def extract_user_shop_feature(target_day):
    print(f"start extract user_shop feature: {str(target_day)}")
    action = pd.read_hdf(base_file_path + "jdata_action.h5")
    action = str_to_day(action, "action_time")
    action = action[action['action_time'] < target_day]

    product = pd.read_hdf(base_file_path + "jdata_product.h5")
    product = str_to_day(product, "market_time")
    user_shop_action = pd.merge(action, product, how="left", on="sku_id")
    del product, action
    user_shop = user_shop_action[['user_id', "shop_id"]].drop_duplicates().dropna()
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    user_shop_len = len(user_shop)
    ######
    # 用户店铺购买次数
    user_shop_shopd_num = user_shop_action[user_shop_action['type'] == 2].groupby(['user_id', "shop_id"])['type'].count().reset_index()
    user_shop_shopd_num.columns = ['user_id', "shop_id", "user_shop_shop_nums"]
    user_shop = merge_feature(user_shop, user_shop_shopd_num)
    print(user_shop.head())
    user_shop['user_shop_shop_nums'].fillna(0, inplace=True)
    del user_shop_shopd_num
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户店铺交互次数
    user_shop_active_num = user_shop_action.groupby(['user_id', "shop_id"])['type'].count().reset_index()
    user_shop_active_num.columns = ['user_id', "shop_id", "user_shop_active_nums"]
    user_shop = merge_feature(user_shop, user_shop_active_num)
    user_shop['user_shop_active_nums'].fillna(0, inplace=True)
    del user_shop_active_num
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户购买天数
    user_shop_shop_day = user_shop_action[user_shop_action["type"] == 2].groupby(['user_id', "shop_id"])['action_time'].nunique().reset_index()
    user_shop_shop_day.columns = ['user_id', "shop_id", "user_shop_shop_day"]
    user_shop = merge_feature(user_shop, user_shop_shop_day)
    user_shop['user_shop_shop_day'].fillna(0, inplace=True)
    del user_shop_shop_day
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户交互天数
    user_shop_active_day = user_shop_action.groupby(['user_id', "shop_id"])['action_time'].nunique().reset_index()
    user_shop_active_day.columns = ['user_id', "shop_id", "user_shop_active_day"]
    user_shop = merge_feature(user_shop, user_shop_active_day)
    user_shop['user_shop_active_day'].fillna(0, inplace=True)
    del user_shop_active_day
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户店铺购买商品数
    user_shop_shop_sku_num = user_shop_action[user_shop_action["type"] == 2].groupby(['user_id', "shop_id"])['sku_id'].nunique().reset_index()
    user_shop_shop_sku_num.columns = ['user_id', "shop_id", "user_shop_shop_sku_num"]
    user_shop = merge_feature(user_shop, user_shop_shop_sku_num)
    user_shop['user_shop_shop_sku_num'].fillna(0, inplace=True)
    del user_shop_shop_sku_num
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户店铺交互商品数
    user_shop_active_sku_num = user_shop_action.groupby(['user_id', "shop_id"])['sku_id'].nunique().reset_index()
    user_shop_active_sku_num.columns = ['user_id', "shop_id", "user_shop_active_sku_num"]
    user_shop = merge_feature(user_shop, user_shop_active_sku_num)
    user_shop['user_shop_active_sku_num'].fillna(0, inplace=True)
    del user_shop_active_sku_num
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户查看购买率
    user_shop['user_shop_shop_sku_rate'] = user_shop['user_shop_shop_sku_num'] / (user_shop['user_shop_active_sku_num'] + 1e-5)
    user_shop['user_shop_shop_day_rate'] = user_shop['user_shop_shop_day'] / (user_shop['user_shop_active_day'] + 1e-5)
    user_shop['user_shop_shop_num_rate'] = user_shop['user_shop_shop_nums'] / (user_shop['user_shop_active_nums'] + 1e-5)

    # 用户店铺的购买间隔
    user_shop_shop_snap = user_shop_action[['user_id', "shop_id", "type", "action_time"]]
    user_shop_shop_snap = user_shop_shop_snap[user_shop_shop_snap['type'] == 2][['user_id', "shop_id", "action_time"]].drop_duplicates()
    user_shop_shop_snap.sort_values(by=['user_id',"shop_id", "action_time"], inplace=True)
    user_shop_shop_snap["before_shop_time"] = user_shop_shop_snap.groupby(['user_id', "shop_id"])['action_time'].shift(1)
    user_shop_shop_snap = user_shop_shop_snap[~(user_shop_shop_snap['before_shop_time'].isnull())]
    user_shop_shop_snap['user_shop_snap'] = (user_shop_shop_snap['action_time'] - user_shop_shop_snap['before_shop_time']).apply(lambda s:s.days)

    user_shop_shop_snap_statistic = get_statistic_info(user_shop_shop_snap, ['user_id', "shop_id"], 'user_shop_snap', name="user_shop_shop_snap", agg_dict={"_before2th":lambda s:s.tolist()[-2] if  len(s) >= 2 else -1, "_before1th":lambda s:s.tolist()[-1]})
    user_shop = merge_feature(user_shop, user_shop_shop_snap_statistic)
    del user_shop_shop_snap_statistic, user_shop_shop_snap
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户该在店铺买的商品的好评率,差评率。上一次购买商品的评价。统计特征。
    comment_info = pd.read_hdf(base_file_path + "jdata_comment.h5")
    comment_info_rate = comment_info.groupby("sku_id").sum().reset_index()
    comment_info_rate['good_comments'] = np.where(comment_info_rate['good_comments'] > comment_info_rate['comments'], comment_info_rate['comments'], comment_info_rate['good_comments'])
    good_comments_rate, alpha, beta = bys_smooth(comment_info_rate['good_comments'].values, comment_info_rate['comments'].values)
    comment_info_rate['good_comments_rate'] = good_comments_rate
    bad_comments_rate, alpha, beta = bys_smooth(comment_info_rate['bad_comments'].values, comment_info_rate['comments'].values)
    comment_info_rate['bad_comments_rate'] = bad_comments_rate

    user_shop_shop_sku_comments = user_shop_action[user_shop_action["type"] == 2][['user_id', "shop_id", "sku_id", "action_time"]]
    user_shop_shop_sku_comments = pd.merge(user_shop_shop_sku_comments, comment_info_rate, how="left", on="sku_id")[['user_id',"shop_id","sku_id", "good_comments_rate", "bad_comments_rate", "action_time"]]
    user_shop_shop_sku_comments.sort_values(by=['user_id',"shop_id","sku_id", "action_time"], inplace=True)
    user_shop_shop_sku_good_comment = get_statistic_info(user_shop_shop_sku_comments, ['user_id',"shop_id"], "good_comments_rate", name="user_shop_shop_sku_good_comment", agg_dict={"_before1th":lambda s:s.tolist()[-1]})
    user_shop_shop_sku_bad_comment = get_statistic_info(user_shop_shop_sku_comments, ['user_id',"shop_id"], "bad_comments_rate", name="user_shop_shop_sku_bad_comment", agg_dict={"_before1th":lambda s:s.tolist()[-1]})
    user_shop = merge_feature(user_shop, user_shop_shop_sku_good_comment)
    user_shop = merge_feature(user_shop, user_shop_shop_sku_bad_comment)
    del user_shop_shop_sku_comments, user_shop_shop_sku_good_comment, user_shop_shop_sku_bad_comment, comment_info_rate
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户在当前店铺下买的商品中最多复购次数
    user_shop_sku_max_shop_count = user_shop_action[user_shop_action['type'] == 2][["user_id", "shop_id", "sku_id", "action_time"]]
    user_shop_sku_max_shop_count = user_shop_sku_max_shop_count.groupby(['user_id', "shop_id", "sku_id"])['action_time'].nunique().reset_index()
    user_shop_sku_max_shop_count = user_shop_sku_max_shop_count.groupby(['user_id', "shop_id"])['action_time'].max().reset_index()
    user_shop_sku_max_shop_count.columns = ['user_id', "shop_id", "user_shop_shop_sku_max_count"]
    user_shop = merge_feature(user_shop, user_shop_sku_max_shop_count)
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 用户和店铺最后一次的行为距离目标日期天数
    user_shop_last_action_to_target = user_shop_action[['user_id', "shop_id", "type", "action_time"]]
    user_shop_last_action_to_target['user_shop_last_action_to_target'] = user_shop_last_action_to_target['action_time'].apply(lambda s :(target_day - s).days)
    for type_ in [1,2,3,4]:
        sub_user_shop_last_action_to_target = user_shop_last_action_to_target[user_shop_last_action_to_target['type'] == type_]
        sub_user_shop_last_action_to_target = sub_user_shop_last_action_to_target.groupby(['user_id', "shop_id"])['user_shop_last_action_to_target'].min().reset_index()
        sub_user_shop_last_action_to_target.columns = ['user_id', "shop_id", f"user_shop_last_action_{type_}_to_target"]
        user_shop = merge_feature(user_shop, sub_user_shop_last_action_to_target)
        assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 店铺购买比例占所有购买比例
    user_shop_count = user_shop_action[user_shop_action['type'] == 2][['user_id',"type"]]
    user_shop_count = user_shop_count.groupby("user_id")['type'].count().reset_index()
    user_shop_count.columns = ['user_id', "user_shop_num"]

    user_shop_shop_day = user_shop[['user_id',"shop_id", "user_shop_shop_nums"]]
    user_shop_shop_rate = pd.merge(user_shop_shop_day, user_shop_count, how="left", on="user_id")
    user_shop_shop_rate['user_shop_shop_rate'] = user_shop_shop_rate['user_shop_shop_nums'] / user_shop_shop_rate['user_shop_num']

    user_shop = merge_feature(user_shop, user_shop_shop_rate[['user_id', "shop_id","user_shop_shop_rate"]])
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    # 店铺在用户购买店铺中的排名
    user_shop_shop_day['user_shop_shop_rank'] = user_shop_shop_day.groupby(['user_id'])['user_shop_shop_nums'].apply(lambda s: s/np.sum(s))
    print(user_shop_shop_day.head())
    user_shop = merge_feature(user_shop, user_shop_shop_day[['user_id',"shop_id","user_shop_shop_rank"]])
    assert len(user_shop) == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    assert user_shop_len == len(user_shop[['user_id', "shop_id"]].drop_duplicates())
    print(user_shop.info())
    print(f"extract {str(target_day)} user shop action feature...")
    user_shop.to_hdf(feature_path + f"user_shop_feature_{str(target_day)}.h5", key='df', mode='w')
if __name__ == "__main__":

    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_user_shop_feature(i)








