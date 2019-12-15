import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
from tools import bys_smooth, reduce_mem_usage
from config import feature_path, base_file_path
'''
用户类别交互特征:
划窗特征：




'''


def user_cate_feature(date, days):
    user_cate = date[['user_id', 'cate']].drop_duplicates()
    user_cate_n = user_cate.shape[0]
    user_cate_type = date[['user_id', "cate", "type","sku_id", "shop_id", "action_time"]]

    # 用户类别商品数
    user_action_sku_id_num = user_cate_type.groupby(['user_id', "cate"])['sku_id'].nunique().reset_index()
    user_action_sku_id_num.columns = ['user_id', "cate", f"{days}_user_cate_sku_id_uniq_num"]
    user_cate = pd.merge(user_cate, user_action_sku_id_num, how="left", on=['user_id', "cate"])

    # 店铺数
    user_action_shop_id_num = user_cate_type.groupby(['user_id', "cate"])['shop_id'].nunique().reset_index()
    user_action_shop_id_num.columns = ['user_id', "cate", f"{days}_user_cate_shop_id_uniq_num"]
    user_cate = pd.merge(user_cate, user_action_shop_id_num, how="left", on=['user_id', "cate"])

    user_action_num = user_cate_type.groupby(['user_id', "cate"])['type'].count().reset_index()
    user_action_num.columns = ['user_id', "cate", f"{days}_user_cate_action_num"]
    user_cate = pd.merge(user_cate, user_action_num, how="left", on=['user_id', "cate"])


    for type_ in [1,2,3,4]:
        sub_date = user_cate_type[user_cate_type['type'] == type_]
        sub_date = sub_date.groupby(['user_id', "cate"])['sku_id'].count().reset_index()
        sub_date.columns = ['user_id', "cate", f"{days}_user_cate_{type_}_count"]
        user_cate = pd.merge(user_cate, sub_date, how="left", on=['user_id', "cate"])
        user_cate[f"{days}_user_cate_{type_}_rate"] = user_cate[f"{days}_user_cate_{type_}_count"] / user_cate[f"{days}_user_cate_action_num"]
        
        if type_ in [1,2]:
            sub_date = user_cate_type[user_cate_type['type'] == type_]
            sub_date = sub_date.groupby(['user_id', "cate"])['action_time'].nunique().reset_index()
            sub_date.columns = ['user_id', "cate", f"{days}_user_cate_{type_}_time_nuique"]
            user_cate = pd.merge(user_cate, sub_date, how="left", on=['user_id', "cate"])

    assert user_cate_n == user_cate.shape[0]
    return user_cate


def user_feature(date, days):
    user = date[['user_id']].drop_duplicates()
    user_n = user.shape[0]
    user_cate_type = date[['user_id', "cate", "type", "sku_id", "shop_id"]]

    # 用户类别数
    user_action_cate_num = user_cate_type.groupby(['user_id'])['cate'].nunique().reset_index()
    user_action_cate_num.columns = ['user_id', f"{days}_user_cate_uniq_num"]
    user = pd.merge(user, user_action_cate_num, how="left", on=['user_id'])

    # 用户店铺
    user_action_shop_id_num = user_cate_type.groupby(['user_id'])['shop_id'].nunique().reset_index()
    user_action_shop_id_num.columns = ['user_id', f"{days}_user_shop_id_uniq_num"]
    user = pd.merge(user, user_action_shop_id_num, how="left", on=['user_id'])

    # 用户商品
    user_action_sku_id_num = user_cate_type.groupby(['user_id'])['sku_id'].nunique().reset_index()
    user_action_sku_id_num.columns = ['user_id', f"{days}_user_sku_id_uniq_num"]
    user = pd.merge(user, user_action_sku_id_num, how="left", on=['user_id'])


    user_action_num = user_cate_type.groupby(['user_id'])['type'].count().reset_index()
    user_action_num.columns = ['user_id', f"{days}_user_action_num"]
    user = pd.merge(user, user_action_num, how="left", on=['user_id'])

    for type_ in [1,2,3,4]:
        sub_date = user_cate_type[user_cate_type['type'] == type_]
        sub_date =sub_date.groupby(['user_id'])['sku_id'].count().reset_index()
        sub_date.columns = ['user_id', f"{days}_user_{type_}_count"]
        user = pd.merge(user, sub_date, how="left", on=['user_id'])
        user[f"{days}_user_{type_}_rate"] = user[f"{days}_user_{type_}_count"] / user[f"{days}_user_action_num"]
    
    assert user_n == user.shape[0]
    return user

def cate_feature(date, days):
    cate = date[['cate']].drop_duplicates()
    cate_n = cate.shape[0]
    user_cate_type = date[['user_id', "cate", "type", "sku_id", "shop_id", 'market_time_to_target_day']]
    # 用户类别数
    cate_user_num = user_cate_type.groupby(['cate'])['user_id'].nunique().reset_index()
    cate_user_num.columns = ['cate', f"{days}_cate_user_uniq_num"]
    cate = pd.merge(cate, cate_user_num, how="left", on=['cate'])
    cate_user_num = user_cate_type.groupby(['cate'])['user_id'].count().reset_index()
    cate_user_num.columns = ['cate', f"{days}_cate_user_count_num"]
    cate = pd.merge(cate, cate_user_num, how="left", on=['cate'])    
    user_action_shop_id_num = user_cate_type.groupby(['cate'])['shop_id'].nunique().reset_index()
    user_action_shop_id_num.columns = ['cate', f"{days}_cate_shop_id_uniq_num"]
    cate = pd.merge(cate, user_action_shop_id_num, how="left", on=['cate'])
    user_action_sku_id_num = user_cate_type.groupby(['cate'])['sku_id'].nunique().reset_index()
    user_action_sku_id_num.columns = ['cate', f"{days}_cate_sku_id_uniq_num"]
    cate = pd.merge(cate, user_action_sku_id_num, how="left", on=['cate'])
    user_action_num = user_cate_type.groupby(['cate'])['type'].count().reset_index()
    user_action_num.columns = ['cate', f"{days}_cate_action_num"]
    cate = pd.merge(cate, user_action_num, how="left", on=['cate'])
    user_mark_num = user_cate_type[["cate", "sku_id","market_time_to_target_day"]].drop_duplicates(["cate", "sku_id"])
    user_mark_num = user_mark_num.groupby("cate")['market_time_to_target_day'].apply(lambda s: sum([1 if i < days else 0 for i in s])).reset_index()
    user_mark_num.columns = ["cate", f"{days}_cate_update_products"]
    cate = pd.merge(cate, user_mark_num[['cate', f"{days}_cate_update_products"]], how="left", on=['cate'])
    for type_ in [1,2,3,4]:
        sub_date = user_cate_type[user_cate_type['type'] == type_]
        sub_date = sub_date.groupby(['cate'])['sku_id'].count().reset_index()
        sub_date.columns = ['cate', f"{days}_cate_{type_}_count"]
        cate = pd.merge(cate, sub_date, how="left", on=['cate'])
        cate[f"{days}_cate_{type_}_rate"] = cate[f"{days}_cate_{type_}_count"] / cate[f"{days}_cate_action_num"]

        if type_ == 2:
            sub_date = user_cate_type[user_cate_type['type'] == type_]
            sub_date = sub_date.groupby(['cate', "sku_id"])['user_id'].count().reset_index()
            sub_date = sub_date.groupby("cate")['user_id'].max().reset_index()
            sub_date.columns = ["cate", f"{days}_cate_sku_id_max_shop_count"]
            cate = pd.merge(cate, sub_date[["cate",f"{days}_cate_sku_id_max_shop_count"]], how="left", on=['cate'])    
    assert cate_n == cate.shape[0]
    return cate



def move_window_extract_feature(date, target_days, windows):
    for days in windows:
        print(f"extract: {str(target_days)}_{days}")
        start_days = target_days - timedelta(days=days)
        sub_date = date[(date['action_time'] >= start_days) & (date['action_time'] < target_days)]
        user_table = user_feature(sub_date, days)
        user_table = reduce_mem_usage(user_table)
        user_table.to_hdf(feature_path + f"move_window_user_feature_{str(target_days)}_{days}.h5", key='df', mode='w')

        cate_table = cate_feature(sub_date, days)
        cate_table = reduce_mem_usage(cate_table)
        cate_table.to_hdf(feature_path + f"move_window_cate_feature_{str(target_days)}_{days}.h5", key='df', mode='w')

        user_cate_feature_ = user_cate_feature(sub_date, days)
        user_cate_feature_ = reduce_mem_usage(user_cate_feature_)
        user_cate_feature_.to_hdf(feature_path + f"move_window_user_cate_feature_{str(target_days)}_{days}.h5", key='df', mode='w')



def extract_user_cate_feature(target_date, windows=[1, 3, 7, 14, 30]):
    print("load action date")
    action = pd.read_hdf(base_file_path + "jdata_action.h5")
    shoped_user = action[action['type'] == 2]['user_id'].unique().tolist()
    print(f"购买过的用户数：{len(shoped_user)}")
    # action = action[action['user_id'].isin(shoped_user)]
    action['action_time'] = action['action_time'].apply(lambda s: date(*(int(i) for i in s.split(" ")[0].split("-"))))
    action = action[action['action_time'] < target_date]

    product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
    product_table.drop_duplicates(['sku_id'], inplace=True)
    product_table['market_time'] = product_table['market_time'].apply(
        lambda s: date(*(int(i) for i in s.split(" ")[0].split("-"))))
    product_table['market_time_to_target_day'] = (target_date - product_table['market_time']).apply(lambda s: s.days)
    action = pd.merge(action, product_table, how="left", on="sku_id")
    print("开始统计划窗特征")
    move_window_extract_feature(action, target_date, windows)

    print("###### 用户类别历史行为特征 #######")
    user_cate = action[['user_id', 'cate']].drop_duplicates()

    print("用户类别的平均购买时间")
    shoped_action = action[action['type'] == 2]
    shoped_action.sort_values(by=['user_id', 'cate', 'action_time'], inplace=True)
    shoped_action["before_shop_time"] = shoped_action.groupby(['user_id', 'cate'])['action_time'].shift(1)
    shoped_action = shoped_action[~(shoped_action['before_shop_time'].isnull())]
    shoped_action['diff'] = (shoped_action['action_time'] - shoped_action['before_shop_time']).apply(lambda s:s.days)

    user_cate_shop_span = shoped_action.groupby(['user_id', 'cate'])['diff'].agg({
        "user_cate_max_diff_day":np.max,
        "user_cate_min_diff_day":np.min,
        "user_cate_avg_diff_day":np.mean,
        "user_cate_median_diff_day":np.median,
        "user_cate_std_diff_day":np.std,
        "user_cate_before_diff_day": lambda s:s.tolist()[-1]
    }).reset_index()
    user_cate_shop_span.columns = ['user_id', 'cate',"user_cate_max_diff_day", "user_cate_min_diff_day", "user_cate_avg_diff_day", "user_cate_median_diff_day", "user_cate_std_diff_day" ,"user_cate_before_diff_day"]
    user_cate = pd.merge(user_cate, user_cate_shop_span, how="left", on=['user_id', 'cate'])
    del user_cate_shop_span

    # 去除同一天多次下单是情况，计算平均值
    print("类别平均下单间隔， 去除同一天")
    action.sort_values(by=['user_id', 'cate', 'action_time'], inplace=True)

    shoped_action = action[action['type'] == 2]
    shoped_action = shoped_action.drop_duplicates(['user_id', 'cate', 'action_time'])
    shoped_action["before_shop_time"] = shoped_action.groupby(['user_id', 'cate'])['action_time'].shift(1)
    shoped_action = shoped_action[~(shoped_action['before_shop_time'].isnull())]
    shoped_action['diff'] = (shoped_action['action_time'] - shoped_action['before_shop_time']).apply(lambda s:s.days)

    user_cate_shop_span = shoped_action.groupby(['user_id', 'cate'])['diff'].agg({
        "user_cate_min_diff_day": np.min,
        "user_cate_avg_diff_day": np.mean,
        "user_cate_median_diff_day": np.median,
        "user_cate_std_diff_day": np.std,
        "user_cate_before_diff_day": lambda s:s.tolist()[-1]
    }).reset_index()
    user_cate_shop_span.columns = ['user_id', 'cate', "user_cate_min_diff_day_del_same_day",
                                   "user_cate_avg_diff_day_del_same_day", "user_cate_median_diff_day_del_same_day", "user_cate_std_diff_day_del_same_day", "user_cate_before_diff_day_del_same_day"]
    user_cate = pd.merge(user_cate, user_cate_shop_span, how="left", on=['user_id', 'cate'])
    del user_cate_shop_span


    print("用户对某品类下某产品重复购买的次数")
    shop_action = action[action['type'] == 2]
    shop_sku_id_count = shop_action.groupby(['user_id', "sku_id"])['type'].count().reset_index()
    shop_sku_id_count.columns = ['user_id', "sku_id", "shop_sku_id_count"]
    shop_action = pd.merge(shop_action,shop_sku_id_count, how="left", on=['user_id', "sku_id"])
    shop_sku_id_count = shop_action.groupby(['user_id', "cate"])['shop_sku_id_count'].max().reset_index()
    shop_sku_id_count.columns = ['user_id', "cate", 'sku_id_max_shop_counts']
    user_cate = pd.merge(user_cate, shop_sku_id_count, how="left", on=['user_id', 'cate'])
    del shop_action, shop_sku_id_count
    

    print("用户对某品类下某产品重复购买的天数")
    shop_action = action[action['type'] == 2]
    shop_sku_id_count = shop_action.groupby(['user_id', "sku_id"])['action_time'].nunique().reset_index()
    shop_sku_id_count.columns = ['user_id', "sku_id", "shop_sku_id_nunique"]
    shop_action = pd.merge(shop_action,shop_sku_id_count, how="left", on=['user_id', "sku_id"])
    shop_sku_id_count = shop_action.groupby(['user_id', "cate"])['shop_sku_id_nunique'].max().reset_index()
    shop_sku_id_count.columns = ['user_id', "cate", 'sku_id_max_shop_nunique']
    user_cate = pd.merge(user_cate, shop_sku_id_count, how="left", on=['user_id', 'cate'])
    del shop_action, shop_sku_id_count
    
    print("提取用户该目标类别的最后一次行为时间 类别 各个行为的次数")
    # 用户该目标类别的最后一次行为时间 类别 各个行为的次数
    for type_ in [1,2,3,4]:
        action_ = action[action['type'] == type_]
        action_ = action_[['user_id', 'cate', 'action_time']]

        user_cate_type_count = action_.groupby(['user_id', 'cate'])["action_time"].count().reset_index()
        user_cate_type_count.columns = ['user_id', 'cate', f"user_cate_{type_}_count"]
        user_cate = pd.merge(user_cate, user_cate_type_count, how="left", on=['user_id', 'cate'])

        user_cate_type_unique = action_.groupby(['user_id', 'cate'])["action_time"].nunique().reset_index()
        user_cate_type_unique.columns = ['user_id', 'cate', f"user_cate_{type_}_nunique_days"]
        user_cate = pd.merge(user_cate, user_cate_type_unique, how="left", on=['user_id', 'cate'])

        user_cate[f'user_cate_{type_}_avg_day_counts'] = user_cate[f"user_cate_{type_}_count"] / user_cate[f"user_cate_{type_}_nunique_days"]

        action_ = action_.groupby(['user_id', 'cate']).tail(1).reset_index()
        action_[f'last_{type_}_to_now_days'] = action_['action_time'].apply(lambda s:(target_date - s).days)
        user_cate = pd.merge(user_cate, action_[["user_id", "cate", f'last_{type_}_to_now_days']], how="left", on=['user_id', 'cate'])

    user_cate['user_cate_watch_shop_rate'] = user_cate[f"user_cate_{2}_count"] / (user_cate[f"user_cate_{1}_count"] + user_cate[f"user_cate_{2}_count"])

    print("用户平均购物间隔　－　最后一次到目标区间的时间")
    user_cate["avg_diff_shop_day_substruct_last_shop_to_target"] = user_cate['user_cate_avg_diff_day'] - user_cate["last_2_to_now_days"]
    user_cate["avg_diff_shop_day_substruct_last_shop_to_target_del_same_shop_days"] = user_cate['user_cate_avg_diff_day_del_same_day'] - user_cate["last_2_to_now_days"]
    user_cate["median_diff_shop_day_substruct_last_shop_to_target"] = user_cate['user_cate_median_diff_day'] - user_cate["last_2_to_now_days"]
    user_cate["median_diff_shop_day_substruct_last_shop_to_target_del_same_shop_days"] = user_cate['user_cate_median_diff_day_del_same_day'] - user_cate["last_2_to_now_days"]
    user_cate['avg_diff_shop_day__'] = np.where((user_cate["avg_diff_shop_day_substruct_last_shop_to_target"] <=7) &(user_cate["avg_diff_shop_day_substruct_last_shop_to_target"] >=0), 1, 0)
    user_cate['avg_diff_shop_day__same'] = np.where((user_cate["avg_diff_shop_day_substruct_last_shop_to_target_del_same_shop_days"] <=7) &(user_cate["avg_diff_shop_day_substruct_last_shop_to_target_del_same_shop_days"] >=0), 1, 0)

    user_cate['median_diff_shop_day__'] = np.where((user_cate["median_diff_shop_day_substruct_last_shop_to_target"] <= 7) & (
                user_cate["median_diff_shop_day_substruct_last_shop_to_target"] >= 0), 1, 0)
    user_cate['median_diff_shop_day__same'] = np.where(
        (user_cate["median_diff_shop_day_substruct_last_shop_to_target_del_same_shop_days"] <= 7) & (
                    user_cate["median_diff_shop_day_substruct_last_shop_to_target_del_same_shop_days"] >= 0), 1, 0)
    user_cate = reduce_mem_usage(user_cate)
    print(f"extract {str(target_date)} user cate action feature...")
    user_cate.to_hdf(feature_path + f"user_cate_feature_{str(target_date)}.h5", key='df', mode='w')

if __name__ == "__main__":

    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_user_cate_feature(i)