import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from tools import bys_smooth, reduce_mem_usage
import os
from config import feature_path, base_file_path
"""
###  使用用户所有的历史action数据刻画用户画像  #### 

# 为了降低内存和提高加载速度，所以用了，preprocess.py 预处理，转化 csv 为 h5
特征如下：
用户购买的平均间隔天数
除去 3-27 or 3-28 后用户的平均购买间隔天数
用户最后一次购买时间距离 target_date 的天数
用户最后一次购买时间距离 target_date 的天数（去掉 3-27， 3-28）
用户最后一次行为时间距离 target_date 的天数
用户最后一次行为时间距离 target_date 的天数（去掉 3-27， 3-28）
用户的最后一次购物是否在 活动时间"(3-27， 3-28）
用户的最后一次行为是否在 活动时间"(3-27， 3-28）
用户交互过的商品数
用户购买过的商品数
用户的活动天数
用户购买天数
用户有交互的商品的平均好评率
用户购买的商品的平均好评率
用户购买的cate类别数
用户交互的cate类别数
用户购买的品牌数
用户购买的店铺数
用户交互的店铺数
用户购买率（贝叶斯平滑）
用户平均时间 - 用户最后一次交互距离目标区间的时间。
用户购买天数/用户活动天数

其他特征：未添加

用户最喜欢的类别（交互次数最多 or 交互次数最多）前三名
用户最喜欢的店铺（购买次数最多 or 交互次数最多）前三名

"""

# 用户平均购物间隔 以及上一次购买时间间隔
def avg_shop_snap_day(df, sdf, columns, name=[]):
    avg_shop_snap_day = sdf[sdf['type'] == 2]
    avg_shop_snap_day['next_action_time'] = avg_shop_snap_day.groupby(columns)["action_time"].shift(1)
    avg_shop_snap_day = avg_shop_snap_day[~(avg_shop_snap_day['next_action_time'].isnull())]
    avg_shop_snap_day['diff_shop'] = (avg_shop_snap_day['action_time'] - avg_shop_snap_day['next_action_time']).apply(lambda s:s.days)
    if not name:
        name = ["_".join(columns) + "shop_snap_days"]
    avg_shop_snap_day = avg_shop_snap_day.groupby(columns)['diff_shop'].agg({
        f"{name[0]}_min":np.min,
        f"{name[0]}_max":np.max,
        f"{name[0]}_avg":np.mean,
        f"{name[0]}_std": "std",
        f"{name[0]}_median":np.median,
        f"{name[0]}_before":lambda s:s.tolist()[-1]
    }).reset_index()
    print(avg_shop_snap_day.head())
    df = pd.merge(df, avg_shop_snap_day, how="left", on=columns)
    return df

# 用户上一次购物距离现在的时间
def last_shop_to_now(df, sdf, now_time, columns, name=[]):
    sdf = sdf[sdf['type'] == 2]
    last_shop = sdf.groupby("user_id")['action_time'].max().reset_index()
    last_shop['action_time'] = last_shop['action_time'].apply(lambda s: (now_time - s).days)
    if not name:
        name = ["_".join(columns) + "last_shop_time_to_now"]
    last_shop.columns = columns + name
    df = pd.merge(df, last_shop, how="left", on=columns)
    df.fillna({name[0]:999},inplace=True)
    return df

def last_shop_is_special_day(df, sdf, now_time, columns, name=[]):
    sdf = sdf[sdf['type'] == 2]
    last_shop = sdf.groupby("user_id")['action_time'].max().reset_index()
    last_shop['action_time'] = last_shop['action_time'].apply(lambda s: 1 if s in [date(2018, 3, 27), date(2018, 3, 28)] else 1)
    if not name:
        name = ["_".join(columns) + "last_shop_is_special_day"]
    last_shop.columns = columns + name
    df = pd.merge(df, last_shop, how="left", on=columns)
    df.fillna({name[0]:-1},inplace=True)
    return df


def last_action_to_now(df, sdf, now_time, columns, name=[]):
    last_shop = sdf.groupby("user_id")['action_time'].max().reset_index()
    last_shop['action_time'] = last_shop['action_time'].apply(lambda s: (now_time - s).days)
    if not name:
        name =  ["_".join(columns) + "last_action_time_to_now"]

    last_shop.columns = columns + name
    df = pd.merge(df, last_shop, how="left", on=columns)
    df.fillna({name[0]:999},inplace=True)
    return df


def last_action_is_special_day(df, sdf, now_time, columns, name=[]):
    last_shop = sdf.groupby("user_id")['action_time'].max().reset_index()
    last_shop['action_time'] = last_shop['action_time'].apply(lambda s: 1 if s in [date(2018, 3, 27), date(2018, 3, 28)] else 1)
    if not name:
        name = ["_".join(columns) + "last_action_is_special_day"]
    last_shop.columns = columns + name
    df = pd.merge(df, last_shop, how="left", on=columns)
    df.fillna({name[0]:-1},inplace=True)
    return df

# 用户唯一值数
def user_nunique(df, sdf, columns, value, name=[]):
    nunique = sdf.groupby(columns)[value].nunique().reset_index()
    if not name:
        name = ["_".join(columns) + f"{value}_nuniques"]
    nunique.columns = columns + name
    df = pd.merge(df, nunique, how="left", on=columns)
    df.fillna({name[0]:0}, inplace=True)
    return df



# 用户浏览，购买比率 平滑
def watch_shop_rate(df, sdf, columns , name=[]):
    watch = sdf[sdf['type'] == 1]
    watch = watch.groupby(columns)["type"].count().reset_index()
    watch.columns = columns + ['watch']
    shop = sdf[sdf['type'] == 2]
    shop = shop.groupby(columns)["type"].count().reset_index()
    shop.columns = columns + ['shop']
    shop_watch_rate = pd.merge(watch, shop, on=columns)
    I = (shop_watch_rate["watch"] + shop_watch_rate['shop']).values
    C = shop_watch_rate['shop'].values
    rate, alpha, beta = bys_smooth(I,C)
    if name:
        pass
    else:
        name = ["_".join(columns) + "_watch_shop_rate"]
    shop_watch_rate[name[0]] = rate
    shop_watch_rate = shop_watch_rate[columns + name]
    df = pd.merge(df, shop_watch_rate, how="left", on=columns)
    df.fillna({name[0]:alpha/(alpha + beta)}, inplace=True)
    return df

# 好评率，填充均值
def user_shop_avg_good_rate(df, sdf, columns, value, name=[]):
    comment_info = pd.read_hdf(base_file_path + "jdata_comment.h5")
    comment_info = comment_info[['sku_id', 'comments', 'good_comments', 'bad_comments']]
    comment_info = comment_info.groupby("sku_id", as_index=False).sum()
    I = comment_info["comments"]
    C = comment_info['good_comments']
    rate, alpha, beta = bys_smooth(I,C)
    comment_info['good_rate'] = rate
    sdf = pd.merge(sdf, comment_info[["sku_id",'good_rate']], how="left", on="sku_id")

    user_shop_avg_good_rate = sdf.groupby(columns)["good_rate"].mean().reset_index()
    if name:
        pass
    else:
        name = ["_".join(columns) + "_shop_good_rate"]
    user_shop_avg_good_rate.columns = columns + name
    df = pd.merge(df, user_shop_avg_good_rate, how="left", on=columns)
    df.fillna({name[0]: alpha/ (alpha + beta) }, inplace=True)
    return df



def extract_user_action_feature(target_date):
# 取 target_date 之前的时间作为特征提取区间, 也可以自定义特征提取时间区间。
    action = pd.read_hdf(base_file_path + "jdata_action.h5")

    shoped_user = action[action['type'] == 2]['user_id'].unique().tolist()
    print(f"购买过的用户数：{len(shoped_user)}")
    # action = action[action['user_id'].isin(shoped_user)]

    action['action_time'] = action['action_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    action = action[action['action_time'] < target_date]
    print("时间排序... ")
    action = action.sort_values(by=['user_id', "action_time"])

    # target_date 之前出现过的用户，的用户表。作为join 用户特征的主表
    user = action[['user_id']].drop_duplicates()
    #除去 3-27 or 3-28 后用户的平均购买间隔天数
    print("除去 3-27 or 3-28 后用户的平均购买间隔天数")
    user = avg_shop_snap_day(user, action[~(action['action_time'].isin([date(2018, 3, 27), date(2018, 3, 28)]))], ['user_id'], name=['del_special_days_avg_shop'])
    print("用户的平均购买间隔天数")
    user = avg_shop_snap_day(user, action, ['user_id'])
    # 用户最后一次购买时间距离 target_date 的天数
    print("用户最后一次购买时间距离 target_date 的天数")
    user = last_shop_to_now(user, action, target_date, ['user_id'], )

    # 用户的最后一次是否在 活动时间 3-27 or 3-28
    print("用户的最后一次是否在 活动时间 3-27 or 3-28")
    user = last_action_is_special_day(user,action,target_date,['user_id'])  
    print("用户最后一次行为时间距离 target_date 的天数")
    # 用户最后一次行为时间距离 target_date 的天数
    user = last_action_to_now(user, action, target_date,['user_id'])
    print("用户的最后一次购物是否在 活动时间")
    # 用户的最后一次购物是否在 活动时间
    user = last_shop_is_special_day(user, action,target_date, ['user_id'])
    print("除去活动时间， 最后一次行为距离现在的天数")
    # 除去活动时间， 最后一次行为距离现在的天数
    user = last_action_to_now(user, action[~(action['action_time'].isin([date(2018, 3, 27), date(2018, 3, 28)]))], target_date,['user_id'], name=["del_special_days_last_action_to_now"])
    print("除去活动时间， 最后一次购物距离现在的天数")    
    # 除去活动时间， 最后一次购物距离现在的天数
    user = last_shop_to_now(user, action[~(action['action_time'].isin([date(2018, 3, 27), date(2018, 3, 28)]))], target_date,['user_id'], name=["del_special_days_last_shop_to_now"])
    print("用户有过交互的商品数")
    # 用户有过交互的商品数
    user = user_nunique(user, action, ['user_id'], "sku_id")
    print("用户购买过的商品数")
    # 用户购买过的商品数
    user = user_nunique(user, action[action['type']==2], ['user_id'], "sku_id", ["shop_user_sku_id_nunique"])
    print("用户活动天数")
    # 用户活动天数
    user = user_nunique(user, action, ['user_id'], "action_time", ['active_user_time_nunique'])
    print("用户购买天数")
    # 用户购买天数
    user = user_nunique(user, action[action['type']==2], ['user_id'], "action_time", ["shop_user_action_time_nunique"])

    print("用户有交互的商品的平均好评率")
    # 用户有交互的商品的平均好评率
    user = user_shop_avg_good_rate(user, action, ["user_id"], [])
    print("用户购买的商品的平均好评率")
    # 用户购买的商品的平均好评率
    user = user_shop_avg_good_rate(user, action[action['type'] == 2], ["user_id"], [], name=['shop_good_rate'])

    product_info = pd.read_hdf(base_file_path + "jdata_product.h5")



    # product_info['market_time'].fillna("2000-11-03 10:45:58.0", inplace=True)
    product_info['market_time'].fillna(product_info['market_time'].mode().values[0], inplace=True)
    product_info['market_time'] = product_info['market_time'].apply(
        lambda s: date(*(int(i) for i in s.split(" ")[0].split("-"))))
    action = pd.merge(action, product_info, how="left", on="sku_id")
    del product_info

    print("用户购买的 的cate类别数")
    # 用户购买的 的cate类别数
    user = user_nunique(user, action[action['type']==2], ['user_id'], "cate", ["shop_user_cate_nunique"])
    print("用户有交互的 cate 的类别数")
    # 用户有交互的 cate 的类别数
    user = user_nunique(user, action, ['user_id'], "cate", ["action_user_cate_nunique"])
    print("用户购买的品牌数")
    # 用户购买的品牌数
    user = user_nunique(user, action[action['type']==2], ['user_id'], "brand", ["shop_user_brand_nunique"])
    print("用户交互的品牌数")
    # 用户交互的品牌数
    user = user_nunique(user, action, ['user_id'], "brand", ["action_user_brand_nunique"])
    print("用户交购买铺数")
    # 用户交购买铺数
    user = user_nunique(user, action[action['type']==2], ['user_id'], "shop_id", ["shop_user_shop_id_nunique"])
    print("用户交互店铺数")
    # 用户交互店铺数
    user = user_nunique(user, action, ['user_id'], "shop_id", ["shop_id_user_brand_nunique"])


    print("用户的 购买率（shop/watch） 贝叶斯平滑")
    # 用户的 购买率（shop/watch） 贝叶斯平湖
    user = watch_shop_rate(user, action, ['user_id'])

    shop_info = pd.read_hdf(base_file_path + "jdata_shop.h5")
    shop_info.rename(columns={"cate":"shop_cate"}, inplace=True)
    action = pd.merge(action, shop_info, how="left", on="shop_id")
    action['cate_is_shop_cate'] = action['cate'] == action['shop_cate']
    action['cate_is_shop_cate'] = action['cate_is_shop_cate'].astype("int")

    print("用户购买的商品 类别 是 店铺的主营商品 的概率")
    # 用户购买的商品 类别 是 店铺的主营商品 的概率
    sub = action[action['type'] == 2][['user_id','cate_is_shop_cate']]
    sub = sub.groupby("user_id")["cate_is_shop_cate"].agg({"counts":"count", "sums":"sum"}).reset_index()
    sub['cate_is_shop_cate_rate'] = sub['sums'] / (sub['counts'] + 1)
    user = pd.merge(user, sub[["user_id", 'cate_is_shop_cate_rate']], how="left", on="user_id")
    user.fillna({"cate_is_shop_cate_rate":0}, inplace=True)

    print("用户购买的商品平均发布时间， 代表用户喜欢产品的新旧程度，有些类别的产品喜欢新品例如电子产品，但是日用品又不在乎这些。")
    # 用户购买的商品平均发布时间， 代表用户喜欢产品的新旧程度，有些类别的产品喜欢新品例如电子产品，但是日用品又不在乎这些。
    user_shop_diff_market = action[action['type'] == 2][['user_id', "action_time", "market_time"]]

    user_shop_diff_market['user_shop_diff_market'] = (
        user_shop_diff_market['action_time'] - user_shop_diff_market['market_time']).apply(lambda s: s.days)

    user_shop_diff_market = user_shop_diff_market[['user_id', "user_shop_diff_market"]].groupby("user_id")["user_shop_diff_market"].agg({
        "user_shop_diff_market_avg": np.mean,
        "user_shop_diff_market_median": np.median,
        "user_shop_diff_market_max": np.max,
        "user_shop_diff_market_min": np.min,
        "user_shop_diff_market_std": np.std,
        "user_shop_diff_market_count":"count"
    }).reset_index()

    user = pd.merge(user, user_shop_diff_market, how="left", on="user_id")
    user.fillna({"user_shop_diff_market_count":0}, inplace=True)

    print("平均购买时间 - 用户最后一次购买距离target 的时间")
    # 平均购买时间 - 用户最后一次购买距离target 的时间
    user['next_shop_day'] = user["user_idshop_snap_days_avg"] -  user["_".join(["user_id"]) + "last_shop_time_to_now"] 
    user['next_shop_day_remove_special_day'] = user["del_special_days_avg_shop_avg"] - user["del_special_days_last_shop_to_now"]
    user['next_shop_day_remove_special_day_in_7'] = np.where((user['next_shop_day_remove_special_day'] <= 7)&(user['next_shop_day_remove_special_day'] >= 0),1,0)
    
    user["shop_active_rate"] = user['shop_user_action_time_nunique'] / user['active_user_time_nunique']

    user.to_hdf(feature_path + f'user_feature_{str(target_date)}.h5', key='df', mode='w')

if __name__ == "__main__":
    # 截止时间（<） 不取target_date当天

    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_user_action_feature(i)














