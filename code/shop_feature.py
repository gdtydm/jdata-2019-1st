import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from tools import bys_smooth, reduce_mem_usage
from config import feature_path, base_file_path
from sklearn.preprocessing import minmax_scale
import os
import gc

"""
施工中..
"""


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


def extract_shop_info(target_day):
    print(f"start extract {str(target_day)} shop feature...")
    # if os.path.exists(feature_path + f"shop_info_{str(target_day)}.h5"):
    #     return pd.read_hdf(feature_path + f"shop_info_{str(target_day)}.h5")
    shop_info = pd.read_hdf(base_file_path + "jdata_shop.h5")
    shop_info.rename(columns={"cate":"shop_cate"}, inplace=True)
    shop_info = shop_info.drop_duplicates()

    ###
    ## 开店距离目标时间。
    shop_info['shop_reg_tm'].fillna(shop_info['shop_reg_tm'].mode().values[0], inplace=True)
    shop_info['shop_reg_tm'] = shop_info['shop_reg_tm'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    shop_info['shop_reg_tm_to_target'] = shop_info["shop_reg_tm"].apply(lambda s: (target_day - s).days)

    ## 店铺涨粉速度
    shop_info['fans_add_speed'] = shop_info['fans_num'] / shop_info["shop_reg_tm_to_target"]
    shop_info['vip_add_speed'] = shop_info['vip_num'] / shop_info['shop_reg_tm_to_target']
    
    ## 店铺类别是否缺失
    shop_info["shop_cate_isnull"] = np.where(shop_info["shop_cate"].isnull(), 1, 0)
    
    # 店铺的商家开了几家店铺。
    vender_id_n_shop = shop_info.groupby("vender_id")['shop_id'].count().reset_index()
    vender_id_n_shop.columns = ['vender_id', "vender_id_n_shop"]

    shop_info = pd.merge(shop_info, vender_id_n_shop, how="left", on="vender_id")

    assert shop_info['shop_id'].nunique() == len(shop_info)

    shop = shop_info[["shop_id", "shop_cate", "fans_num", "shop_score", "vip_num", "fans_add_speed","vip_add_speed", "shop_reg_tm_to_target", "shop_cate_isnull", "vender_id_n_shop"]]
    shop.dropna(subset=["shop_id"], inplace=True)
    del vender_id_n_shop, shop_info

    shop_num = len(shop)

    ## 店铺商品数
    product_info = pd.read_hdf(base_file_path + "jdata_product.h5")
    product_info = product_info.drop_duplicates(subset=["sku_id"])
    shop_nunique_sku = product_info.groupby("shop_id")['sku_id'].nunique().reset_index()
    shop_nunique_sku.columns = ['shop_id', "shop_nunique_sku"]
    shop = pd.merge(shop, shop_nunique_sku, how="left", on="shop_id")
    del shop_nunique_sku

    ## 商品数/运营时间
    shop['update_sku_freq'] = shop['shop_nunique_sku'] / shop['shop_reg_tm_to_target']

    ## 店铺品牌数
    shop_nunique_brand = product_info.groupby("shop_id")['brand'].nunique().reset_index()
    shop_nunique_brand.columns = ['shop_id', "shop_nunique_brand"]
    shop = pd.merge(shop, shop_nunique_brand, how="left", on="shop_id")
    del shop_nunique_brand

    ## 店铺用户商品类别数
    shop_nunique_cate = product_info.groupby("shop_id")['cate'].nunique().reset_index()
    shop_nunique_cate.columns = ['shop_id', "shop_nunique_cate"]
    shop = pd.merge(shop, shop_nunique_cate, how="left", on="shop_id")
    del shop_nunique_cate

    ## 店铺商品上市到目标区间天数的统计值
    product_info['market_time'] = product_info['market_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    product_info['market_to_target_day'] = product_info['market_time'].apply(lambda s:(target_day - s).days)

    shop_sku_statistic_info = get_statistic_info(product_info, "shop_id", "market_to_target_day", name="shop_sku_market_to_target", agg_dict={"to_tail2th": lambda s: sorted(s.tolist())[1] if len(s.tolist()) >= 2 else -1})
    # shop_sku_statistic_info = product_info.groupby("shop_id")['market_to_target_day'].agg({
    #     "shop_sku_avg_market_to_target":np.mean,
    #     "shop_sku_median_market_to_target":np.median,
    #     "shop_sku_max_market_to_target":np.max,
    #     "shop_sku_min_market_to_target":np.min,
    #     "shop_sku_std_market_to_target":np.std,
        
    # }).reset_index()
    shop = pd.merge(shop, shop_sku_statistic_info, how="left", on="shop_id")
    del shop_sku_statistic_info
    
    # 用户目标期间第一周发售新品数
    shop_before_7day_update_sku_num = product_info.groupby("shop_id")['market_to_target_day'].apply(lambda s:sum([1 if i<=7 else 0 for i in s])).reset_index()
    shop_before_7day_update_sku_num.columns = ['shop_id', "shop_before_7day_update_sku_num"]

    # 用户目标期间第二周发售新品数
    shop_before_7_14day_update_sku_num = product_info.groupby("shop_id")['market_to_target_day'].apply(lambda s:sum([1 if ((i>7) & (i<=14)) else 0 for i in s])).reset_index()
    shop_before_7_14day_update_sku_num.columns = ['shop_id', "shop_before_7_14day_update_sku_num"]

    #　用户目标期间前２周发售新品数
    shop = pd.merge(shop, shop_before_7day_update_sku_num, how="left", on="shop_id")
    shop = pd.merge(shop, shop_before_7_14day_update_sku_num, how="left", on="shop_id")
    shop['shop_before_14day_update_sku_num'] = shop['shop_before_7day_update_sku_num'] + shop['shop_before_7_14day_update_sku_num']

    ## 评论数据有些奇怪。算好评率直接更具评论加和（贝叶斯平滑）。好评数。取评论数最大值的时候的值,
    comment_info = pd.read_hdf(base_file_path + "jdata_comment.h5")
    comment_info_rate = comment_info.groupby("sku_id").sum().reset_index()
    comment_info_rate['good_comments'] = np.where(comment_info_rate['good_comments'] > comment_info_rate['comments'], comment_info_rate['comments'], comment_info_rate['good_comments'])
    print(comment_info_rate.head())
    good_comments_rate, alpha, beta = bys_smooth(comment_info_rate['good_comments'].values, comment_info_rate['comments'].values)
    comment_info_rate['good_comments_rate'] = good_comments_rate
    bad_comments_rate, alpha, beta = bys_smooth(comment_info_rate['bad_comments'].values, comment_info_rate['comments'].values)
    comment_info_rate['bad_comments_rate'] = bad_comments_rate

    comment_info.sort_values(by=["comments"], ascending=False, inplace=True)
    comment_info = comment_info.drop_duplicates(subset=['sku_id'])


    comment_info = pd.merge(comment_info, comment_info_rate[['sku_id', "good_comments_rate", "bad_comments_rate"]], how="left", on="sku_id")

    comment_info['comments_rank'] = minmax_scale(comment_info['comments'].rank())

    product_comment_info = pd.merge(product_info, comment_info, how="inner", on="sku_id")
    del comment_info

    ### 店铺商品数，好评率统计特征
    shop_sku_good_comments_rate_statistic = get_statistic_info(product_comment_info, "shop_id", "good_comments_rate", "shop_sku_good_comments_rate")

    shop = pd.merge(shop, shop_sku_good_comments_rate_statistic, how="left", on="shop_id")
    del shop_sku_good_comments_rate_statistic

    ### 店铺商品数，评论数 rank 统计特征
    shop_sku_comments_rank_statistic = get_statistic_info(product_comment_info, "shop_id", "comments_rank", "shop_sku_comments_rank")
    shop = pd.merge(shop, shop_sku_comments_rank_statistic, how="left", on="shop_id")
    del shop_sku_comments_rank_statistic, product_info

    ### 商品评论数 / 商品上市时长。
    product_comment_info["sku_comments_add_rate"] = product_comment_info['comments'] / product_comment_info["market_to_target_day"]
    shop_sku_comments_statistic_info = get_statistic_info(product_comment_info, "shop_id", "sku_comments_add_rate", "shop_sku_comments_add_rate")
    shop = pd.merge(shop, shop_sku_comments_statistic_info, how="left", on="shop_id")
    del shop_sku_comments_statistic_info


    ### 用户和行为和shop_id ###
    action = pd.read_hdf(base_file_path + "jdata_action.h5")
    action['action_time'] = action['action_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    print(
        f"获取目标区间 {str(target_day)}　之前的行为数据。"
    )
    action = action[action['action_time'] < target_day]
    action_product_comment_info = pd.merge(action, product_comment_info, how="left", on="sku_id")
    del action
    




    ### 店铺复购率, 购买一次是用户数， 购买多次的用户数, 购买大于等于3次的用户数， 大于等于3次用户数占比
    shop_repeat_shop_rate = action_product_comment_info[action_product_comment_info['type'] == 2].groupby(["user_id", "shop_id"])['type'].count().reset_index()
    shop_user_num = shop_repeat_shop_rate.groupby("shop_id")['user_id'].nunique().reset_index()
    shop_user_num.columns = ['shop_id', "shop_user_num"]

    shop_once_user_num = shop_repeat_shop_rate[shop_repeat_shop_rate['type'] == 1].groupby("shop_id")['user_id'].nunique().reset_index()
    shop_once_user_num.columns = ['shop_id', "shop_once_user_num"]

    shop_n_count_user_num = shop_repeat_shop_rate[shop_repeat_shop_rate['type'] > 1].groupby("shop_id")['user_id'].count().reset_index()
    shop_n_count_user_num.columns = ['shop_id', "shop_n_count_user_num"]


    shop_over_3_count_user_num = shop_repeat_shop_rate[shop_repeat_shop_rate['type'] >= 3].groupby("shop_id")['user_id'].count().reset_index()
    shop_over_3_count_user_num.columns = ['shop_id', "shop_over_3_count_user_num"]

    shop_re_shop = pd.merge(shop_user_num, shop_n_count_user_num, how="left", on="shop_id")
    shop_re_shop = pd.merge(shop_re_shop, shop_over_3_count_user_num, how="left", on="shop_id")
    shop_re_shop = pd.merge(shop_re_shop, shop_once_user_num, how="left", on="shop_id")
    
    shop_re_shop.fillna(0, inplace=True)
    shop_re_shop["re_shop_rate"] = shop_re_shop['shop_n_count_user_num'] / (shop_re_shop['shop_user_num'] + 1e-5)
    shop_re_shop['over_3_shop_rate'] = shop_re_shop['shop_over_3_count_user_num'] / (shop_re_shop['shop_user_num'] + 1e-5)
    shop_re_shop['shop_once_rate'] = shop_re_shop['shop_once_user_num'] /  (shop_re_shop['shop_user_num'] + 1e-5)

    # rank 特征
    for column in shop_re_shop.columns:
        if column == "user_id":
            continue
        shop_re_shop["{}_rank".format(column)] = minmax_scale(shop_re_shop[column].rank())

    shop = pd.merge(shop, shop_re_shop, how="left", on="shop_id")
    del shop_re_shop, shop_over_3_count_user_num, shop_n_count_user_num, shop_once_user_num, shop_user_num, shop_repeat_shop_rate
    gc.collect()
    
    # 用户 浏览购买率。 
    user_watch_num = action_product_comment_info[action_product_comment_info["type"] ==1 ].groupby(['user_id', "shop_id"])['type'].count().reset_index()
    user_watch_num.columns = ['user_id', "shop_id", "watch_num"]
    user_shop_num = action_product_comment_info[action_product_comment_info['type'] == 2].groupby(['user_id', "shop_id"])['type'].count().reset_index()
    user_shop_num.columns = ['user_id', "shop_id", "shop_num"]

    user_watch_shop_rate = pd.merge(user_watch_num,user_shop_num, how="left", on=['user_id', "shop_id"])
    user_watch_shop_rate.dropna(inplace=True)
    user_watch_shop_rate['watch_num'] = np.where(user_watch_shop_rate['shop_num'] > user_watch_shop_rate['watch_num'], user_watch_shop_rate['shop_num'],user_watch_shop_rate['watch_num'] )

    watch_shop_rate_bys, alpha, beta = bys_smooth(user_watch_shop_rate['watch_num'].values, user_watch_shop_rate['shop_num'].values)
    user_watch_shop_rate['watch_shop_rate_bys'] = watch_shop_rate_bys
    user_watch_shop_rate['watch_shop_rate'] = user_watch_shop_rate['shop_num'] / user_watch_shop_rate['watch_num']

    shop_shop_rate_bys = get_statistic_info(user_watch_shop_rate, 'shop_id', "watch_shop_rate_bys", "shop_watch_shop_rate_bys")
    shop_shop_rate = get_statistic_info(user_watch_shop_rate, 'shop_id', "watch_shop_rate", "shop_watch_shop_rate")

    shop = pd.merge(shop, shop_shop_rate_bys, how="left", on="shop_id")
    shop = pd.merge(shop, shop_shop_rate, how="left", on="shop_id")

    del shop_shop_rate_bys, shop_shop_rate, watch_shop_rate_bys, user_watch_num

    # 浏览购买天数比
    user_watch_days = action_product_comment_info[action_product_comment_info["type"] ==1 ].groupby(['user_id', "shop_id"])['action_time'].nunique().reset_index()
    user_watch_days.columns = ['user_id', "shop_id", "watch_days"]
    user_shop_days = action_product_comment_info[action_product_comment_info['type'] == 2].groupby(['user_id', "shop_id"])['action_time'].nunique().reset_index()
    user_shop_days.columns = ['user_id', "shop_id", "shop_days"]

    user_watch_shop_days_rate = pd.merge(user_watch_days,user_shop_days, how="left", on=['user_id', "shop_id"])
    user_watch_shop_days_rate.dropna(inplace=True)
    user_watch_shop_days_rate['watch_days'] = np.where(user_watch_shop_days_rate['shop_days'] > user_watch_shop_days_rate['watch_days'], user_watch_shop_days_rate['shop_days'],user_watch_shop_days_rate['watch_days'] )

    watch_shop_days_rate_bys, alpha, beta = bys_smooth(user_watch_shop_days_rate['watch_days'].values, user_watch_shop_days_rate['shop_days'].values)
    user_watch_shop_days_rate['watch_shop_days_rate_bys'] = watch_shop_days_rate_bys
    user_watch_shop_days_rate['watch_shop_day_rate'] = user_watch_shop_days_rate['shop_days'] / user_watch_shop_days_rate['watch_days']

    shop_shop_days_rate_bys = get_statistic_info(user_watch_shop_days_rate, 'shop_id', "watch_shop_days_rate_bys", "shop_watch_shop_days_rate_bys")
    shop_shop_days_rate = get_statistic_info(user_watch_shop_days_rate, 'shop_id', "watch_shop_day_rate", "shop_watch_shop_days_rate")

    shop = pd.merge(shop, shop_shop_days_rate_bys, how="left", on="shop_id")
    shop = pd.merge(shop, shop_shop_days_rate, how="left", on="shop_id")

    del shop_shop_days_rate_bys, shop_shop_days_rate, watch_shop_days_rate_bys, user_shop_days, user_watch_days
    gc.collect()


    #店铺每个行为数的ｒａｎｋ，每个行为数的占比。
    shop_action_num = action_product_comment_info.groupby("shop_id")['type'].count().reset_index()
    shop_action_num.columns = ['shop_id', "shop_action_num"]
    shop_action_num['shop_action_num_rank'] = minmax_scale(shop_action_num['shop_action_num'].rank())
    shop = pd.merge(shop, shop_action_num, how="left", on="shop_id")
    del shop_action_num

    sub_df = action_product_comment_info[["shop_id", "user_id"]]
    base_user_info = pd.read_hdf(base_file_path + "jdata_user.h5")
    base_user_info.fillna(-1, inplace=True)
    sub_df = pd.merge(sub_df, base_user_info[["user_id",'sex', "age", "user_lv_cd", "city_level"]], how="left", on="user_id")
    for column in ['sex', "age", "city_level", "user_lv_cd"]:
        column_one_hot = pd.get_dummies(sub_df[column], prefix=f"shop_{column}")
        column_one_hot = pd.concat([sub_df[['shop_id']], column_one_hot], axis=1)
        column_one_hot = column_one_hot.groupby("shop_id").sum().reset_index()
        column_one_hot_ = column_one_hot.drop("shop_id",axis=1).apply(lambda s:s/np.sum(s),axis=1)
        column_one_hot = pd.concat([column_one_hot[['shop_id']], column_one_hot_], axis=1)
        shop = pd.merge(shop, column_one_hot, how="left", on="shop_id")

    for i in [1,2,3,4]:
        sub_df = action_product_comment_info[action_product_comment_info['type'] == i]
        sub_df = sub_df.groupby("shop_id")["type"].count().reset_index()
        sub_df.columns = ['shop_id', f"shop_{i}_action_num"]
        sub_df[f"shop_{i}_action_num_rank"] = minmax_scale(sub_df[f"shop_{i}_action_num"].rank())
        shop = pd.merge(shop, sub_df, how="left", on="shop_id")
        shop[f'shop_{i}_rate'] = shop[f"shop_{i}_action_num"] / (shop['shop_action_num'] + 1e-5)


    #  店铺购买间隔， 每一个购买过的用户的店铺平均购买间隔的统计特征。
    sub_df = action_product_comment_info[['user_id', "type", "shop_id", "action_time"]]
    sub_df = sub_df[sub_df['type'] == 2]
    sub_df.sort_values(by=['user_id', "shop_id", "action_time"], inplace=True)
    sub_df['before_shop'] = sub_df.groupby(['user_id', "shop_id"])['action_time'].shift(1)
    sub_df = sub_df[~(sub_df['before_shop'].isnull())]
    sub_df['shop_diff'] = (sub_df['action_time'] - sub_df['before_shop']).apply(lambda s:s.days)

    sub_df_median = sub_df.groupby(['user_id',"shop_id"])['shop_diff'].median().reset_index()

    sub_df_median.columns = ['user_id', "shop_id", "median_shop_diff"]

    sub_df_mean = sub_df.groupby(['user_id',"shop_id"])['shop_diff'].mean().reset_index()
    sub_df_mean.columns = ['user_id', "shop_id", "mean_shop_diff"]
    print(sub_df_median.head())

    shop_median_shop_diff =  get_statistic_info(sub_df_median, "shop_id", "median_shop_diff", "shop_median_shop_diff")
    print(sub_df_mean.head())
    shop_mean_shop_diff =  get_statistic_info(sub_df_mean, "shop_id", "mean_shop_diff", "shop_mean_shop_diff")

    shop = pd.merge(shop, shop_median_shop_diff, how="left", on="shop_id")
    shop = pd.merge(shop, shop_mean_shop_diff, how="left", on="shop_id")

    del shop_median_shop_diff, shop_mean_shop_diff, sub_df_mean, sub_df_median

    # 店铺最近一次购买距离目标区间时间的统计特征
    sub_df = action_product_comment_info[['user_id', "type", "shop_id", "action_time"]]
    sub_df = sub_df[sub_df['type'] == 2]
    sub_df = sub_df.groupby(['shop_id'])['action_time'].max().reset_index()
    sub_df['shop_last_shop_to_target_day'] = sub_df['action_time'].apply(lambda s: (target_day - s).days)
    shop = pd.merge(shop, sub_df[["shop_id", "shop_last_shop_to_target_day"]], how="left", on="shop_id")
    #product_info

    # 店铺的最后一次行为距离目标区间
    sub_df = action_product_comment_info[['user_id', "type", "shop_id", "action_time"]]
    sub_df = sub_df.groupby(['shop_id'])['action_time'].max().reset_index()
    sub_df['shop_last_action_to_target_day'] = sub_df['action_time'].apply(lambda s: (target_day - s).days)
    shop = pd.merge(shop, sub_df[["shop_id", "shop_last_action_to_target_day"]], how="left", on="shop_id")  

    assert shop_num == len(shop)
    shop = reduce_mem_usage(shop)
    print(shop['shop_id'].nunique())
    print(len(shop))
    print(f"shop feature exteact complete...\ndate shape: {shop.shape}")
    shop.to_hdf(feature_path + f"shop_info_{str(target_day)}.h5", key='df', mode='w')



if __name__ == "__main__":
        # 截止时间（<） 不取target_date当天
    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_shop_info(i)
