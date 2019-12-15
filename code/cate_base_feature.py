import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from tools import bys_smooth
import gc
import os
from tools import reduce_mem_usage
from config import feature_path, base_file_path

"""
数据路径都放在："../input/*"下面。因为内存有限为了加快加载速度和降低内存 运行了 preprocess.py 进行了与处理。

根据目标区间前的所有数据。提取一些基础的类别特征。有助于对类别进行刻画。
特征如下：

类别下的： 
商品数
品牌数 
店铺数
店铺注册时间到目标区间的中位数
所有商品的平均好评率（好评率使用了贝叶斯平滑，如果 一个商品的评论数为1，好评数为1，
算出来为 100%， 而如果评论数为1000，好评数为980，好评率为 98%， 
第一个高于第二个不符合实际情况）
在所有卖该类别的店铺中，主营类别为改类别的概率
所有店铺评分的中位数
所有店铺vip数量的中位数
所有店铺粉丝数的中位数
所有店铺注册时间到目标区间的中位数
（这里可以统计更多的信息，例如最大值，最小值，标准差，平均数之类的统计特征，鉴于内存原因只提取了中位数）


还可以提取的特征：（还未实现）
类别下所有用户平均的购买时间间隔。
类别下换店铺购买的概率
取top热度的店铺。并统计该类别top店铺的数量。（考虑到如果一个类别比较热门的店铺少，那么他下次还在这个店铺购买的可能性更大）

"""

def extract_cate_base_feature(target_date):
    print("load product table...")
    product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
    product_table.drop_duplicates(['sku_id'], inplace=True)
    product_table['market_time'] = product_table['market_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    product_table['market_time_to_target_day']  = (target_date - product_table['market_time']).apply(lambda s:s.days)
    print(product_table.head())

    print("load shop table...")
    shop_table = pd.read_hdf(base_file_path + "jdata_shop.h5")
    print(shop_table.head())
    shop_table['shop_reg_tm'].fillna("2000-11-03 10:45:58.0", inplace=True)
    shop_table['shop_reg_tm'] = shop_table['shop_reg_tm'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    shop_table['shop_reg_tm_to_target_day'] = (target_date - shop_table['shop_reg_tm']).apply(lambda s: s.days)
    shop_table.rename(columns={"cate":"shop_cate"},inplace=True)
    print(shop_table.head())

    # 评论数据有点奇怪。 随着时间的增加 评论数还会变少、主要利用 好评率，所以采取求和， 好评率用贝叶斯平滑
    print("load comment table...")
    comment_table = pd.read_hdf(base_file_path + "jdata_comment.h5")
    comment_table.drop("dt", axis=1, inplace=True)
    comment_table = comment_table.groupby("sku_id").sum().reset_index()
    good_rates, alpha, beta = bys_smooth(comment_table['comments'].values, comment_table['good_comments'].values)
    comment_table['good_rate'] = good_rates
    comment_table = comment_table[['sku_id', "good_rate"]]
    print(comment_table.head())

    print("merge product table, comments table, shop table...")

    product_shop_comments_table = pd.merge(pd.merge(product_table, comment_table, how="left", on="sku_id"), shop_table, how="left", on="shop_id")
    print(product_shop_comments_table.head())
    del comment_table, shop_table
    gc.collect()

    print("创建类别表...")
    cate_table = product_table[['cate']].drop_duplicates()
    print(f"类别数: {len(cate_table)}")
    print(product_shop_comments_table.columns)
    product_shop_comments_table['cate_is_shop_cate'] = np.where(product_shop_comments_table['cate'] == product_shop_comments_table['shop_cate'], 1, 0)

    print("店铺商品数")
    cate_un_feature = product_shop_comments_table.groupby("cate").agg({
        "sku_id":lambda s: s.nunique(),
        "brand":lambda s:s.nunique(),
        "shop_id": lambda s:s.nunique(),
        "shop_reg_tm_to_target_day": lambda s:np.median(s),
        "good_rate":lambda s: np.mean(s),
        "cate_is_shop_cate": lambda s:np.sum(s) / len(s),
    }).reset_index()
    cate_un_feature.columns = ['cate', "un_sku_id", "un_brand", "un_shop_id","shop_reg_tm_to_target_day_median", "avg_good_rate", "cate_is_shop_cate_rate"]

    cate_table = pd.merge(cate_table, cate_un_feature, how="left", on="cate")

    del cate_un_feature
    gc.collect()


    cate_unit_shop = product_shop_comments_table[['cate',"shop_score", "vip_num","fans_num","shop_reg_tm_to_target_day", "shop_id"]].drop_duplicates("shop_id")

    cate_unit_shop = cate_unit_shop.groupby("cate").agg({
        "shop_score": lambda s: np.median(s),
        "vip_num":lambda s: np.median(s),
        "fans_num":lambda s: np.median(s),
        "shop_reg_tm_to_target_day": lambda s:np.median(s)
    }).reset_index()
    cate_unit_shop.columns = ['cate', "cate_shop_score_median", "cate_vip_num_median", "cate_fans_num_median", "cate_shop_reg_tm_to_target_day_median"]
    cate_table = pd.merge(cate_table, cate_unit_shop, how="left", on="cate")
    del cate_unit_shop
    gc.collect()

    print(cate_table.shape)
    print("提取基础类别特征完成。")

    cate_table = reduce_mem_usage(cate_table) 
    cate_table.to_hdf(feature_path + f'base_cate_feature_{str(target_date)}.h5', key='df', mode='w')

if __name__ == "__main__":
    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_cate_base_feature(i)