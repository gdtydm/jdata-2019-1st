import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
from config import feature_path, base_file_path
from functools import partial
from sklearn.preprocessing import minmax_scale
from tools import reduce_mem_usage


merge_feature = partial(pd.merge, on=['cate', "shop_id"], how="left")

def str_to_day(df, column):
    df[column].fillna(df[column].mode().values[0], inplace=True)
    df[column] = df[column].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    return df


def extract_cate_shop_feature(target_day):
    ## 类别下各行为商铺比例
    action = pd.read_hdf(base_file_path + "jdata_action.h5")
    action = str_to_day(action, "action_time")

    product_info = pd.read_hdf(base_file_path + "jdata_product.h5")
    product_info = str_to_day(product_info, "market_time")
    user_shop_action = pd.merge(action, product_info, how="left", on="sku_id")
    del product_info, action

    print("根据目标区间获取行为区间。。。")
    user_shop_action = user_shop_action[user_shop_action['action_time']<target_day]

    # cate 行为次数, 行为天数， 用户数， 商品数， 

    cate_shop = user_shop_action[["cate", "shop_id"]].drop_duplicates().dropna()
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())
    cate_shop_len = len(cate_shop)
    print(cate_shop_len)
    for type_ in [1, 2, 3, 4]:
        sub_data = user_shop_action[user_shop_action['type'] == type_]
        sub_data_count = sub_data.groupby(['cate',"shop_id"])['type'].count().reset_index()
        sub_data_count.columns = ['cate', "shop_id", f"{type_}_count_num"]
        sub_data_count[f"{type_}_count_rate"] = sub_data_count.groupby("cate")[f"{type_}_count_num"].apply( lambda s: s/np.sum(s))
        sub_data_count[f"{type_}_count_rank"] = sub_data_count.groupby("cate")[f"{type_}_count_num"].rank(ascending=False)
        cate_shop = merge_feature(cate_shop, sub_data_count[['cate',"shop_id", f"{type_}_count_rate", f"{type_}_count_rank"]])
        print(cate_shop.shape)
        assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())

    # 数量        
    cate_shop_watch = user_shop_action[user_shop_action['type']==1][['cate', "shop_id", "type"]]
    cate_shop_watch = cate_shop_watch.groupby(['cate', "shop_id"])['type'].count().reset_index()
    cate_shop_watch.columns = ['cate', "shop_id", "cate_shop_watch"]

    cate_shop_shop = user_shop_action[user_shop_action['type']==2][['cate', "shop_id", "type"]]
    cate_shop_shop = cate_shop_shop.groupby(['cate', "shop_id"])['type'].count().reset_index()
    cate_shop_shop.columns = ['cate', "shop_id", "cate_shop_shop"]
    cate_shop_watch_shop = merge_feature(cate_shop_watch, cate_shop_shop)
    cate_shop_watch_shop.fillna(0, inplace=True) 
    cate_shop_watch_shop['cate_shop_watch'] = np.where(cate_shop_watch_shop['cate_shop_watch'] < cate_shop_watch_shop['cate_shop_shop'], cate_shop_watch_shop['cate_shop_shop'],cate_shop_watch_shop['cate_shop_watch']  )
    cate_shop_watch_shop['cate_shop_shop_rate'] = cate_shop_watch_shop['cate_shop_shop'] / cate_shop_watch_shop['cate_shop_watch']
    cate_shop_watch_shop['cate_shop_shop_rank'] = minmax_scale(cate_shop_watch_shop['cate_shop_shop'].rank())
    cate_shop = merge_feature(cate_shop, cate_shop_watch_shop[['cate', "shop_id", "cate_shop_shop_rate", "cate_shop_shop_rank"]])
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())


    # 用户数。 看过的用户数，有多少人选择了在该店铺下购买
    cate_shop_watch_user = user_shop_action[user_shop_action['type']==1][['cate', "shop_id", "user_id"]]
    cate_shop_watch_user = cate_shop_watch_user.groupby(['cate', "shop_id"])['user_id'].nunique().reset_index()
    cate_shop_watch_user.columns = ['cate', "shop_id", "cate_shop_watch_user"]

    cate_shop_shop_user = user_shop_action[user_shop_action['type']==2][['cate', "shop_id", "user_id"]]
    cate_shop_shop_user = cate_shop_shop_user.groupby(['cate', "shop_id"])['user_id'].nunique().reset_index()
    cate_shop_shop_user.columns = ['cate', "shop_id", "cate_shop_shop_user"]

    cate_shop_watch_shop_user = merge_feature(cate_shop_watch_user, cate_shop_shop_user)
    cate_shop_watch_shop_user.fillna(0, inplace=True) 
    cate_shop_watch_shop_user['cate_shop_watch_user'] = np.where(cate_shop_watch_shop_user['cate_shop_watch_user'] < cate_shop_watch_shop_user['cate_shop_shop_user'], cate_shop_watch_shop_user['cate_shop_shop_user'],cate_shop_watch_shop_user['cate_shop_watch_user']  )
    cate_shop_watch_shop_user['cate_shop_shop_user_rate'] = cate_shop_watch_shop_user['cate_shop_shop_user'] / cate_shop_watch_shop_user['cate_shop_watch_user']
    cate_shop_watch_shop_user['cate_shop_shop_user_rank'] = minmax_scale(cate_shop_watch_shop_user['cate_shop_shop_user'].rank())
    cate_shop = merge_feature(cate_shop, cate_shop_watch_shop_user[['cate', "shop_id", "cate_shop_shop_user_rate", "cate_shop_shop_user_rank"]])
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())


    # 该类别店铺，用户回购率
    cate_re_shop_rate = user_shop_action[user_shop_action["type"]==2][['user_id', "cate", "shop_id", "action_time"]]
    cate_re_shop_rate.sort_values(by=['user_id',"cate","shop_id","action_time"], inplace=True)
    cate_re_shop_rate = cate_re_shop_rate.groupby(['cate', "shop_id","user_id"])['action_time'].nunique().reset_index()
    cate_shop_shop_user_nums = cate_re_shop_rate.groupby(['cate',"shop_id"])['action_time'].apply(lambda s: len(s)).reset_index()
    cate_shop_shop_user_nums.columns = ['cate', "shop_id", "cate_shop_shop_user_num"]

    cate_shop_shop_user_nums_2 = cate_re_shop_rate.groupby(['cate',"shop_id"])['action_time'].apply(lambda s: sum([1 if i > 1 else 0 for i in s])).reset_index()
    cate_shop_shop_user_nums_2.columns = ['cate', "shop_id", "cate_shop_shop_user_num_2"]

    cate_shop = merge_feature(cate_shop, cate_shop_shop_user_nums)
    print(cate_shop.shape)
    cate_shop = merge_feature(cate_shop, cate_shop_shop_user_nums_2)
    print(cate_shop.shape)
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())

    cate_shop['cate_shop_re_shop_rate'] = cate_shop['cate_shop_shop_user_num_2'] / cate_shop['cate_shop_shop_user_num']
    print(cate_shop.shape)
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())
    # cate_shop = reduce_mem_usage(cate_shop)
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())
    print(cate_shop.info())
    print(f"extract {str(target_day)} cate shop action feature...")
    cate_shop.to_hdf(feature_path + f"cate_shop_feature_{str(target_day)}.h5", key='df', mode='w')
    


if __name__ == "__main__":
     # 截止时间（<） 不取target_date当天
    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_cate_shop_feature(i)

   




