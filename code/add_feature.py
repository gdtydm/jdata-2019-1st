import pandas as pd
import numpy as np
from datetime import datetime, date,timedelta
import os
from config import *
import gc
def add_user_feature(action_table, target_time):
    if os.path.exists(feature_path + f"add_user_feature_{str(target_time)}.h5"):
        return pd.read_hdf(feature_path + f"add_user_feature_{str(target_time)}.h5")
    action_table = action_table[action_table['action_time'] < target_time]
    user = action_table[["user_id"]].drop_duplicates()
    
    cate_shop_top3 = action_table[["cate", "shop_id", "type"]]
    cate_shop_top3 = cate_shop_top3[cate_shop_top3['type'] == 2]
    cate_shop_top3 = cate_shop_top3.groupby(['cate', "shop_id"])['type'].count().reset_index()
    cate_shop_top3_t = cate_shop_top3.sort_values(by=['cate', "type"], ascending=False)
    cate_shop_top3_t['cate_shop_rank'] = cate_shop_top3_t.groupby(['cate'])['type'].rank(method="min", ascending=False)
    cate_shop_rank = cate_shop_top3_t[['cate', "shop_id", "cate_shop_rank"]]

    user_shop_top_rate = action_table[action_table['type'] == 2]
    user_shop_top_rate  =  pd.merge(user_shop_top_rate[['user_id', "cate", "shop_id"]], cate_shop_rank, how="left", on=['cate',"shop_id"])
    for num in [1, 2, 3]:
        sub_set = user_shop_top_rate.groupby("user_id")['cate_shop_rank'].apply(lambda s:sum([1 if i <= num else 0 for i in s.tolist()])/len(s)).reset_index()
        sub_set.columns = ['user_id', f'shop_low_cate_rank_{num}_count']
        user = pd.merge(user, sub_set, how="left", on=['user_id'])

        sub_set = user_shop_top_rate.groupby("user_id")['cate_shop_rank'].apply(lambda s:sum([1 if i == num else 0 for i in s.tolist()])/len(s)).reset_index()
        sub_set.columns = ['user_id', f'shop_low_queal_rank_{num}_count']
        user = pd.merge(user, sub_set, how="left", on=['user_id'])    

    sub_set = user_shop_top_rate.groupby("user_id")['cate_shop_rank'].apply(lambda s:sum([1 if i > num else 0 for i in s.tolist()])/len(s)).reset_index()
    sub_set.columns = ['user_id', f'shop_over_cate_rank_{num}_count']
    user = pd.merge(user, sub_set, how="left", on=['user_id'])
    assert len(user) == len(user['user_id'].drop_duplicates())
    user.to_hdf(feature_path + f"add_user_feature_{str(target_time)}.h5", key='df', mode='w')
    return user

def add_user_cate_feature(action_table, target_time):
    if os.path.exists(feature_path + f"add_user_cate_feature_{str(target_time)}.h5"):
        return pd.read_hdf(feature_path + f"add_user_cate_feature_{str(target_time)}.h5")
    action_table = action_table[action_table['action_time'] < target_time]
    user_cate = action_table[["user_id", "cate"]].drop_duplicates()

    cate_shop_top3 = action_table[["cate", "shop_id", "type"]]
    cate_shop_top3 = cate_shop_top3[cate_shop_top3['type'] == 2]
    cate_shop_top3 = cate_shop_top3.groupby(['cate', "shop_id"])['type'].count().reset_index()
    cate_shop_top3_t = cate_shop_top3.sort_values(by=['cate', "type"], ascending=False)
    cate_shop_top3_t['cate_shop_rank'] = cate_shop_top3_t.groupby(['cate'])['type'].rank(method="min", ascending=False)
    cate_shop_rank = cate_shop_top3_t[['cate', "shop_id", "cate_shop_rank"]]

    user_shop_top_rate = action_table[action_table['type'] == 2]
    user_shop_top_rate  =  pd.merge(user_shop_top_rate[['user_id', "cate", "shop_id"]], cate_shop_rank, how="left", on=['cate',"shop_id"])
    for num in [3]:
        sub_set = user_shop_top_rate.groupby(["user_id", "cate"])['cate_shop_rank'].apply(lambda s:sum([1 if i <= num else 0 for i in s.tolist()])/len(s)).reset_index()
        sub_set.columns = ['user_id', "cate", f'user_cate_shop_low_cate_rank_{num}_count']
        user_cate = pd.merge(user_cate, sub_set, how="left", on=['user_id', "cate"])

        # sub_set = user_shop_top_rate.groupby(["user_id", "cate"])['cate_shop_rank'].apply(lambda s:sum([1 if i == num else 0 for i in s.tolist()])/len(s)).reset_index()
        # sub_set.columns = ['user_id',"cate", f'user_cate_shop_low_queal_rank_{num}_count']
        # user_cate = pd.merge(user_cate, sub_set, how="left", on=['user_id'])    

    sub_set = user_shop_top_rate.groupby(["user_id", "cate"])['cate_shop_rank'].apply(lambda s:sum([1 if i > num else 0 for i in s.tolist()])/len(s)).reset_index()
    sub_set.columns = ['user_id',"cate", f'user_cate_shop_over_cate_rank_{num}_count']
    user_cate = pd.merge(user_cate, sub_set, how="left", on=['user_id', "cate"])
    assert len(user_cate) == len(user_cate[['user_id', "cate"]].drop_duplicates())
    user_cate.to_hdf(feature_path + f"add_user_cate_feature_{str(target_time)}.h5", key='df', mode='w')
    return user_cate


def add_cate_shop_feature(action_table, target_time):
    if os.path.exists(feature_path + f"add_cate_shop_feature_{str(target_time)}.h5"):
        return pd.read_hdf(feature_path + f"add_cate_shop_feature_{str(target_time)}.h5")
        # 该店铺是否是该品类销量前三的店铺
    action_table = action_table[action_table['action_time'] < target_time]
    cate_shop = action_table[["cate", "shop_id"]].drop_duplicates()
    
    # 是否是好评率前三的店铺
    cate_shop_top3 = action_table[["cate", "shop_id", "type"]]
    cate_shop_top3 = cate_shop_top3[cate_shop_top3['type'] == 2]
    cate_shop_top3 = cate_shop_top3.groupby(['cate', "shop_id"])['type'].count().reset_index()
    cate_shop_top3_t = cate_shop_top3.sort_values(by=['cate', "type"], ascending=False)
    
    cate_shop_top3_t['shop_shoped_rate'] = cate_shop_top3_t.groupby("cate")['type'].apply(lambda s:s/np.sum(s))
    cate_shop_top3_t['shop_shoped_rank'] = cate_shop_top3_t.groupby("cate")['type'].rank(method="min", ascending=False)
    cate_shop = pd.merge(cate_shop, cate_shop_top3_t[['cate',"shop_id", "shop_shoped_rate", "shop_shoped_rank"]], how="left", on=['cate', "shop_id"])

    
    cate_shop_top3_t = cate_shop_top3_t.groupby("cate").head(3)
    cate_shop_top3_t['shop_is_cate_shop_top3'] = 1
    cate_shop = pd.merge(cate_shop, cate_shop_top3_t[['cate',"shop_id", "shop_is_cate_shop_top3"]], how="left", on=["cate", "shop_id"])
    cate_shop.fillna({"shop_is_cate_shop_top3":0}, inplace=True)
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())
    del cate_shop_top3, cate_shop_top3_t
    # 是否是好评率前三的店铺

    # 是否是目标区间前一周销量最好的top3
    cate_shop_top3_before_week = action_table[["cate", "shop_id", "type","action_time"]]
    cate_shop_top3_before_week = cate_shop_top3_before_week[cate_shop_top3_before_week['type']==2]
    cate_shop_top3_before_week = cate_shop_top3_before_week[(cate_shop_top3_before_week['action_time']<target_time) & (cate_shop_top3_before_week['action_time']>=(target_time - timedelta(7)))]
    cate_shop_top3_before_week = cate_shop_top3_before_week.groupby(['cate', "shop_id"])['type'].count().reset_index()
    cate_shop_top3_before_week = cate_shop_top3_before_week.sort_values(by=['cate', "type"], ascending=False)
    cate_shop_top3_before_week['shop_shoped_rate'] = cate_shop_top3_before_week.groupby("cate")['type'].apply(lambda s:s/np.sum(s))
    
    cate_shop = pd.merge(cate_shop, cate_shop_top3_before_week[['cate',"shop_id", "shop_shoped_rate"]], how="left", on=['cate', "shop_id"])

    cate_shop_top3_before_week = cate_shop_top3_before_week.groupby("cate").head(3)
    cate_shop_top3_before_week['shop_is_cate_shop_top3_before_week'] = 1
    cate_shop = pd.merge(cate_shop, cate_shop_top3_before_week[['cate',"shop_id", "shop_is_cate_shop_top3_before_week"]], how="left", on=['cate', "shop_id"])
    cate_shop.fillna({"shop_is_cate_shop_top3_before_week":0}, inplace=True)
    assert len(cate_shop) == len(cate_shop[['cate',"shop_id"]].drop_duplicates())
    del cate_shop_top3_before_week
    cate_shop.to_hdf(feature_path + f"add_cate_shop_feature_{str(target_time)}.h5", key='df', mode='w')
    return cate_shop
    #
if __name__ == "__main__":
    action_table = pd.read_hdf(base_file_path + "jdata_action.h5")
    action_table['action_time'] = action_table['action_time'].apply(lambda s:date(*(int(i) for i in s.split(" ")[0].split("-"))))
    product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
    action_table = pd.merge(action_table, product_table, how="left", on="sku_id")
    action_table = action_table[['user_id', 'cate', "shop_id","action_time", "type"]]
    del product_table
    gc.collect()

    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        add_user_cate_feature(action_table, i)
        add_user_feature(action_table, i)
        add_cate_shop_feature(action_table, i)



