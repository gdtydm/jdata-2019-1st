import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os
from config import base_file_path, result_path
import gc
from config import feature_path
from tools import reduce_mem_usage

def merge_feature(df1, df2):
    return pd.merge(df1, df2, how="left", on="shop_id")

def extract_shop_feature_window(sub_df, window_size):
    shop = sub_df[['shop_id']].drop_duplicates()
    shop_num = len(shop)
    assert len(shop) == len(shop[["shop_id"]].drop_duplicates())

    sub_df = sub_df[['shop_id', "cate", "user_id", "sku_id", "type"]]
    # 交互用户数
    interactive_user = sub_df.groupby(['shop_id'])['user_id'].nunique().reset_index()
    interactive_user.columns = ['shop_id', f"{window_size}_active_user_num"]

    # 互动商品数
    interactive_sku = sub_df.groupby(['shop_id'])['sku_id'].nunique().reset_index()
    interactive_sku.columns = ['shop_id', f"{window_size}_active_sku_num"]

    # 互动次数
    interactive_all = sub_df.groupby(['shop_id'])['type'].count().reset_index()
    interactive_all.columns = ['shop_id', f"{window_size}_active_all_num"]

    shop = merge_feature(shop, interactive_user)
    shop = merge_feature(shop, interactive_sku)
    shop = merge_feature(shop, interactive_all)
    assert len(shop) == len(shop[["shop_id"]].drop_duplicates())

    for type_ in [1,2,3,4]:
        type_df = sub_df[sub_df['type'] == type_]
        type_user = type_df.groupby(['shop_id'])['user_id'].nunique().reset_index()
        type_user.columns = ['shop_id', f"{window_size}_{type_}_user_num"]

        type_sku = type_df.groupby(['shop_id'])['sku_id'].nunique().reset_index()
        type_sku.columns = ['shop_id', f"{window_size}_{type_}_sku_num"]

        type_count = type_df.groupby("shop_id")['type'].count().reset_index()
        type_count.columns = ['shop_id', f"{window_size}_{type_}_count_num"]

        shop = merge_feature(shop, type_user)
        shop = merge_feature(shop, type_sku)
        shop = merge_feature(shop, type_count)
        assert len(shop) == len(shop[["shop_id"]].drop_duplicates())

        shop.fillna(0, inplace=True)

        shop[f"{window_size}_{type_}_user_num_rate"] = shop[f"{window_size}_{type_}_user_num"] / (shop[f"{window_size}_active_user_num"] + 1e-5)
        shop[f"{window_size}_{type_}_sku_num_rate"] = shop[f"{window_size}_{type_}_sku_num"] / (shop[f"{window_size}_active_sku_num"] + 1e-5)
        shop[f"{window_size}_{type_}_count_num_rate"] = shop[f"{window_size}_{type_}_count_num"] / (shop[f"{window_size}_active_all_num"] + 1e-5)
    
    assert len(shop) == len(shop[["shop_id"]].drop_duplicates())

    return shop



def extract_shop_move_windows_feature(date, target_days, windows):
    action = date[date['action_time'] < target_days]
    product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
    product_table.drop_duplicates(subset=['sku_id'], inplace=True)
    product_table['market_time'] = product_table['market_time'].fillna(product_table['market_time'].mode().values[0])
    product_table['market_time'] = pd.to_datetime(product_table['market_time'])
    product_table['market_time'] = product_table['market_time'].apply( lambda s:s.date())
    product_table['market_time_to_target_day'] = (target_days - product_table['market_time']).apply(lambda s: s.days)
    action = pd.merge(action, product_table, how="left", on="sku_id")
    
    print(f"start extract {str(target_days)} shop move feature...")
    for days in windows:
        print(f"extract: {str(target_days)}_{days}")
        start_days = target_days - timedelta(days=days)
        sub_date = action[(action['action_time'] >= start_days) & (action['action_time'] < target_days)]

        if not os.path.exists(feature_path + f"move_window_shop_feature_{str(target_days)}_{days}.h5"):
            move_shop_feature = extract_shop_feature_window(sub_date, days)
            print(f"extract {str(target_days)}, {days} complete...")
            move_shop_feature.to_hdf(feature_path + f"move_window_shop_feature_{str(target_days)}_{days}.h5", key='df', mode='w')



if __name__ == "__main__":
    print("load action date")
    action = pd.read_hdf(base_file_path + "jdata_action.h5")
    # shoped_user = action[action['type'] == 2]['user_id'].unique().tolist()
    # print(f"购买过的用户数：{len(shoped_user)}")
    # action = action[action['user_id'].isin(shoped_user)]
    action['action_time'] = action['action_time'].apply(lambda s: date(*(int(i) for i in s.split(" ")[0].split("-"))))

    for i in [date(2018, 4, 16),date(2018, 4, 9),date(2018, 4, 2)]:
        extract_shop_move_windows_feature(action, i, [1,3,7,14,30])
