import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import os
from config import feature_path, base_file_path
from tools import reduce_mem_usage
from functools import partial

merge_feature = partial(pd.merge, how="left", on=['user_id', "shop_id"])

def extract_shop_user_move_feature(df, windows_size):
    shop_user = df[['user_id', "shop_id"]].drop_duplicates()

    assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())

    # 交互次数
    shop_user_active_num = df.groupby(['user_id', "shop_id"])['type'].count().reset_index()
    shop_user_active_num.columns = ['user_id', "shop_id",f"shop_user_active_num_{windows_size}"]
    shop_user_active_num[f"shop_user_active_rate_{windows_size}"] = shop_user_active_num.groupby("user_id")[f"shop_user_active_num_{windows_size}"].apply(lambda s: s/sum(s))

    shop_user = merge_feature(shop_user, shop_user_active_num)
    assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())

    # 交互天数
    shop_user_active_day = df.groupby(['user_id', "shop_id"])['action_time'].nunique().reset_index()
    shop_user_active_day.columns = ['user_id', "shop_id",f"shop_user_active_day_{windows_size}"]

    shop_user_active_day[f"shop_user_active_day_rate_{windows_size}"] = shop_user_active_day.groupby("user_id")[f'shop_user_active_day_{windows_size}'].apply(lambda s: s/sum(s))

    shop_user = merge_feature(shop_user, shop_user_active_day)
    assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())

    # 互动商品数
    shop_user_active_sku_num = df.groupby(['user_id', "shop_id"])['sku_id'].nunique().reset_index()
    shop_user_active_sku_num.columns = ['user_id', "shop_id",f"shop_user_active_sku_{windows_size}"]
    shop_user_active_sku_num[f"shop_user_active_sku_rate_{windows_size}"] = shop_user_active_sku_num.groupby("user_id")[f'shop_user_active_sku_{windows_size}'].apply(lambda s: s/sum(s))
    shop_user = merge_feature(shop_user, shop_user_active_sku_num )
    assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())

    for type_ in [1, 2, 3, 4]:
        # 歌类型的次数，和唯一值以及比例。 
        type_df = df[df['type'] == type_]

        type_user_count = type_df.groupby(["user_id", 'shop_id'])['type'].count().reset_index()
        type_user_count.columns = ["user_id", "shop_id", f"shop_user_{windows_size}_{type_}_num"]
        shop_user = merge_feature(shop_user, type_user_count)
        assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())

        type_user_days = type_df.groupby(["user_id", 'shop_id'])['action_time'].nunique().reset_index()
        type_user_days.columns = ["user_id", 'shop_id', f"shop_user_{windows_size}_{type_}_days"]
        shop_user = merge_feature(shop_user, type_user_days)
        assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())

        type_user_sku = type_df.groupby(["user_id", 'shop_id'])['sku_id'].nunique().reset_index()
        type_user_sku.columns = ["user_id", 'shop_id', f"shop_user_{windows_size}_{type_}_sku"]
        shop_user = merge_feature(shop_user, type_user_sku)
        assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())

        shop_user[f"shop_user_num_{windows_size}_{type_}_rate"] = shop_user[f"shop_user_{windows_size}_{type_}_num"] / shop_user[f"shop_user_active_num_{windows_size}"]
        shop_user[f"shop_user_day_{windows_size}_{type_}_rate"] = shop_user[f"shop_user_{windows_size}_{type_}_days"] / shop_user[f"shop_user_active_day_{windows_size}"]
        shop_user[f"shop_user_sku_{windows_size}_{type_}_rate"] = shop_user[f"shop_user_{windows_size}_{type_}_sku"] / shop_user[f"shop_user_active_sku_{windows_size}"]

        assert len(shop_user) == len(shop_user[["user_id","shop_id"]].drop_duplicates())
    return shop_user



def extract_shop_user_move_windows_feature(date, target_days, windows):
    action = date[date['action_time'] < target_days]
    product_table = pd.read_hdf(base_file_path + "jdata_product.h5")
    product_table.drop_duplicates(['sku_id'], inplace=True)
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

        move_shop_feature = extract_shop_user_move_feature(sub_date, days)
        assert len(move_shop_feature) == len(move_shop_feature[["user_id","shop_id"]].drop_duplicates())

        print(f"extract {str(target_days)}, {days} complete...")
        move_shop_feature.to_hdf(feature_path + f"move_window_user_shop_feature_{str(target_days)}_{days}.h5", key='df', mode='w')


if __name__ == "__main__":
    print("load action date")
    action = pd.read_hdf(base_file_path + "jdata_action.h5")
    # shoped_user = action[action['type'] == 2]['user_id'].unique().tolist()

    # print(f"购买过的用户数：{len(shoped_user)}")
    action['action_time'] = action['action_time'].apply(lambda s: date(*(int(i) for i in s.split(" ")[0].split("-"))))
    
    for i in [date(2018, 4, 16),date(2018, 4, 9)]:
        extract_shop_user_move_windows_feature(action, i, windows=[1, 3, 7, 14, 30])




