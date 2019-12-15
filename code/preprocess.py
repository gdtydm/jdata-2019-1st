import pandas as pd
from tools import reduce_mem_usage, bys_smooth
from config import base_file_path
# def sku_good_rate():
#     sku_comment = pd.read_hdf("../input/jdata_comment.h5")
#     sku_comment.drop_duplicates("sku_id",inplace=True)



if __name__ == "__main__":
    jdata_action=reduce_mem_usage(pd.read_csv(base_file_path + 'jdata_action.csv',sep=','))
    jdata_comment=reduce_mem_usage(pd.read_csv(base_file_path + 'jdata_comment.csv',sep=','))
    jdata_product=reduce_mem_usage(pd.read_csv(base_file_path + 'jdata_product.csv',sep=','))
    jdata_shop=reduce_mem_usage(pd.read_csv(base_file_path + 'jdata_shop.csv',sep=','))
    jdata_user=reduce_mem_usage(pd.read_csv(base_file_path + 'jdata_user.csv',sep=','))
    
    jdata_action.to_hdf(base_file_path + 'jdata_action.h5', key='df', mode='w')
    jdata_comment.to_hdf(base_file_path + 'jdata_comment.h5', key='df', mode='w')
    jdata_product.to_hdf(base_file_path + 'jdata_product.h5', key='df', mode='w')
    jdata_shop.to_hdf(base_file_path + 'jdata_shop.h5', key='df', mode='w')
    jdata_user.to_hdf(base_file_path + 'jdata_user.h5', key='df', mode='w')
