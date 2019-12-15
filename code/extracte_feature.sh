#!/bin/bash
# 将数据文件放在 ../input/ 下

echo "数据预处理..."
python preprocess.py

#
echo "提取用户基础特征..."
python user_base_feature.py
#
echo "提取类别基础特征..."
python cate_base_feature.py
#
echo "提取用户行为特征..."
python user_action_feature.py
#
echo "提取用户类别交互特征以及划窗特征..."
python user_cate_action_feature.py
#
echo "提取店铺特征..."
python shop_feature.py
#
echo　"提取用户店铺特征..."
python user_shop_feature.py
#
echo "提取类别店铺特征..."
python cate_shop_feature.py
#
echo "提取店铺划窗特征..."
python shop_move_windows_feature.py
#
echo "提取用户店铺划窗特征..."
python shop_user_move_windows_feature.py
#
echo "提取类别店铺划窗特征..."
python cate_shop_move_windows_feature.py
#
echo "提取补充特征"
python add_feature.py

