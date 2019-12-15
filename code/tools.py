import numpy as np
import scipy.special as special
# rate = plot_user_appear_rate(dev_position, dev_user_feature_ares)

# 通过改变数值类型降低内存
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        try:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        except Exception as e:
            print(e)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# 贝叶斯平滑。
def bys_smooth(imps, clks, iter_num=10000, epsilon=1e-5):
    print("usys bys smooth")
    def __fixed_point_iteration(imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)

    rate = np.array(clks) / np.array(imps)
    avg = np.mean(rate)
    var = np.var(rate)
    alpha = avg * (avg * (1 - avg) / var -1)
    beta = (1 - avg) * (avg * (1 - avg) / var - 1)
    
    #考虑到时间消耗，没用使用迭代法。直接使用的矩估计结果
    # for i in range(iter_num):
    #     new_alpha, new_beta = __fixed_point_iteration(imps, clks, alpha, beta)
    #     if abs(new_alpha-alpha)<epsilon and abs(new_beta-beta)<epsilon:
    #         break
    #     alpha = new_alpha
    #     beta = new_beta
    ctr = []
    for i in range(len(imps)):
        ctr.append((clks[i]+alpha)/(imps[i]+alpha+beta))
    return ctr, alpha, beta

