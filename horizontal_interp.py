import numpy as np
from .lib import model_info_2d


def interp_xy_nearest(
        model_global    : model_info_2d = None,
        model_rigion    : model_info_2d = None,
        data            : np.ndarray    = None,
) -> np.ndarray:

    """
    用于利用输入的模式信息, 将数据从全球模式插值到区域尺度, 使用最邻近插值法
    适用于区域尺度远小于全球尺度的插值(只是一种用于测试的方法)
    更新记录:
        2023-03-17 22:29:00 Sola 编写源代码
    """

    xlon, xlat = model_rigion.get_grid()
    xx, yy = model_global.grid_ids(xlon, xlat)
    print(f"dimension origin: {data.shape}")
    result = data[..., yy, xx]
    print(f"dimension change: {data.shape} => {result.shape}")
    print("method: nearest interp")
    if not data.shape[:-2] == result.shape[:-2]:
        print(f"[ERROR] dimension change after horizontal interp!")
    
    return result

def interp_xy_linalg(
        model_global    : model_info_2d = None,
        model_rigion    : model_info_2d = None,
        data            : np.ndarray    = None,
) -> np.ndarray:

    """
    用于利用输入的模式信息, 将数据从全球模式插值到区域尺度, 使用双线性插值法
        考虑了一下, 不应该考虑越界的问题, 本身从一个大网格插值到小网格, 还能
        越界就是错误的好吧, 越界问题不应该在这里处理, 要么外面给的就是空, 要
        么就报错.
    参考 Rainer Schmitz (University of Chile - Santiago, Chile) 和
        Steven Peckham (NOAA/ESRL/GSD - Boulder, CO) 在WRF_boundary_coupling
        软件中的方法写下了双线性插值的算法
    更新记录:
        2023-03-17 22:29:00 Sola 编写源代码
        2023-03-17 22:31:07 Sola 修改源代码为双线性方式
    """

    xlon, xlat = model_rigion.get_grid()
    xxf, yyf = model_global.grid_ids_float(xlon, xlat)
    xx0, xx1, yy0, yy1 = np.floor(xxf).astype(int), np.ceil(xxf).astype(int), \
        np.floor(yyf).astype(int), np.ceil(yyf).astype(int)
    print(f"dimension origin: {data.shape}")
    result = data[..., yy0, xx0]*(xx1 - xxf)*(yy1 - yyf) +\
             data[..., yy0, xx1]*(xxf - xx0)*(yy1 - yyf) +\
             data[..., yy1, xx0]*(xx1 - xxf)*(yyf - yy0) +\
             data[..., yy1, xx1]*(xxf - xx0)*(yyf - yy0)
    print(f"dimension change: {data.shape} => {result.shape}")
    print("method: linalg interp")
    if not data.shape[:-2] == result.shape[:-2]:
        print(f"[ERROR] dimension change after horizontal interp!")
    
    return result