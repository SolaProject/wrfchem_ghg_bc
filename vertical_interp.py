from .lib import compute_eta, compute_vcoord_1d_coeffs, pressure_vcorrd_3d
import numpy as np


def get_pressure_vcorrd_3d(
        p_surf          : np.ndarray,
        p_top           : float         = None,
        c3              : np.ndarray    = None,
        c4              : np.ndarray    = None,
        znw             : np.ndarray    = None,
        hybrid_opt      : int           = None,
        etac            : float         = None,
        p1000mb         : float         = None,
        nlayer          : int           = None,
        eta_levels      : np.ndarray    = None,
        auto_levels_opt : int           = None,
        max_dz          : float         = None,
        dz_bot          : float         = None,
        dzstretch_s     : float         = None,
        dzstretch_u     : float         = None
) -> np.ndarray:
    """
    主要就是根据输入参数, 计算3d静力气压场, 基本内容与WRF的计算方式保持一致
    更新记录:
        2023-03-16 19:51:54 Sola 编写源代码
        2023-03-16 19:52:02 Sola 经过测试后, 与原先代码输出结果一致, 可以正常运行
    """
    
    if not (c3 is None or c4 is None): # 如果c3和c4都有, 那就直接计算
        need_cal = []
    elif not znw is None: # 如果没有 c3, c4. 但是有 znw
        need_cal = ["pressure_1d"]
    elif not nlayer is None: # 如果 c3, c4, znw 都没有. 但是有层数
        need_cal = ["znw", "pressure_1d"]
    else: # 如果啥都没有, 那就GG
        sys.exit()
    
    if "znw" in need_cal: # 计算ZNW
        znw = compute_eta(nlayer, eta_levels=eta_levels, 
            auto_levels_opt=auto_levels_opt, p_top=p_top, max_dz=max_dz,
            dzbot=dz_bot, dzstretch_s=dzstretch_s, dzstretch_u=dzstretch_u)
    if "pressure_1d" in need_cal: # 计算eta的相关参数
        result = compute_vcoord_1d_coeffs(znw=znw, hybrid_opt=hybrid_opt,
            etac=etac, p1000mb=p1000mb, p_top=p_top)
        c3 = result[7]
        c4 = result[8]
    # 计算3D的静力气压场
    pressure_3d = pressure_vcorrd_3d(p_surf, p_top=p_top, c3=c3, c4=c4)

    return pressure_3d


def interp_3d(
        data            : np.ndarray,
        pressure_global : np.ndarray,
        pressure_target : np.ndarray,
        type            : str           = None, # level or layer
) -> np.ndarray:
    """
    主要就是将三维的数据根据气压场插值到对应的高度, 要求各网格在水平方向上经纬度
        是一致的, 仅存在垂直方向上气压的不同. 具体的插值方式会依据是layer还是level
        而变.
    更新记录:
        2023-03-16 19:58:09 Sola 编写源代码
        2023-03-16 20:51:51 Sola 加入对全球的气压场的判断
    """
    
    if data.shape[0] == pressure_global.shape[0] - 1: # 判断全球气压场的类型
        type_temp = "level" # 如果垂直方向比标量场多一个, 那么认为是 level 类型
    elif data.shape[0] == pressure_global.shape[0]:
        type_temp = "layer" # 如果垂直方向和标量场一致, 那么认为其是 layer 类型
    else: # 如果两个都不是, 那么认为输入的数据有问题, 但也不是不能继续算
        print("[WARNING] global model pressure shape dont match data shape")

    if type is None: # 如果没有预先给定类型, 那么根据计算结果自动给定
        type = type_temp
    elif not type.lower()[0:5] == type_temp: # 否则判断输入是否与计算结果一致
        # 如果不一致, 则提醒, 但依旧以输入的为准
        print(f"[WARNING] input pressure type is {type}, but program think " +
              f"it is {type_temp}. Will use {type}, please check it")
    print(f"INPUT: {type}; PROGRAM: {type_temp}") # 打印提示信息

    kk, jj, ii = get_zz_3d(pressure_global=pressure_global, 
                           pressure_target=pressure_target, type=type)
    result = data[kk, jj, ii] # 提取对应的结果

    return result


def get_zz_3d(
        pressure_global : np.ndarray,
        pressure_target : np.ndarray,
        type            : str           = None, # level or layer
):
    """
    从3d场插值的代码中单独将计算垂直位置的代码拆出来, 以方便重复计算
    可以计算区域模式每个网格在全球网格上的对应位置
    更新记录:
        2023-03-16 21:00:26 Sola 编写源代码
    """
    nz, ny, nx = pressure_target.shape # 获取输出数据的维度
    kk, jj, ii = np.meshgrid(range(nz), range(ny), range(nx), indexing="ij")
    for k in range(nz): # 获取每一层的位置
        kk[k] = get_zz_2d(pressure_global=pressure_global,
                       pressure_target=pressure_target[k],
                       type=type)
    return kk, jj, ii


def get_zz_2d(
        pressure_global : np.ndarray,
        pressure_target : np.ndarray,
        type            : str           = None,
        iz_float        : bool          = False,
        fun_pre2ling    : function      = None,
) -> np.ndarray:
    
    """
    用于计算该层气压在全球模式所处的层级, 有两种计算方式:
        1. level, 输入的气压是网格边界气压, 所以找到比其大的最小层即可
        2. layer, 输入的气压是网格中心气压(对数中心), 所以找到其距离最近的即可
    更新记录:
        2023-03-16 20:40:03 Sola 编写源代码
        2023-03-16 21:07:02 Sola 修正了区域模式结果可能比全球最顶层气压还低的问题
    """

    if type is None: type = "level" # 如果没有给定计算类型, 则默认输入的是level
    if fun_pre2ling is None: fun_pre2ling = lambda data: np.log(data)

    if type.lower() in ["level", "levels"]:
        iz = np.argmin(pressure_global > pressure_target, axis=0) - 1
        iz[pressure_global[0]<pressure_target] = 0 # 比全球最底层气压还高的取0
        iz[pressure_global[-1]>pressure_target] = -1 # 比全球最顶层气压还低的取-1
        # iz[iz<0] = 0 # 如果模式最底层比全球结果更低(气压更高), 则取最底层
    elif type.lower() in ["layer", "layers"]:
        iz = np.argmin(np.abs(fun_pre2ling(pressure_global) - 
                              fun_pre2ling(pressure_target)), axis=0)
    
    if iz_float:
        jj, ii = np.meshgrid(range(pressure_global.shape[1]), 
                             range(pressure_global.shape[2]), indexing="ij")
        if type.lower() in ["level", "levels"]:
            # 计算所对应层的压力差, 并根据模式的气压判断其相对位置
            pre_bot = fun_pre2ling(pressure_global[iz, jj, ii]) # 变换当前层底层压力
            pre_upp = fun_pre2ling(pressure_global[iz+1, jj, ii]) # 变换当前层顶层压力
            dp_global = pre_bot - pre_upp # 计算当前层压力差
            dp_wg = fun_pre2ling(pressure_target) - pre_bot # 计算与区域模式压力差
            iz += dp_wg/dp_global # 获取层内垂直位置
        elif type.lower() in ["layer", "layers"]:
            pre_mid = fun_pre2ling(pressure_global[iz, jj, ii]) # 所在层气压
            iz_bot = iz - 1 # 获取所在层下一层id
            iz_bot[iz_bot==-1] = 0 # 对于下一层id为-1的, 将其赋值为0
            iz_upp = iz + 1 # 获取所在层上一层id
            iz_upp[(iz_upp==0)|(iz_upp==pressure_global.shape[0])] = -1 # 将模式顶层的上一层设置为模式顶层
            pre_bot = fun_pre2ling(pressure_global[iz_bot, jj, ii]) # 获取下一层气压
            pre_upp = fun_pre2ling(pressure_global[iz_upp, jj, ii]) # 获取上一层气压
            pre_mid = fun_pre2ling(pressure_global[iz, jj, ii]) # 获取中间层气压
            pre_wrf = fun_pre2ling(pressure_target) # 获取区域模式气压
            dp_g_upp = pre_mid - pre_upp # 计算上层气压差
            dp_g_bot = pre_bot - pre_mid # 计算下层气压差
            # 判断区域模式气压与中间层气压大小, 对于模式气压更高的(处于中间层下方)
            #   使用下层的气压差, 否则使用上层的气压差, 然后将其加到iz上即可
            iz += np.where(pre_wrf>pre_mid, (pre_mid-pre_wrf)/dp_g_bot, (pre_mid-pre_wrf)/dp_g_upp) + 0.5

    return iz
