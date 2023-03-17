from math import log, exp
import numpy as np
import sys


def pressure_vcorrd_3d(
        p_surf  : np.ndarray,
        c3      : np.ndarray,
        c4      : np.ndarray,
        p_top   : float         = None,
        **kargs
) -> np.ndarray:
    """
    适用于WRFv4以后的垂直气压计算, 对应 hybrid_opt = 2(其实旧方法也适用)
    2023-03-15 21:56:57 Sola 编写源代码, 根据WRF程序代码推断出的计算结果
    2023-03-15 21:57:07 Sola 利用现有数据进行了核对, 认为误差一般不超过千分之一,
                                最大不超过千分之五(近地面层)
    2023-03-15 22:01:04 Sola 仔细思考了下, 即便是旧方法, 该计算方式依旧适用
                                因为 C3 = ZN, C4 = 0, 结果会退化为 ZN*(PS-PT)+PT
    2023-03-16 19:52:27 Sola 优化代码格式, 添加了默认的模式顶气压
    """

    if p_top is None: p_top = 5000.

    pressure_3d = c3[:, None, None]*(p_surf - p_top)[None, :, :] + c4[:, None, None] + p_top

    return pressure_3d


def compute_vcoord_1d_coeffs(
        znw         : np.ndarray,
        hybrid_opt  : int   = None,
        etac        : float = None, 
        p1000mb     : float = None,
        p_top       : float = None,
        **kargs
) -> np.ndarray:
    """
    2023-03-15 20:59:29 Sola 编写源代码, 修改自WRF程序
    2023-03-15 21:16:29 Sola 与Fortran结果对比, 验证计算结果正确
    2023-03-16 19:52:57 Sola 优化代码格式, 添加了默认项
    """

    if hybrid_opt   is None: hybrid_opt = 2
    if etac         is None: etac       = 0.2
    if p1000mb      is None: p1000mb    = 100000.
    if p_top        is None: p_top      = 5000.

    if hybrid_opt == 0:
        c3f = znw
    elif hybrid_opt == 1:
        c3f = znw
    elif hybrid_opt == 2:
        B1 = 2*etac**2*(1 - etac)
        B2 = -etac*(4 - 3*etac - etac**3)
        B3 = 2*(1-etac**3)
        B4 = -(1 - etac**2)
        B5 = (1-etac)**4
        c3f = (B1 + B2*znw + B3*znw**2 + B4*znw**3)/B5
        c3f[znw<etac] = 0
        c3f[0], c3f[-1] = 1, 0
    elif hybrid_opt == 3:
        c3f = znw*np.sin(0.5*np.pi*znw)**2
        c3f[0], c3f[-1] = 1, 0
    else:
        print(
            """
            ERROR: --- hybrid_opt
            ERROR: --- hybrid_opt=0    ==> Standard WRF terrain-following coordinate
            ERROR: --- hybrid_opt=1    ==> Standard WRF terrain-following coordinate, hybrid c1, c2, c3, c4
            ERROR: --- hybrid_opt=2    ==> Hybrid, Klemp polynomial
            ERROR: --- hybrid_opt=3    ==> Hybrid, sin^2
            ERROR: --- Invalid option
            """
        )
        sys.exit(1205)
    
    c4f = (znw - c3f)*(p1000mb - p_top)
    znu = (znw[:-1] + znw[1:])/2
    c3h = (c3f[:-1] + c3f[1:])/2
    c4h = (znu - c3h)*(p1000mb - p_top)
    c1f = np.zeros_like(c3f)
    c1f[1:-1] = (c3h[1:] - c3h[:-1])/(znu[1:] - znu[:-1])
    c1f[0] = 1
    if hybrid_opt == 0 or hybrid_opt == 1:
        c1f[-1] = 1
    else:
        c1f[-1] = 0
    c2f = (1 - c1f)*(p1000mb - p_top)
    c1h = (c3f[1:] - c3f[:-1])/(znw[1:] - znw[:-1])
    c2h = (1 - c1h)*(p1000mb - p_top)
    print("compute_vcoord_1d_coeffs output: znu, c1f, c2f, c3f, c4f, c1h, c2h, c3h, c4h")

    return znu, c1f, c2f, c3f, c4f, c1h, c2h, c3h, c4h


def compute_eta(
        nlayer          : int,
        eta_levels      : np.ndarray    = None,
        auto_levels_opt : int           = None,
        p_top           : float         = None,
        max_dz          : float         = None,
        dzbot           : float         = None,
        dzstretch_s     : float         = None,
        dzstretch_u     : float         = None,
        **kargs
):
    """
    2023-03-15 20:34:12 Sola 编写源代码, 由WRF源代码简化而来, 仅求取参数, 不进行验证
    2023-03-16 19:53:19 Sola 优化代码格式, 添加了默认项
    """

    if auto_levels_opt is None: auto_levels_opt = 2

    if not eta_levels is None:
        znw = np.array(eta_levels)
    elif auto_levels_opt == 1:
        print(f"do not support {auto_levels_opt=} now!")
        sys.exit()
    elif auto_levels_opt == 2:
        znw = levels(nlayer, p_top, max_dz, dzbot, dzstretch_s, dzstretch_u, **kargs)
    else:
        print(f"unknown option {auto_levels_opt=}")
    
    return znw


def levels(
        nlev        : int,
        p_top       : float = None,
        max_dz      : float = None,
        dzbot       : float = None,
        dzstretch_s : float = None,
        dzstretch_u : float = None,
        r_d         : float = None,
        g           : float = None,
        **kargs
) -> np.ndarray:
    """
    2023-03-15 19:47:14 Sola 编写源代码, 从WRF的原始代码改编而来
    2023-03-15 19:47:51 Sola 经过和Fortran程序输出对比, 结果准确
    2023-03-16 19:53:31 Sola 优化代码格式, 添加默认项
    """

    if p_top        is None: p_top          = 5000.
    if max_dz       is None: max_dz         = 1000.
    if dzbot        is None: dzbot          = 50.
    if dzstretch_s  is None: dzstretch_s    = 1.3
    if dzstretch_u  is None: dzstretch_u    = 1.1
    if r_d          is None: r_d            = 287.
    if g            is None: g              = 9.81

    # 参数设定
    print('using new automatic levels program')
    print("level,  dz(m),   zup(m),   eta,     a")
    print("=====, ======, ========, =====, =====")
    zup = np.zeros((nlev))
    pup = np.zeros((nlev))
    eta = np.zeros((nlev + 1))
    tt = 290.
    ztop = r_d*tt/g*log(1.e5/p_top) # 模式顶部高度
    # zscale = r_d*tt/g
    dz = dzbot # 先将dz初始化为最底层高度

    zup[0] = dz # 第一层顶部高度
    pup[0] = 1.e5*exp(-g*zup[0]/r_d/tt) # 第一层顶部气压
    eta[0] = 1.0 # 设置底部eta
    eta[1] = (pup[0] - p_top)/(1.e5 - p_top) # 计算第1层顶部eta
    print(f"{0:5d}, {dz:6.1f}, {zup[0]:8.1f}, {eta[0+1]:5.3f}") # 打印第一层信息

    isave = 0 # 初始化层数
    for i in range(nlev - 1): # 对所有层(不包括顶层)进行循环, 从最底层开始
        # 当dz达到最大厚度的一半之前, 比例系数会逐渐从dzstretch_s(达不到)降低到dzstretch_u
        # 当dz达到最大厚度的一半之后, 比例系数都是上层的伸缩系数
        a = dzstretch_u + (dzstretch_s - dzstretch_u)*\
            max((max_dz*0.5 - dz)/(max_dz*0.5), 0.) # 比例系数
        dz = a*dz # 计算下一层层高
        dztest = (ztop - zup[isave])/(nlev - isave - 1) # 剩余每层平均可分配的高度 
        if dztest < dz: break # 如果剩余空间比该层小, 那就停止循环
        isave = i + 1 # 层数递增
        zup[i+1] = zup[i] + dz # 设置下一层层顶高度
        pup[i+1] = 1.e5*exp(-g*zup[i+1]/r_d/tt) # 计算下一层层顶气压
        eta[i+2] = (pup[i+1] - p_top)/(1.e5 - p_top) # 计算下一level的eta值, 是距离模式层顶压力差与整层压力差的比值
        print(f"{i+1:5d}, {dz:6.1f}, {zup[i+1]:8.1f}, {eta[i+2]:5.3f}, {a:5.3f}") # 打印下一层的信息
        if i == nlev - 1:
            print("""
                You need one of four things:
                1) More eta levels: e_vert
                2) A lower p_top: p_top_requested
                3) Increase the lowest eta thickness: dzbot
                4) Increase the stretching factor: dzstretch_s or dzstretch_u
                All are namelist options
                not enough eta levels to reach p_top
            """)
            sys.exit(7893)
    print(f"\n[ INFO ] {ztop=:8.1f}, {zup[isave]=:8.1f}, {nlev=:02d}, {isave=:02d}\n") # 输出当前层信息

    dz = (ztop - zup[isave])/(nlev - isave - 1) # 获取剩余层平均高度
    if dz > 1.5*max_dz:
        print(
            """
            Warning: Upper levels may be too thick
            You need one of five things:
            1) More eta levels: e_vert
            2) A lower p_top: p_top_requested
            3) Increase the lowest eta thickness: dzbot
            4) Increase the stretching factor: dzstretch_s or dzstretch_u
            5) Increase the maximum allowed thickness: max_dz
            All are namelist options
            Upper levels may be too thick
            """
        )
        sys.exit(7908)
    
    print("level,  dz(m),   zup(m),   eta")
    print("=====, ======, ========, =====")
    for i in range(isave, nlev - 1): # 对剩余层数进行循环
        zup[i+1] = zup[i] + dz # 下层层顶高度递增dz
        pup[i+1] = 1.e5*exp(-g*zup[i+1]/r_d/tt) # 计算对应的气压
        eta[i+2] = (pup[i+1] - p_top)/(1.e5 - p_top) # 计算eta
        if i + 2 == nlev: eta[i+2] = 0. # 将最顶层eta设置为0
        print(f"{i+1:5d}, {dz:6.1f}, {zup[i+1]:8.1f}, {eta[i+2]:5.3f}") # 打印信息
    
    return eta