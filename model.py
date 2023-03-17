import netCDF4 as nc
from model_info_2d import *
# 2022-08-21 11:16:44 Sola 修改读取地理信息时不用给定时间
# 2022-08-21 20:35:35 Sola 增加读取地形时的timestamp(还是至少得给定一个的)
# 2022-08-21 20:40:58 Sola 修正读取数据时读数的错误(将取余写成了整除)
# 2022-08-21 20:55:13 Sola 将其垂直高度插值到模式网格正中央
class CT2019B(model_info_2d):
    """
    model_info_2d的一个子类, 用于读取CarbonTracker2019B产品模式数据
    """
    def __init__(self, proj=ccrs.PlateCarree(), nx=120, ny=90, dx=3, dy=2, \
        lowerleft_lonlat=[-178.5, -89], nt=8, dt=3, var_list=['CO2'], \
        type=['CT2019B'], fun_nc_dat_dir='') -> None:
        """
        增加了读取模式数据所需要的文件路径及函数(不推荐使用匿名函数, 在并行时
        可能会出现问题)
        默认配置全球120*90的网格, 网格间距3°*2°, 3小时间隔, 每个文件8个时间
        类型为'CT2019B'
        """
        super().__init__(proj, nx, ny, dx, dy, lowerleft_lonlat, nt, dt, var_list, type)
        self.fun_sim_dat_dir = fun_nc_dat_dir
    def read_data(self, timestamp, var, layer=None):
        with nc.Dataset(self.fun_sim_dat_dir(timestamp)) as data:
            if layer is None:
                result = data.variables[var][timestamp%(24*60*60)//(self.dt*60*60)]
            else:
                result = data.variables[var][timestamp%(24*60*60)//(self.dt*60*60), layer]
        return result
    def read_height(self, timestamp):
        height_temp = self.read_data(timestamp, 'gph')
        result = (height_temp[1:] + height_temp[:-1])/2
        return result