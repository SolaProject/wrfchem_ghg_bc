import datetime as dt
import numpy as np
from .vertical_interp import *
from .horizontal_interp import *

if __name__ == "__main__":
    # 测试脚本
    get_it("2019-01-01", 0, 1800)
    get_it(["2019-01-01", "2019-01-02"], 0, 1800)
    get_it(1, 0, 1800)
    get_it(dt.datetime(2019, 1, 1), 0, 1800)
    get_it(np.arange(0, 1000, 10, "datetime64[s]").reshape([10, -1]), 0, 1800)