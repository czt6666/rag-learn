import math
import numpy as np
import pandas as pd
from math import sqrt
from math import sqrt as msqrt

# from math import *

print(sqrt(16))
print(math.sqrt(16))

# 自定义模块
import utils.file  # 1
from utils import file  # 2
from utils.file import func  # 3
from file import func
from ..core import model

# myproj/
# ├── main.py
# └── utils/
#    ├── __init__.py
#    └── file.py

utils.file.func()  # 1
file.func()  # 2
func()  # 3

if use_torch:
    import torch
