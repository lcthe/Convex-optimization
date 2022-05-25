#

import numpy as np
from scipy.optimize import linprog

c = np.array([[3, 9, 20, 19]], dtype=np.float64)
A = np.array([[110, 160, 420, 260], [4, 8, 4, 14], [2, 285, 22, 80]],
             dtype=np.float64)
b = np.array([[2000], [55], [800]], dtype=np.float64)

# 用linprog求解，比较结果
A_eq = None
b_eq = None
A_ub = -A
b_ub = -b

r = linprog(c, A_ub, b_ub, A_eq, b_eq)

print("linprog得到的解：\n", r)