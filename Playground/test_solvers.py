
import numpy as np
from scipy.linalg import ldl

#auto mat_tensor = mgr.tensor({
#		3,2,1,
#		2,4,2,
#		1,2,5
#		});
#
#auto rhs_tensor = mgr.tensor({
#	1,1,1
#	});
#
#auto sol_tensor = mgr.tensor({
#	0,0,0
#	});

A = np.array([[3,2,1],[2,4,2],[1,2,5]])
rhs = np.array([1,1,1])

lu, d, parm = ldl(A)

print(lu)
print(d)

sol = np.linalg.solve(A,rhs)
print(sol)
