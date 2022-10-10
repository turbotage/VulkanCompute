import time
import torch

nelem = 500000
ndata = 21
nparam = 4

for i in range(10):
    J = torch.rand(nelem, ndata, nparam, device = torch.device(0))
    r = torch.rand(nelem, ndata, 1, device = torch.device(0))
    H = torch.rand(nelem, nparam, nparam, device = torch.device(0))

    start = time.time()

    for j in range(30):
        grad = torch.bmm(J.transpose(1,2), r).neg()
        H = torch.bmm(J.transpose(1,2), J)

        A_LU = torch.lu(H)
        x = torch.lu_solve(grad, *A_LU)

    torch.cuda.synchronize()

    end = time.time()

    print(end - start)