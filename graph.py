import numpy as np
import torch

def gen_graph(level=1, init_nV=64, graph_type='ring'):
    ## A with shape [init_nV//level, init_nV//level]
    nV = init_nV // level
    
    A = np.zeros((nV, nV))
    if graph_type == 'ring':
        for i in range(A.shape[0]):
            A[i, i] = 1
            if i == 0:
                A[i, i+1] = 1
                A[i+1, i] = 1
                A[nV-1, i] = 1
                A[i, nV-1] = 1
            elif i == A.shape[0]-1:
                A[i, i-1] = 1
                A[i-1, i] = 1
                A[0, i] = 1
                A[i, 0] = 1
            else:
                A[i, i+1] = 1
                A[i+1, i] = 1
                A[i, i-1] = 1
                A[i-1, i] = 1
    elif graph_type == 'chain':
        for i in range(A.shape[0]):
            A[i, i] = 1
            if i == 0:
                A[i, i+1] = 1
                A[i+1, i] = 1
            elif i == A.shape[0]-1:
                A[i, i-1] = 1
                A[i-1, i] = 1
            else:
                A[i, i+1] = 1
                A[i+1, i] = 1
                A[i, i-1] = 1
                A[i-1, i] = 1
    else:
        raise ValueError('Wrong graph type')
    
    ## check A is symmetric
    # print("if A is symmetric: ", np.all(A == np.transpose(A)))
    print(A)

    return torch.from_numpy(A)