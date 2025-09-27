from kernel import *

alias type = DType.float32

fn add(a: Scalar[type], b: Scalar[type]) -> Scalar[type]:
    return a + b

fn matrix_mult(A: Tensor[type], B: Tensor[type]) -> Tensor[type]:
    var C = Tensor[type](A.shape[0], B.shape[1])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C
