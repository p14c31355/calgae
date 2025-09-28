from sys.info import *
from memory import Pointer, UnsafePointer

alias type = Float32

@export
fn matrix_mult_c(
    n: Int, p: Int, q: Int,
    a: UnsafePointer[type], b: UnsafePointer[type],
    c: UnsafePointer[type]
) -> Int:
    """FFI-compatible matrix mult: C = A * B, where A is n x q, B is q x p, C is n x p.
    Flattened row-major arrays. Returns 0 on success."""
    if n <= 0 or p <= 0 or q <= 0:
        return -1
    
    # Initialize C to zero
    for i in range(n * p):
        c.store(i, 0.0)
    
    # Optimized loop
    for i in range(n):
        for j in range(p):
            var acc: type = 0.0
            for k in range(q):
                acc += a.load(i * q + k) * b.load(k * p + j)
            c.store(i * p + j, acc)
    
    return 0
