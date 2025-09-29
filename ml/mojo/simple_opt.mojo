from sys.info import *

alias type = Float32

@export
fn factorial(n: Int) -> Int:
    """Recursive factorial"""
    if n == 0:
        return 1
    return n * factorial(n - 1)

@export
fn optimize_calc(y: type) -> type:
    """LLVM optimized calculation: y^2 + 2y + 1"""
    return y * y + 2 * y + 1

# For FFI example usage
