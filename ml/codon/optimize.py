import ctypes
import os
import sys

# Load Mojo library for optimized kernels
mojo_lib_path = os.path.join(os.path.dirname(__file__), '..', 'mojo', 'libsimple_opt.so')
if not os.path.exists(mojo_lib_path):
    raise RuntimeError(f"Mojo library not found at {mojo_lib_path}. Compile simple_opt.mojo first.")

mojo_lib = ctypes.CDLL(mojo_lib_path)

# Define arg and return types for factorial
mojo_lib.factorial.argtypes = [ctypes.c_int]
mojo_lib.factorial.restype = ctypes.c_int

# Define arg and return types for optimize_calc
mojo_lib.optimize_calc.argtypes = [ctypes.c_float]
mojo_lib.optimize_calc.restype = ctypes.c_float

def factorial(n):
    """Mojo optimized factorial"""
    return mojo_lib.factorial(n)

def optimize_calc(x):
    """Mojo optimized calculation"""
    return float(mojo_lib.optimize_calc(ctypes.c_float(x)))

def run_optimize():
    print("Optimized factorial: ", factorial(5))
    print("Optimized calc: ", optimize_calc(3.0))

if __name__ == "__main__":
    run_optimize()
