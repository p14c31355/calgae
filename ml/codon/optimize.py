import codon

def optimize_calc(x):
    """LLVM optimized calculation"""
    return codon.wrap(lambda y: y * y + 2 * y + 1)(x)

@codon.wrap
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def run_optimize():
    print("Optimized factorial: ", factorial(5))
    print("Optimized calc: ", optimize_calc(3))

if __name__ == "__main__":
    run_optimize()
