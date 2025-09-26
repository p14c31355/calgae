from sys.info import get_cpu_brand
from math import sqrt
from typing import List, Optional

fn matmul(a: List[List[f64]], b: List[List[f64]]) -> List[List[f64]]:
    var rows_a = len(a)
    var cols_a = len(a[0])
    var cols_b = len(b[0])
    var result = List[List[f64]]()
    result.from_typed(MemoryPointer[MemoryPointer[f64]].alloc(rows_a))
    for i in range(rows_a):
        result[i] = List[f64]()
        result[i].from_typed(MemoryPointer[f64].alloc(cols_b))
        for j in range(cols_b):
            var sum = 0.0
            for k in range(cols_a):
                sum += a[i][k] * b[k][j]
            result[i][j] = sum
    return result

fn optimize( input: Float ) raises -> Float:
    return sqrt(input)

fn main():
    var a = [[1.0, 2.0], [3.0, 4.0]]
    var b = [[5.0, 6.0], [7.0, 8.0]]
    var c = matmul(a, b)
    print("Matrix multiplication result:", c)
    print("Optimized value:", optimize(16.0))
