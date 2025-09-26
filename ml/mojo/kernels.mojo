from math import sqrt
from typing import List

fn matmul(a: List[List[f64]], b: List[List[f64]]) -> List[List[f64]]:
    var rows_a = len(a)
    var cols_a = 0 if rows_a == 0 else len(a[0])
    var rows_b = len(b)
    var cols_b = 0 if rows_b == 0 else len(b[0])
    var result = List[List[f64]]()
    result.reserve_exact(rows_a)
    for i in range(rows_a):
        var row = List[f64]()
        row.reserve_exact(cols_b)
        for j in range(cols_b):
            var sum = 0.0
            for k in range(cols_a):
                sum += a[i][k] * b[k][j]
            row.append(sum)
        result.append(row)
    return result

fn optimize( input: Float ) raises -> Float:
    return sqrt(input)

fn main():
    var a = [[1.0, 2.0], [3.0, 4.0]]
    var b = [[5.0, 6.0], [7.0, 8.0]]
    var c = matmul(a, b)
    print("Matrix multiplication result:", c)
    print("Optimized value:", optimize(16.0))
