from math import sqrt
from typing import List

struct Matrix:
    var rows: Int
    var cols: Int
    var data: List[f64]

fn matrix_from_list(a: List[List[f64]]) raises -> Matrix:
    var rows = len(a)
    if rows == 0:
        raise Error("Cannot create matrix from empty list")
    var cols = len(a[0])
    var data = List[f64]()
    data.reserve_exact(rows * cols)
    for i in range(rows):
        var row = a[i]
        if len(row) != cols:
            raise Error("Inconsistent row length in matrix")
        for elem in row:
            data.append(elem)
    return Matrix(rows=rows, cols=cols, data=data)

fn matrix_to_list(m: Matrix) -> List[List[f64]]:
    var result = List[List[f64]]()
    result.reserve_exact(m.rows)
    for i in range(m.rows):
        var row = List[f64]()
        row.reserve_exact(m.cols)
        for j in range(m.cols):
            row.append(m.data[i * m.cols + j])
        result.append(row)
    return result

fn matmul(a: List[List[f64]], b: List[List[f64]]) raises -> List[List[f64]]:
    var mat_a = matrix_from_list(a)
    var mat_b = matrix_from_list(b)
    if mat_a.cols != mat_b.rows:
        raise Error("Incompatible matrix dimensions: columns of A ({}) must equal rows of B ({}).".format(mat_a.cols, mat_b.rows))
    var mat_c = Matrix(rows=mat_a.rows, cols=mat_b.cols, data=List[f64]())
    mat_c.data.reserve_exact(mat_a.rows * mat_b.cols)
    for i in range(mat_a.rows):
        for j in range(mat_b.cols):
            var sum = 0.0
            for k in range(mat_a.cols):
                sum += mat_a.data[i * mat_a.cols + k] * mat_b.data[k * mat_b.cols + j]
            mat_c.data.append(sum)
    return matrix_to_list(mat_c)

fn optimize( input: Float ) raises -> Float:
    return sqrt(input)

fn main():
    var a = [[1.0, 2.0], [3.0, 4.0]]
    var b = [[5.0, 6.0], [7.0, 8.0]]
    var c = matmul(a, b)
    print("Matrix multiplication result:", c)
    print("Optimized value:", optimize(16.0))
