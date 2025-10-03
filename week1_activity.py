import math

import numpy as np
import time


def dot_product_loop(vec1, vec2):
    """
    Calculate dot product using a for loop.

    Parameters:
    vec1 (list): First vector as a Python list
    vec2 (list): Second vector as a Python list

    Returns:
    float: The dot product of vec1 and vec2
    """
    # Your implementation here
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    result = 0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]
    return float(result)
    pass


def dot_product_numpy(vec1, vec2):
    """
    Calculate dot product using NumPy.

    Parameters:
    vec1 (np.ndarray): First vector as NumPy array
    vec2 (np.ndarray): Second vector as NumPy array

    Returns:
    float: The dot product of vec1 and vec2
    """
    # Your implementation here
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    return float(v1@v2)
    pass


def vector_norm_loop(vec):
    """
    Calculate L2 norm using a for loop.

    Parameters:
    vec (list): Input vector as a Python list

    Returns:
    float: The L2 norm of the vector
    """
    # Your implementation here
    total =0
    for x in vec:
        total += x**2
    return math.sqrt(total)
    pass


def vector_norm_numpy(vec):
    """
    Calculate L2 norm using NumPy.

    Parameters:
    vec (np.ndarray): Input vector as NumPy array

    Returns:
    float: The L2 norm of the vector
    """
    # Your implementation here
    return np.linalg.norm(vec)
    pass


def matrix_multiply_loop(mat1, mat2):
    """
    Multiply two matrices using nested for loops.

    Parameters:
    mat1 (list of lists): First matrix as nested Python lists
    mat2 (list of lists): Second matrix as nested Python lists

    Returns:
    list of lists: Result of matrix multiplication
    """
    # Your implementation here
    rows_mat1 = len(mat1)
    cols_mat1 = len(mat1[0])
    rows_mat2 = len(mat2)
    cols_mat2 = len(mat2[0])

    # Check if matrix multiplication is possible
    if cols_mat1 != rows_mat2:
        raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")

    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols_mat2)] for _ in range(rows_mat1)]

    # Perform matrix multiplication using three nested loops
    for i in range(rows_mat1):  # Iterate through rows of mat1
        for j in range(cols_mat2):  # Iterate through columns of mat2
            for k in range(cols_mat1):  # Iterate through common dimension
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
    pass


def matrix_multiply_numpy(mat1, mat2):
    """
    Multiply two matrices using NumPy.

    Parameters:
    mat1 (np.ndarray): First matrix as NumPy array
    mat2 (np.ndarray): Second matrix as NumPy array

    Returns:
    np.ndarray: Result of matrix multiplication
    """
    # Your implementation here
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    if m1.shape[1] != m2.shape[0]:
        raise ValueError("Number of columns does not match number of rows")
    return np.dot(m1, m2)
    pass


def matrix_transpose_loop(mat):
    """
    Transpose a matrix using for loops.

    Parameters:
    mat (list of lists): Input matrix as nested Python lists

    Returns:
    list of lists: Transposed matrix
    """
    # Your implementation here
    rows = len(mat)  # number of rows in input
    cols = len(mat[0])  # number of columns in input

    # Create empty result with flipped dimensions
    result = [[0 for _ in range(rows)] for _ in range(cols)]

    # Fill result by swapping rows <-> cols
    for i in range(rows):  # iterate over rows of mat
        for j in range(cols):  # iterate over cols of mat
            result[j][i] = mat[i][j]

    return result
    pass


def matrix_transpose_numpy(mat):
    """
    Transpose a matrix using NumPy.

    Parameters:
    mat (np.ndarray): Input matrix as NumPy array

    Returns:
    np.ndarray: Transposed matrix
    """
    # Your implementation here
    m = np.array(mat)
    return np.transpose(m)
    pass


def performance_comparison(size=1000):
    """
    Compare execution time between loop-based and NumPy implementations.

    Parameters:
    size (int): Size of vectors/matrices to test

    Returns:
    dict: Dictionary containing timing results with keys:
          - 'dot_product_loop_time'
          - 'dot_product_numpy_time'
          - 'matrix_multiply_loop_time'
          - 'matrix_multiply_numpy_time'
          - 'speedup_dot_product' (numpy_time / loop_time)
          - 'speedup_matrix_multiply' (numpy_time / loop_time)
    """
    results = {}

    # Generate random test data
    vec1_list = [np.random.random() for _ in range(size)]
    vec2_list = [np.random.random() for _ in range(size)]
    vec1_np = np.array(vec1_list)
    vec2_np = np.array(vec2_list)

    # For matrix multiplication, use smaller size to avoid timeout
    mat_size = min(100,max(1, size // 10))
    mat1_list = [[np.random.random() for _ in range(mat_size)]
                 for _ in range(mat_size)]
    mat2_list = [[np.random.random() for _ in range(mat_size)]
                 for _ in range(mat_size)]
    mat1_np = np.array(mat1_list)
    mat2_np = np.array(mat2_list)

    # Time dot product operations
    # Your timing implementation here
    t0 = time.perf_counter()
    dp_loop = dot_product_loop(vec1_list, vec2_list)
    t1 = time.perf_counter()
    dp_loop_time = t1 - t0

    t2 = time.perf_counter()
    dp_np = dot_product_numpy(vec1_np, vec2_np)
    t3 = time.perf_counter()
    dp_numpy_time = t3 - t2

    # Time matrix multiplication operations
    # Your timing implementation here
    t4 = time.perf_counter()
    mm_loop = matrix_multiply_loop(mat1_list, mat2_list)
    t5 = time.perf_counter()
    mm_loop_time = t5 - t4

    t6 = time.perf_counter()
    mm_np = mat1_np@mat2_np
    t7 = time.perf_counter()
    mm_numpy_time = t7 - t6

    # Calculate speedup ratios
    speedup_dp = dp_loop_time/dp_numpy_time
    speedup_mm = mm_loop_time/mm_numpy_time
    # Your calculation here

    results['dot_product_loop_time'] = dp_loop_time
    results['dot_product_numpy_time'] = dp_numpy_time
    results['matrix_multiply_loop_time'] = mm_loop_time
    results['matrix_multiply_numpy_time'] = mm_numpy_time
    results['speedup_dot_product'] = dp_loop_time / max(dp_numpy_time,1e-12)
    results['speedup_matrix_multiply'] = mm_loop_time / mm_numpy_time

    return results


def main():
    """
    Main function to demonstrate all implementations and print results.
    """
    print("=" * 50)
    print("NumPy vs For Loops Performance Comparison")
    print("=" * 50)

    # Test with small examples for correctness
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    print("\n--- Correctness Check ---")
    print(f"Dot Product (Loop): {dot_product_loop(vec1, vec2)}")
    print(f"Dot Product (NumPy): {dot_product_numpy(vec1_np, vec2_np)}")

    mat = [[1, 2], [3, 4]]
    mat_np = np.array(mat)
    print(f"\nOriginal Matrix: {mat}")
    print(f"Transpose (Loop): {matrix_transpose_loop(mat)}")
    print(f"Transpose (NumPy): {matrix_transpose_numpy(mat_np).tolist()}")

    # Performance comparison
    print("\n--- Performance Analysis ---")
    results = performance_comparison(size=1000)

    print(f"\nDot Product:")
    print(f"  Loop Time: {results['dot_product_loop_time']:.6f} seconds")
    print(f"  NumPy Time: {results['dot_product_numpy_time']:.6f} seconds")
    print(f"  NumPy Speedup: {results['speedup_dot_product']:.2f}x faster")

    print(f"\nMatrix Multiplication:")
    print(f"  Loop Time: {results['matrix_multiply_loop_time']:.6f} seconds")
    print(f"  NumPy Time: {results['matrix_multiply_numpy_time']:.6f} seconds")
    print(f"  NumPy Speedup: {results['speedup_matrix_multiply']:.2f}x faster")

    return results


if __name__ == "__main__":
    main()