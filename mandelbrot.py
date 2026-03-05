import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
import warnings
import numba
from numba import njit
from numba import int32
from numba import complex128
import cProfile
import pstats
import math
import random

"""
Mandelbrot Set Generator
Author : [Jakob Rundlett]
Course : Numerical Scientific Computing 2026
"""

def benchmark(func, *args, n_runs=3) :
    """ Time func, return median of n_runs."""
    print(f'Running {func.__name__} {n_runs}x...')
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter()-t0)
    median_t = statistics.median(times)
    # print(f"Median: {median_t:.4f}s" f"(min={min(times):.4f}, max={max(times):.4f})")
    print(f'{func.__name__} computation took {median_t:.3f} seconds')
    return median_t, result

# @profile
def compute_mandelbrot_naive(x_min, x_max, x_res, y_min, y_max, y_res, max_iter):
    # for 1024x1024    
    region_x_step = (x_max - x_min) / x_res
    region_y_step = (y_max - y_min) / y_res
    # create the pixel cooridnates
    points_x = np.arange(x_min, x_max, region_x_step)
    points_y = np.arange(y_min, y_max, region_y_step)

    # return the number of iterations given a complex number
    # @profile
    def mandelbrot_point(c):
        z = [0 + 0j]
        for n in range(max_iter):
            new_z = z[-1]**2 + c
            if abs(new_z) > 2:
                return n
            else:
                z.append(new_z)
        return max_iter

    # create the grid of complex numbers for each pixel
    complex_grid = []
    for x in points_x:
        new_line = []
        for y in points_y:
            new_line.append(complex(x, y))
        complex_grid.append(new_line)

    complex_grid = np.array(complex_grid)

    # replace all the complex numbers with the corresponding number of iterations
    for row in range(len(complex_grid)):
        for col in range(len(complex_grid[0])):
            complex_grid[row][col] = mandelbrot_point(complex_grid[row][col])

    # I don't know why the grid still has complex numbers. They should all be replaced with integers
    # this also throws a warning every time, so we supress it... 
    # "oh no, you lose the imaginary part if you turn an imaginary number into an integer!!!" Yeah no duh
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complex_grid = complex_grid.astype(int)

    return np.rot90(complex_grid)

def compute_mandelbrot_numpy(x_min, x_max, x_res, y_min, y_max, y_res, max_iter):
    x = np.linspace (x_min, x_max, x_res) # x resolution
    y = np.linspace (y_min, y_max, y_res) # y resolution
    X, Y = np.meshgrid(x, y) # standard grid
    C = X + 1j * Y # complex grid

    Z = np.zeros(C.shape, dtype=C.dtype)
    M = np.zeros(C.shape, dtype=int)

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

def compute_mandelbrot_hybrid_numba(x_min, x_max, x_res, y_min, y_max, y_res, max_iter):
    # for 1024x1024    
    region_x_step = (x_max - x_min) / x_res
    region_y_step = (y_max - y_min) / y_res
    # create the pixel cooridnates
    points_x = np.arange(x_min, x_max, region_x_step)
    points_y = np.arange(y_min, y_max, region_y_step)

    # return the number of iterations given a complex number
    # @profile
    @njit
    def mandelbrot_point(c: complex128) -> int32:
        z = 0j
        for n in range(max_iter):
            z = z*z + c
            if z.real*z.real + z.imag*z.imag > 4.0:
                return n
        return max_iter

    # create the grid of complex numbers for each pixel
    complex_grid = []
    for x in points_x:
        new_line = []
        for y in points_y:
            new_line.append(complex(x, y))
        complex_grid.append(new_line)

    complex_grid = np.array(complex_grid)

    # replace all the complex numbers with the corresponding number of iterations
    for row in range(len(complex_grid)):
        for col in range(len(complex_grid[0])):
            complex_grid[row][col] = mandelbrot_point(complex_grid[row][col])

    # I don't know why the grid still has complex numbers. They should all be replaced with integers
    # this also throws a warning every time, so we supress it... 
    # "oh no, you lose the imaginary part if you turn an imaginary number into an integer!!!" Yeah no duh
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complex_grid = complex_grid.astype(int)


    return np.rot90(complex_grid)

@njit
def compute_mandelbrot_naive_numba(x_min, x_max, x_res, y_min, y_max, y_res, max_iter, dtype=np.float32):
    """Fully JIT-compiled Mandelbrot --- structure identical to naive."""
    x = np.linspace(x_min, x_max, x_res).astype(dtype)
    y = np.linspace(y_min, y_max, y_res).astype(dtype)
    result = np.zeros((y_res, x_res), dtype=np.int32)

    for i in range(y_res): # compiled loop
        for j in range(x_res): # compiled loop
            c = x[j] + 1j*y[i]
            z = 0j # complex literal : type inference works !
            n = 0
            while n < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0:
                z = z*z + c
                n += 1
            result [i, j] = n
    return result

# def estimate_pi_serial(num_samples):
#     hits = 0
#     for _ in range(num_samples):
#         x = np.random.rand() # random x-value in unit square
#         y = np.random.rand() # random y-value in unit square
#         if x*x + y*y <= 1.0: # check if inside unit circle
#             hits += 1
#     return (hits / num_samples) * 4.0

def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

# -1: do nothing
# 0: run the functions normally
# 1: check performance of row vs col sums for numpy
# 2: check the profiles for the original functions
# 3: data type test
# 4: Monte Carlo pi estimation test

run_tests = 4
if run_tests == 0:
    gen_res_list = [256, 512, 1024, 2048, 4096, 4096*2, 4096*4]
    gen_res = 4096*2
    max_iter = 512
    n_runs = 1

    run_naive = False
    run_numpy = False
    run_hybrid_numba = False
    run_naive_numba = True
    run_version = [run_naive, run_numpy, run_hybrid_numba, run_naive_numba]

    view_image = False
    save_image = True
    plot_times = False

    versions = [compute_mandelbrot_naive, compute_mandelbrot_numpy, compute_mandelbrot_hybrid_numba, compute_mandelbrot_naive_numba]
    if run_version.count(False) == 0:
        for func in versions:
            _, _ = benchmark(func, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
    else:
        if run_version.count(True) >= 2:
            func1 = None
            func2 = None
            for i in range(len(run_version)):
                if run_version[i] and func1 == None:
                    func1 = versions[i]
                elif run_version[i] and func2 == None:
                    func2 = versions[i]

            _, func1_result = benchmark(func1, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
            _, func2_result = benchmark(func2, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
            
            # check the strict numerical difference between the results
            if np.allclose(func1_result, func2_result):
                print("Results match!")
            else:
                print("Results differ!")
                diff = np.abs(func1_result - func2_result)
                print(f"Max difference: {diff.max()}")
                print(f"Different pixels: {(diff > 0).sum()} ({(((diff > 0).sum()/(gen_res**2))*100):.1f}%)")

            # plot the results next to each other
            if view_image or save_image:
                # Create subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                ax1.imshow(func1_result)
                ax1.set_title(f'{func1.__name__} Result')
                ax1.axis('off')

                ax2.imshow(func2_result)
                ax2.set_title(f'{func2.__name__} Result')
                ax2.axis('off')

                # Show/save the plots
                plt.tight_layout()
                if save_image:
                    plt.savefig('mandelbrotFigure')
                if view_image:
                    plt.show()

        elif run_naive:
            _, naive_result = benchmark(compute_mandelbrot_naive, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
            
            if view_image or save_image:
                plt.imshow(naive_result)
                plt.title('Naive Result')
                plt.colorbar()
                if save_image:
                    plt.savefig('mandelbrotFigure')
                if view_image:
                    plt.show()
        
        elif run_numpy:

            if plot_times:
                res_times = []
                for new_res in gen_res:
                    new_time, numpy_result = benchmark(compute_mandelbrot_numpy, -2, 1, new_res, -1.5, 1.5, new_res, max_iter, n_runs=n_runs)
                    res_times.append(new_time)
                print(res_times)

            else:
                _, numpy_result = benchmark(compute_mandelbrot_numpy, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)

                if view_image or save_image:
                    plt.imshow(numpy_result)
                    plt.title('Numpy Result')
                    plt.colorbar()
                    if save_image:
                        plt.savefig('mandelbrotFigure')
                    if view_image:
                        plt.show()

        elif run_hybrid_numba:
            _, hybrid_numba_result = benchmark(compute_mandelbrot_hybrid_numba, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
            print('finished hybrid numba version...')

            if view_image or save_image:
                plt.imshow(hybrid_numba_result)
                plt.title('Hybrid Numba Result')
                plt.colorbar()
                if save_image:
                    plt.savefig('mandelbrotFigure')
                if view_image:
                    plt.show()

        elif run_naive_numba:
            _, naive_numba_result = benchmark(compute_mandelbrot_naive_numba, -1.5, 0.5, gen_res, -1, 1, gen_res, max_iter, n_runs=n_runs)
            print('finished naive numba version...')

            if view_image or save_image:
                plt.figure(figsize=(10, 10))
                plt.axis('off')
                plt.imshow(naive_numba_result)
                # plt.title('Naive Numba Result')
                # plt.colorbar()
                if save_image:
                    plt.savefig('mandelbrotFigure.png', dpi=gen_res/10)
                if view_image:
                    plt.show()

elif run_tests == 1:
    def test_row(N, A):
        for i in range(N): s = np.sum(A[i, :])
        return None

    def test_col(N, A):
        for j in range(N): s = np.sum(A[:, j])
        return None

    N = 10000
    A = np.random.rand(N, N)
    A_f = np.asfortranarray(A)
    n_runs = 3
    print('With standard np.array ...')
    _, _ = benchmark(test_row, N, A, n_runs=n_runs)
    _, _ = benchmark(test_col, N, A, n_runs=n_runs)
    print('With fortran np.array ...')
    _, _ = benchmark(test_row, N, A_f, n_runs=n_runs)
    _, _ = benchmark(test_col, N, A_f, n_runs=n_runs)

elif run_tests == 2:
    # x_min, x_max, x_res, y_min, y_max, y_res, max_iter
    cProfile.run('compute_mandelbrot_naive(-2, 1, 512, -1.5, 1.5, 512, 100)', 'naive_profile.prof')
    cProfile.run('compute_mandelbrot_numpy(-2, 1, 512, -1.5, 1.5, 512, 100)', 'numpy_profile.prof')

    for name in ('naive_profile.prof', 'numpy_profile.prof'):
        stats = pstats.Stats(name)
        stats.sort_stats('cumulative')
        stats.print_stats(10)

elif run_tests == 3:
    # for dtype in [np.float32, np.float64]:
    #     print(f'running with precision {dtype}')
    #     _, _ = benchmark(compute_mandelbrot_naive_numba, -2, 1, 2048, -1.5, 1.5, 2048, 100, dtype)

    # r16 = compute_mandelbrot_naive_numba(-2, 1, 1024, -1.5, 1.5, 1024, dtype = np.float16)
    r32 = compute_mandelbrot_naive_numba(-2, 1, 2048, -1.5, 1.5, 2048, 100, dtype = np.float32)
    r64 = compute_mandelbrot_naive_numba(-2, 1, 2048, -1.5, 1.5, 2048, 100, dtype = np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    for ax, result, title in zip(axes, [r32, r64],  ['float32', 'float64 (ref)']):
        ax.imshow(result, cmap='hot')
        ax.set_title(title); ax.axis('off')
    plt.savefig('precision_comparison.png', dpi=150)
    print(f"Max diff float32 vs float64: {np.abs(r32 - r64).max()}")
    # print(f"Max diff float16 vs float64: {np.abs(r16 - r64).max()}")

elif run_tests == 4:
    # First we check how long it takes to run the naive Monte Carlo with 10,000,000 samples
    # num_samples = 10**7
    # _, pi_estimate = benchmark(estimate_pi_serial, num_samples)
    # print(f"Estimated pi with {num_samples} samples: {pi_estimate}")
    if __name__ == '__main__':
        num_samples = 10_000_000
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            pi_estimate = estimate_pi_serial(num_samples)
            times.append(time.perf_counter() - t0)
        t_serial = statistics.median(times)
        print(f"pi estimate: {pi_estimate:.6f} (error: {abs(pi_estimate-math.pi):.6f})")
        print(f"Serial time: {t_serial:.3f}s")


