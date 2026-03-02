import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
import warnings

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
    return result

def compute_mandelbrot_naive(x_min, x_max, x_res, y_min, y_max, y_res, max_iter):
    # for 1024x1024    
    region_x_step = (x_max - x_min) / x_res
    region_y_step = (y_max - y_min) / y_res
    # create the pixel cooridnates
    points_x = np.arange(x_min, x_max, region_x_step)
    points_y = np.arange(y_min, y_max, region_y_step)

    # return the number of iterations given a complex number
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

run_tests = False
if run_tests:
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
    _ = benchmark(test_row, N, A, n_runs=n_runs)
    _ = benchmark(test_col, N, A, n_runs=n_runs)
    print('With fortran np.array ...')
    _ = benchmark(test_row, N, A_f, n_runs=n_runs)
    _ = benchmark(test_col, N, A_f, n_runs=n_runs)

else:
    gen_res = 1024
    max_iter = 100
    n_runs = 3

    run_naive_version = False
    run_numpy_version = True
    view_image = False
    save_image = True

    if run_naive_version and run_numpy_version:
        naive_result = benchmark(compute_mandelbrot_naive, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
        numpy_result = benchmark(compute_mandelbrot_numpy, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
        
        # check the strict numerical difference between the results
        if np.allclose(naive_result, numpy_result):
            print("Results match!")
        else:
            print("Results differ!")
            diff = np.abs(naive_result - numpy_result)
            print(f"Max difference: {diff.max()}")
            print(f"Different pixels: {(diff > 0).sum()} ({(((diff > 0).sum()/(gen_res**2))*100):.1f}%)")

        # plot the results next to each other
        if view_image or save_image:
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Display the Mandelbrot set
            ax1.imshow(numpy_result)
            ax1.set_title('Numpy Result')
            ax1.axis('off')  # Hide axes if desired

            # Display the naive results
            ax2.imshow(naive_result)
            ax2.set_title('Naive Result')
            ax2.axis('off')  # Hide axes if desired

            # Show the plots
            plt.tight_layout()
            if save_image:
                plt.savefig('mandelbrotFigure')
            if view_image:
                plt.show()

    elif run_naive_version:
        naive_result = benchmark(compute_mandelbrot_naive, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
        
        if view_image or save_image:
            plt.imshow(naive_result)
            plt.title('Naive Result')
            plt.colorbar()
            if save_image:
                plt.savefig('mandelbrotFigure')
            if view_image:
                plt.show()
    
    elif run_numpy_version:
        numpy_result = benchmark(compute_mandelbrot_numpy, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)

        if view_image or save_image:
            plt.imshow(numpy_result)
            plt.title('Numpy Result')
            plt.colorbar()
            if save_image:
                plt.savefig('mandelbrotFigure')
            if view_image:
                plt.show()