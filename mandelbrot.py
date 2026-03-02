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

def benchmark(func, *args, n_runs=5) :
    """ Time func, return median of n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter()-t0)
    median_t = statistics.median(times)
    # print(f"Median: {median_t:.4f}s" f"(min={min(times):.4f}, max={max(times):.4f})")
    return median_t, result

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

def test_row():
    N = 10000
    A = np.random.rand(N, N)
    for i in range(N): s = np.sum(A[i, :])

def test_col():
    N = 10000
    A = np.random.rand(N, N)
    for j in range(N): s = np.sum(A[:, j])

gen_res = 1024
max_iter = 100
n_runs = 3

print(f'Running naive version {n_runs}x...')
naive_time, naive_result = benchmark(compute_mandelbrot_naive, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
print(f'Naive computation took {naive_time:.3f} seconds')

print(f'Running numpy version {n_runs}x...')
numpy_time, numpy_result = benchmark(compute_mandelbrot_numpy, -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
print(f'Numpy computation took {numpy_time:.3f} seconds')

check_diff = True
save_image = True
view_pair = False

# check the strict numerical difference between the results
if check_diff:
    if np.allclose(naive_result, numpy_result):
        print("Results match!")
    else:
        print("Results differ!")
        diff = np.abs(naive_result - numpy_result)
        print(f"Max difference: {diff.max()}")
        print(f"Different pixels: {(diff > 0).sum()}")

# plot the results next to each other
if view_pair:
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display the Mandelbrot set
    ax1.imshow(numpy_result)
    ax1.set_title('Mandelbrot Set')
    ax1.axis('off')  # Hide axes if desired

    # Display the naive results
    ax2.imshow(naive_result)
    ax2.set_title('Naive Results')
    ax2.axis('off')  # Hide axes if desired

    # Show the plots
    plt.tight_layout()
    if save_image:
        plt.savefig('mandelbrotFigurePair')
    else:
        plt.show()
else:
    plt.imshow(numpy_result)
    plt.title('Mandelbrot set')
    plt.colorbar()
    if save_image:
        plt.savefig('mandelbrotFigure')
    else:
        plt.show()
