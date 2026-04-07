from re import match
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
from multiprocessing import Pool
import os
import psutil
from pathlib import Path
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask

"""
Mandelbrot Set Generator
Author : [Jakob Rundlett]
Course : Numerical Scientific Computing 2026
"""

# MP1

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

@njit(cache=True)
def compute_mandelbrot_naive_numba(x_min, x_max, x_res, y_min, 
                                   y_max, y_res, max_iter, dtype=np.float32):
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

# MP2

def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

def estimate_pi_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle

def estimate_pi_parallel(num_samples, num_processes=4):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    for row in range(row_end - row_start):
        c_imag = y_min + (row + row_start) * dy
        for col in range(N):
            out[row, col] = mandelbrot_pixel(x_min + col*dx, c_imag, max_iter)
    return out

def mandelbrot_serial(x_min, x_max, N1, y_min, y_max, N2, max_iter):
    return mandelbrot_chunk(0, N1, N2, x_min, x_max, y_min, y_max, max_iter)

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel_old(N, x_min, x_max, y_min, y_max, max_iter, n_workers=4):
    chunk_size = max(1, N // n_workers)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=n_workers) as pool:
        pool.map(_worker, chunks) # un-timed warmup
        parts = pool.map(_worker, chunks)
    
    return np.vstack(parts)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers=4, n_chunks=None, pool=None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    if pool is not None: # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny) # warm-up: load JIT cache in workers
        parts = p.map(_worker, chunks)
    return np.vstack(parts)

def mandelbrot_dask(N, x_min, x_max, y_min, y_max, max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

if __name__ == '__main__':
    """ TEST CASES: 
         0: do nothing
         1: run the functions normally
         2: check performance of row vs col sums for numpy
         3: check the profiles for the original functions
         4: data type test
         5: Monte Carlo pi estimation test
         6: Monte Carlo pi estimation test *in parallel*
         7: run parallel mandelbrot computation and save the image
         8: analyze performance of parallel mandelbrot, testing optimal number of workers
         9: same, but chat version
        10: analyze performance of parallel mandelbrot, testing optimal number of chunks
        11: exhaustive chunk and worker test (doesn't work well)
        12: test dask distributed
        13: test mandelbrot_dask
        14: sweep chunk size for dask implementation
        15: test mandelbrot_dask using strato
        16: sweep chunk size for strato implementation
        17: smoke test
    """
    test_case = 17
    match test_case:
        case 0: print("Doing nothing...")
        case 1:
            gen_res_list = [128, 256, 512, 1024, 2048, 4096, 4096*2, 4096*4]
            gen_res  = 2048
            max_iter = 100
            n_runs   = 3

            run_naive        = False
            run_numpy        = False
            run_hybrid_numba = False
            run_naive_numba  = False
            run_serial       = False
            run_parallel     = False

            run_version = [run_naive,        run_numpy, 
                        run_hybrid_numba, run_naive_numba, 
                        run_serial,       run_parallel]

            view_image = False
            save_image = False
            plot_times = False # only use if using gen_res_list

            versions = [compute_mandelbrot_naive, compute_mandelbrot_numpy, 
                        compute_mandelbrot_hybrid_numba, compute_mandelbrot_naive_numba, 
                        mandelbrot_serial, mandelbrot_parallel]
            # run all the functions (idk what the point of this is)
            if run_version.count(False) == 0:
                for func in versions:
                    _, _ = benchmark(func, -2, 1, gen_res, -1.5, 1.5, gen_res, 
                                    max_iter, n_runs=n_runs)

            # compare the first two versions
            elif run_version.count(True) > 1:
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

            # run a single version
            else:
                _, result = benchmark(versions[run_version.index(True)], -2, 1, gen_res, -1.5, 1.5, gen_res, max_iter, n_runs=n_runs)
                print('finished version...')

                if view_image or save_image:
                    plt.figure(figsize=(10, 10))
                    plt.axis('off')
                    plt.imshow(result)
                    if save_image:
                        plt.savefig('mandelbrotFigure.png', dpi=gen_res/10)
                    if view_image:
                        plt.show()
        case 2:
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
        case 3:
            # x_min, x_max, x_res, y_min, y_max, y_res, max_iter
            cProfile.run('compute_mandelbrot_naive(-2, 1, 512, -1.5, 1.5, 512, 100)', 'naive_profile.prof')
            cProfile.run('compute_mandelbrot_numpy(-2, 1, 512, -1.5, 1.5, 512, 100)', 'numpy_profile.prof')

            for name in ('naive_profile.prof', 'numpy_profile.prof'):
                stats = pstats.Stats(name)
                stats.sort_stats('cumulative')
                stats.print_stats(10)
        case 4:
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
        case 5:
            # First we check how long it takes to run the naive Monte Carlo with 
            # 10,000,000 samples
            num_samples = 10**7
            _, pi_estimate = benchmark(estimate_pi_serial, num_samples)
            print(f"Estimated pi with {num_samples} \
                samples: {pi_estimate} \
                    (error: {abs(pi_estimate-math.pi):.6f})")
        case 6:
            # Now we do the same but for parellel computing for every number of 
            # worker-count from 1 to cpu_count()
            # if __name__ == '__main__':
            num_samples = 10**7
            worker_times = {}
            for num_proc in range(1, os.cpu_count() + 1):
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    pi_est = estimate_pi_parallel(num_samples, num_proc)
                    times.append(time.perf_counter() - t0)
                t = statistics.median(times)
                # print(f"{num_proc:2d} workers: {t:.3f}s pi={pi_est:.6f}")
                worker_times[num_proc] = t
                speed_up = worker_times[1] / t if num_proc > 1 else 1
                efficiency = (speed_up / num_proc)*100
                print(f'{num_proc:2d} workers | \
    {t:.3f}s | {speed_up:.2f}x | \
    {efficiency:.1f}%')
        case 7:
            # if __name__ == '__main__':
            result = mandelbrot_parallel(1024, -2.5, 1.0, -1.25, 1.25, 100, n_workers=4)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(result, extent=[-2.5, 1.0, -1.25, 1.25],
                    cmap='inferno', origin='lower', aspect='equal')
            ax.set_xlabel('Re(c)')
            ax.set_ylabel('Im(c)')
            out = Path(__file__).parent / 'mandelbrot.png'
            fig.savefig(out, dpi=150)
            print (f'Saved: {out}')
        case 8: # Best time with 15 workers: 0.012s, speedup=4.11x, eff=27%
            # if __name__ == '__main__':
            # --- MP2 M3: benchmark (in __main__ block) ---
            N, max_iter = 1024, 100
            X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
            
            # Serial baseline (Numba already warm after M1 warm-up)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_serial(X_MIN, X_MAX, N, Y_MIN, Y_MAX, N, max_iter)
                times.append(time.perf_counter() - t0)
            t_serial = statistics.median(times)
            
            for n_workers in reversed(range(1, os.cpu_count() + 1)):
                chunk_size = max(1, N // n_workers)
                chunks, row = [], 0
                while row < N:
                    end = min(row + chunk_size, N)
                    chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
                    row = end
                with Pool(processes=n_workers) as pool:
                    pool.map(_worker, chunks) # warm-up: Numba JIT in all workers
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        np.vstack(pool.map(_worker, chunks))
                        times.append(time.perf_counter() - t0)
                t_par = statistics.median(times)
                speedup = t_serial / t_par
                print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")
        case 9: # Result differs each time. I think more is better. Sometimes not the case with lower resolution, but definitely at higher ones
            # if __name__ == '__main__':
            N, max_iter = 1024, 100
            X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
            
            # Serial baseline
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_serial(X_MIN, X_MAX, N, Y_MIN, Y_MAX, N, max_iter)
                times.append(time.perf_counter() - t0)
            t_serial = statistics.median(times)
            
            results = []
            for n_workers in range(5, os.cpu_count() + 1):  # Forward iteration
                chunk_size = max(1, N // n_workers)
                chunks, row = [], 0
                while row < N:
                    end = min(row + chunk_size, N)
                    chunks.append((row, end, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter))
                    row = end
                
                with Pool(processes=n_workers) as pool:
                    pool.map(_worker, chunks)  # warm-up
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        np.vstack(pool.map(_worker, chunks))
                        times.append(time.perf_counter() - t0)
                
                t_par = statistics.median(times)
                speedup = t_serial / t_par
                efficiency = speedup / n_workers * 100
                results.append((n_workers, t_par, speedup, efficiency))
                print(f"{n_workers:2d} workers: {t_par:.3f}s, speedup={speedup:.2f}x, eff={efficiency:.0f}%")
            
            # Find and report optimal configuration
            best = max(results, key=lambda x: x[2])  # Max by speedup
            print(f"\nOptimal: {best[0]} workers with {best[2]:.2f}x speedup")
        case 10: # best lif score with 8 workers and 64 chunks: 0.042s, LIF=0.73, speedup=4.6x, (with 2048x2048)
            # if __name__ == '__main__':
            N, max_iter = 1024*2, 100
            n_workers = 8 # adjust to your L04 optimum
            X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
            
            mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter) # warm up JIT
            
            # Serial baseline
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
                times.append(time.perf_counter() - t0)
            t_serial = statistics.median(times)
            print(f"Serial: {t_serial:.3f}s")
            
            # Chunk-count sweep (M2): one Pool per config
            tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
            for mult in [1, 2, 4, 8, 16]:
                n_chunks = mult * n_workers
                with Pool(processes=n_workers) as pool:
                    pool.map(_worker, tiny) # warm-up: load JIT cache in workers
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_workers=n_workers, n_chunks=n_chunks, pool=pool)
                        times.append(time.perf_counter() - t0)
                t_par = statistics.median(times)
                lif = n_workers * t_par / t_serial - 1
                print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")
        case 11:
            # if __name__ == '__main__':
            for n_workers in reversed(range(1, os.cpu_count() + 1)):
                time.sleep(30)
                N, max_iter = 1024, 100
                print(f"Testing with {n_workers} workers...")
                X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
                
                mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
                
                # Serial baseline
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
                    times.append(time.perf_counter() - t0)
                t_serial = statistics.median(times)
                print(f"Serial: {t_serial:.3f}s")
                
                # Create ONE pool for this n_workers configuration
                with Pool(processes=n_workers) as pool:
                    tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]
                    pool.map(_worker, tiny)  # warm-up
                    
                    # Chunk-count sweep with the SAME pool
                    for mult in [1, 2, 4, 8, 16]:
                        n_chunks = mult * n_workers
                        times = []
                        for _ in range(3):
                            t0 = time.perf_counter()
                            mandelbrot_parallel(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, 
                                                n_workers=n_workers, n_chunks=n_chunks, pool=pool)
                            times.append(time.perf_counter() - t0)
                        t_par = statistics.median(times)
                        lif = n_workers * t_par / t_serial - 1
                        print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")
                # Pool exits here only once per n_workers value
        case 12:
            from dask.distributed import Client, LocalCluster
            cluster = LocalCluster(n_workers=4, threads_per_worker=1)
            client = Client(cluster)
            print(client.dashboard_link)   # should print a localhost URL
            client.close(); cluster.close()
        case 13:
            N, max_iter = 1024*2, 100
            n_chunks = 8
            X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
            cluster = LocalCluster(n_workers=8, threads_per_worker=1)
            client = Client(cluster)
            client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)) # warm up all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter, n_chunks=n_chunks)
                times.append(time.perf_counter() - t0)
            print(f"Dask local (n_chunks={n_chunks}): {statistics.median(times):.3f} s")
            client.close(); cluster.close()
        case 14: # best lif scores... 
            # with  8 chunks: 0.111s, LIF=2.31, speedup=2.4x, (with 2048x2048)
            # with 48 chunks: 0.306s, LIF=1.73, speedup=2.9x, (with 4096x4096)
            N, max_iter = 1024*4, 100
            n_workers = 8 # adjust to your L04 optimum
            X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
            
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
            client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)) # warm up all workers

            for n_chunks in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
                time.sleep(5)
                _ = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter, n_chunks=n_chunks)
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter, n_chunks=n_chunks)
                    times.append(time.perf_counter() - t0)
                t_cur = statistics.median(times)
                if n_chunks == 1:
                    t_1x = t_cur
                lif = n_workers * t_cur / t_1x - 1
                print(f"{n_chunks:4d} chunks | {t_cur:.3f}s | {t_1x/t_cur:.1f}x | LIF={lif:.2f}")
            client.close(); cluster.close()
        case 15:
            N, max_iter = 1024*2, 100
            n_chunks = 8
            X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
            # Remove these two lines:
            # cluster = LocalCluster(n_workers=8, threads_per_worker=1)
            # client = Client(cluster)
            # Replace with:
            client = Client("tcp://10.92.1.74:8786")

            client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)) # warm up all workers
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter, n_chunks=n_chunks)
                times.append(time.perf_counter() - t0)
            print(f"Dask strato (n_chunks={n_chunks}): {statistics.median(times):.3f} s")
            client.close() # keep this
            # cluster.close() # remove this
        case 16:
            N, max_iter = 1024*4, 100
            X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

            # Replace with:
            client = Client("tcp://10.92.1.74:8786")
            versions = client.run(lambda: __import__('dask').__version__)
            print(versions)   # all values must be identical

            client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)) # warm up all workers

            # Sweep n_chunks to find optimal value
            n_chunks_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
            results = {}

            for n_chunks in n_chunks_values:
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter, n_chunks=n_chunks)
                    times.append(time.perf_counter() - t0)
                
                median_time = statistics.median(times)
                lif_score = median_time * n_chunks  # LIF = Latency × Items per Batch (or similar metric)
                results[n_chunks] = lif_score
                print(f"Dask strato (n_chunks={n_chunks}): median_time={median_time:.3f} s, LIF score={lif_score:.3f}")

            # Find optimal n_chunks (minimum LIF score)
            optimal_n_chunks = min(results, key=results.get)
            optimal_lif = results[optimal_n_chunks]
            print(f"\nOptimal n_chunks: {optimal_n_chunks} with minimum LIF score: {optimal_lif:.3f}")
            client.close() # keep this


