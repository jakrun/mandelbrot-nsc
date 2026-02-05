import numpy as np
import matplotlib.pyplot as plt
import time

"""
Mandelbrot Set Generator
Author : [Jakob Rundlett]
Course : Numerical Scientific Computing 2026
"""

# define boundaries
region_x = (-2, 1)
region_y = (-1.5, 1.5)
# for 1024x1024
region_x_step = (region_x[1] - region_x[0]) / 1024
region_y_step = (region_y[1] - region_y[0]) / 1024
# create the pixel cooridnates
points_x = np.arange(region_x[0], region_x[1], region_x_step)
points_y = np.arange(region_y[0], region_y[1], region_y_step)

# return the number of iterations given a complex number
def mandelbrot_point(c, max_iter):
    z = [0 + 0j]
    for n in range(max_iter):
        new_z = z[-1]**2 + c
        if abs(new_z) > 2:
            return n
        else:
            z.append(new_z)
    return max_iter

def compute_mandelbrot():
    # create the grid of complex numbers for each pixel
    complex_grid = []
    for x in points_x:
        new_line = []
        for y in points_y:
            new_line.append(complex(x, y))
        complex_grid.append(new_line)

    complex_grid = np.array(complex_grid)

    # replace all the complex numbers with the corresponding number of iterations
    max_iter = 100
    for row in range(len(complex_grid)):
        for col in range(len(complex_grid[0])):
            complex_grid[row][col] = mandelbrot_point(complex_grid[row][col], max_iter)

    # I don't know why the grid still has complex numbers. They should all be replaced with integers
    complex_grid = complex_grid.astype(int)

    return complex_grid

start_time = time.time()
mandelbrot_result = compute_mandelbrot()
elapsed_time = time.time() - start_time
print(f'Computation took {elapsed_time:.3f} seconds')

plt.imshow(mandelbrot_result)
plt.title('Mandelbrot set')
plt.colorbar()
plt.savefig()
# plt.show()