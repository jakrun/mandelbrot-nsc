import numpy as np
import matplotlib.pyplot as plt
import time

"""
Mandelbrot Set Generator
Author : [Jakob Rundlett]
Course : Numerical Scientific Computing 2026
"""

x = np.linspace (-2,   1,   1024) # x resolution
y = np.linspace (-1.5, 1.5, 1024) # y resolution
X, Y = np.meshgrid(x, y) # standard grid
C = X + 1j * Y # complex grid
max_iter = 100

# return the number of iterations given a complex number
def mandelbrot_points():
    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    for n in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

# make a new grid, converting the complex numbers to their 'maximum iterations'
def compute_mandelbrot():
    I = mandelbrot_points()
    return I

print(C.shape)
start_time = time.time()
mandelbrot_result = compute_mandelbrot()
elapsed_time = time.time() - start_time
print(f'Computation took {elapsed_time:.3f} seconds')

plt.imshow(mandelbrot_result)
plt.title('Mandelbrot set')
plt.colorbar()
plt.savefig('mandelbrotFigure')
# plt.show()