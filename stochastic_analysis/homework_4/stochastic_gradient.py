import matplotlib.pyplot as plt
import numpy as np
import random


def randomly_populate_matrix():
    for i in range(n):
        for j in range(n):
            A[i, j] = np.random.normal(0, np.sqrt(1 / n))


# Since matrix R contains the single non-zero element n in position J, J,
# the product of A* and R is the matrix having at the J column the J row
# from matrix A multiplied by n (let's denote this vector as c), and zeroes in all other places
# The final product 2 A* R (Ax - b) will be 2 * c * (Ax - b)[J]
# Now, (Ax - b)[J] is (dot product of A[J] and x) - b[J]
# The computational cost of this operation is O(n)
def solve():
    print('Exact solution:')
    print(np.linalg.solve(A, b))
    # initialize x and g with random values from -1 to 1
    x = (np.random.rand(n) - 0.5) * 2.
    g = (np.random.rand(n) - 0.5) * 2.
    norm = np.linalg.norm(g, 2)
    initial_norm = norm
    print('initial_norm', initial_norm)
    sample_size = min(100, n)  # int(np.round(np.sqrt(n))) + 1 #n
    iteration_count = 0
    # if iteration_count exceeds 1,000,000 we consider the precess divergent
    x_ticks = []
    x_vals = []
    while norm > 0.001 and iteration_count < 1000000:
        iteration_count += 1
        for _ in range(sample_size):
            J = random.randrange(0, n)
            g += 2 * A[J] * n * (np.dot(A[J], x) - b[J])
        g = g / sample_size
        x = x - epsilon * g
        norm = np.linalg.norm(g, 2)
        if iteration_count % 500 == 0:
            # print('count', iteration_count, 'norm', norm)
            x_ticks.append(iteration_count)
            x_vals.append(norm)
    print('Calculated solution')
    print(x)
    print('\n\nnorm: ', norm, 'count:', iteration_count)
    plt.plot(x_ticks, x_vals)
    plt.show()


def set_globals(_n):
    global n
    global A
    global b
    n = _n
    A = np.empty([n, n])
    randomly_populate_matrix()
    b = np.empty(n)
    b.fill(0)
    b[0] = 1
    eigvals = np.linalg.eigvals(np.matmul(A.transpose(), A))
    maximal_eigval = eigvals.max()
    global epsilon
    epsilon = 1 / (maximal_eigval ** 2)
    print('maximal_eigval', maximal_eigval)
    print('epsilon', epsilon)


def main():
    set_globals(1000)
    solve()


if __name__ == '__main__':
    main()
