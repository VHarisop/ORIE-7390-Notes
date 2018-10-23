#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

def gradient_updates(A, Xk, Yk, gamma, kappa):
    """
    Computes the gradient updates with respect to X, Y, given the target matrix
    A and the iterates Xk, Yk. gamma, kappa are positive parameters with gamma
    > 1.
    """
    cost = lambda X, Y: (X @ Y - A)
    xFro = max(np.linalg.svd(Yk, compute_uv=False)) ** 2
    lipX = gamma * max(kappa, xFro)
    Xnew = Xk - (1 / lipX) * cost(Xk, Yk) @ Yk.T
    yFro = max(np.linalg.svd(Xk, compute_uv=False)) ** 2
    lipY = gamma * max(kappa, yFro)
    Ynew = Yk - (1 / lipY) * Xnew.T @ cost(Xnew, Yk)
    return Xnew, Ynew


def prox_gradient(A, r, iters, gamma=1.5, kappa=1.0):
    """
    Implements the proximal gradient method for unconstrained matrix
    factorization.

    Arguments
    ---------
    A : numpy.array
        The product matrix
    r : int
        The rank of the desired factorization
    iters : int
        The number of iterations to run for
    gamma, kappa : float
        Step size parameters. `gamma > 1` and `kappa > 0` to be amenable to a
        KL-based convergence analysis.

    Returns
    -------
    dists : numpy.array
        An array containing the distances from the solution over all elapsed
        iterations
    """
    n, _ = A.shape
    X0 = np.random.randn(n, r)
    Y0 = np.random.randn(r, n)
    Xk, Yk = np.copy(X0), np.copy(Y0)
    cost = lambda X, Y: (1 / 2) * np.linalg.norm(A - X @ Y, 'fro') ** 2
    dists = np.zeros(iters)
    dists[0] = cost(Xk, Yk)
    for i in range(1, iters):
        Xk[:], Yk[:] = gradient_updates(A, Xk, Yk, gamma, kappa)
        dists[i] = cost(Xk, Yk)
    return dists


def svd_approx(A, r):
    """
    Approximates `A` using the singular value decomposition to obtain a
    low-rank matrix factorization.

    Arguments
    ---------
    A : numpy.array
        The product matrix
    r : int
        The desired rank of the factorization

    Returns
    -------
    float
        The distance of the solution
    """
    U, S, V = np.linalg.svd(A, compute_uv=True, full_matrices=False)
    S[r:] = 0.0
    return np.linalg.norm(A - np.dot(U * S, V), "fro")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares the proximal gradient and the SVD "
                    "approaches for low-rank approximation")
    parser.add_argument("--dim", help="The dimension of the matrix",
                        type=int, default=50)
    parser.add_argument("--rank", help="The rank of the matrix",
                        type=int, default=15)
    parser.add_argument("--iters", help="The number of iterations",
                        type=int, default=100)
    parser.add_argument("--gamma", help="The parameter gamma of the algorithm",
                        type=float, default=1.5)
    parser.add_argument("--kappa", help="The parameter kappa of the algorithm",
                        type=float, default=0.5)
    args = vars(parser.parse_args())
    dim, rank, iters = args['dim'], args['rank'], args['iters']
    gamma, kappa = args['gamma'], args['kappa']
    A = np.random.randn(dim, rank) @ np.random.randn(rank, dim)
    dists = prox_gradient(A, rank, iters, gamma, kappa)
    np.savetxt("prox_dim_{}_rank_{}.csv".format(dim, rank), dists,
               delimiter=",", comments="")
    err_svd = svd_approx(A, rank)
    print("SVD Error: %e - ProxGrad Error: %e" % (err_svd, dists[-1]))
