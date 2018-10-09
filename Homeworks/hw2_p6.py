#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

def gradient_updates(A, Xk, Yk):
    """
    Computes the gradient updates with respect to X, Y, given the target matrix
    A and the iterates Xk, Yk.
    """
    cost = lambda X, Y: (X @ Y - A)
    lipX = max(np.linalg.svd(Yk, compute_uv=False)) ** 2
    Xnew = Xk - (1 / lipX) * cost(Xk, Yk) @ Yk.T
    lipY = max(np.linalg.svd(Xk, compute_uv=False)) ** 2
    Ynew = Yk - (1 / lipY) * Xnew.T @ cost(Xnew, Yk)
    return Xnew, Ynew


def prox_gradient(A, r, iters):
    n, _ = A.shape
    X0 = np.random.randn(n, r)
    Y0 = np.random.randn(r, n)
    Xk, Yk = np.copy(X0), np.copy(Y0)
    cost = lambda X, Y: (1 / 2) * np.linalg.norm(A - X @ Y, 'fro') ** 2
    dists = np.zeros(iters)
    dists[0] = cost(Xk, Yk)
    for i in range(1, iters):
        Xk[:], Yk[:] = gradient_updates(A, Xk, Yk)
        dists[i] = cost(Xk, Yk)
    return dists

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
    args = vars(parser.parse_args())
    dim, rank, iters = args['dim'], args['rank'], args['iters']
    A = np.random.randn(dim, rank) @ np.random.randn(rank, dim)
    dists = prox_gradient(A, rank, iters)
    np.savetxt("prox_dim_{}_rank_{}.csv".format(dim, rank), dists,
               delimiter=",", comments="")
