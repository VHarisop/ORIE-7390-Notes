#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np


def soft_thres(xk, l):
    return np.sign(xk) * np.maximum(0, np.abs(xk) - l)


def l2_update(xk, l):
    return (l / (2 * l + 1)) * xk


def experiment_l1(dim, iters, l_init):
    """
    Run the proximal point method on the \ell_1 norm.
    """
    x0 = 5 * np.random.randn(dim)
    hist = np.zeros(iters)
    lbds = np.geomspace(l_init, 0.1 * l_init, iters)
    for idx, l in enumerate(lbds):
        hist[idx] = np.linalg.norm(x0, 1)
        x0[:] = soft_thres(x0, l)
    return hist


def experiment_l2(dim, iters, l_init):
    """
    Run the proximal point method on the squared \ell_2 norm.
    """
    x0 = 5 * np.random.randn(dim)
    hist = np.zeros(iters)
    lbds = np.geomspace(l_init, 0.1 * l_init, iters)
    for idx, l in enumerate(lbds):
        hist[idx] = np.linalg.norm(x0) ** 2
        x0[:] = l2_update(x0, l)
    return hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment to validate arguments "
                    "about lojasiewicz property")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--l_init", type=float, default=5.0)
    args = vars(parser.parse_args())
    # run experiments
    data = np.zeros((args["iters"], 3))
    data[:, 1] = experiment_l1(args["dim"], args["iters"], args["l_init"])
    data[:, 2] = experiment_l2(args["dim"], args["iters"], args["l_init"])
    data[:, 0] = np.arange(args["iters"]) + 1
    np.savetxt("experiment.csv", data, fmt="%d, %.8e, %.8e",
        header="k, l1, l2", comments="", delimiter=",")
