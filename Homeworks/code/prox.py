#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

def prox_sequence(lbd, xk):
    """
    Calculate the next proximal point given xk and a positive parameter lbd.
    """
    u, v = xk
    return np.array([
        np.sign(u) * np.maximum(np.abs(u) - (1 / lbd), 0),
        (lbd / (2 + lbd)) * v])

def calculate_prox_sequence(x0=(1.0, 1.0), lbd=4, iters=100):
    values = np.zeros((iters+1, 2))
    values[0, :] = np.array(x0)
    xi = np.array(x0)
    for i in range(iters):
        xi = prox_sequence(lbd, xi)
        values[i+1, :] = np.copy(xi)
    return values

if __name__ == "__main__":
    plt.rcParams['svg.fonttype'] = 'none'
    plt.xlabel(r"\( x(t) \)")
    plt.ylabel(r"\( y(t) \)")
    parser = argparse.ArgumentParser(
        description="Compare proximal and continuous descent trajectories")
    parser.add_argument("--traj_type", type=str,
        choices=['proximal', 'continuous'])
    parser.add_argument("--lambda", type=float, default=4.0)
    parser.add_argument("--end_time", type=float, default=50.0)
    parser.add_argument("--iters", type=int, default=200)
    args = vars(parser.parse_args())
    if args['traj_type'] == 'proximal':
        pts = calculate_prox_sequence(
            x0=(1.0, 1.0), lbd=args['lambda'], iters=args['iters'])
        save_name = ("prox_%.3f.csv" % args['lambda'])
        np.savetxt(save_name, pts, delimiter=",", header="x, y",
                   comments="", fmt="%.3f")
    else:
        pass
