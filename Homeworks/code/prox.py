#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

def prox_sequence(lbd, xk):
    """
    Calculate the next proximal point given xk and a positive parameter lbd.
    """
    u, v = xk
    return np.array([
        np.sign(u) * np.maximum(np.abs(u) - (1 / lbd), 0),
        (lbd / (2 + lbd)) * v])

def calculate_prox_sequence(x0=(1.0, 1.0), lbd=4, iters=100):
    """
    Compute the proximal mapping sequence using a specified [lbd] for [iters]
    iterations.
    """
    values = np.zeros((iters+1, 2))
    values[0, :] = np.array(x0)
    xi = np.array(x0)
    for i in range(iters):
        xi = prox_sequence(lbd, xi)
        values[i+1, :] = np.copy(xi)
    return values

def calculate_trajectory(x0, iters, end_time):
    """
    Calculate the trajectory from 0 to [end_time], sampling [iters] points in
    between.
    """
    values = np.zeros((iters, 2))
    t = np.linspace(0, end_time, iters)
    values[:, 0] = np.maximum(1 - t, 0)
    values[:, 1] = np.exp(-2 * t)
    return values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare proximal and continuous descent trajectories")
    parser.add_argument("--traj_type", type=str, choices=['proximal', 'continuous'])
    parser.add_argument("--lambda", type=float, default=4.0)
    parser.add_argument("--end_time", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=200)
    args = vars(parser.parse_args())
    x0, lbd, iters = (1.0, 1.0), args['lambda'], args['iters']
    if args['traj_type'] == 'proximal':
        pts = calculate_prox_sequence(x0, lbd, iters)
        save_name = ("prox_%.3f.csv" % lbd)
    else:
        pts = calculate_trajectory(x0, iters, args['end_time'])
        save_name = ("trajectory_%.3f.csv" % args["end_time"])
    np.savetxt(save_name, pts, delimiter=",", header="x, y",
               comments="", fmt="%.3f")
