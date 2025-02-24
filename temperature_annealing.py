"""predefined annealing schedules for the temperature in the simulated annealing algorithm"""

import numpy as np


def linear(beta_0, beta_f, frame_number, frame_count):
    return beta_0 + (beta_f - beta_0) * (frame_number / (frame_count + 1))


def exponential(beta_0, beta_f, frame_number, frame_count):
    return beta_0 * ((beta_f / beta_0) ** (frame_number / (frame_count + 1)))


def logarithmic(beta_0, beta_f, frame_number, frame_count):
    if frame_count == 0:
        return beta_0
    return beta_0 + (beta_f - beta_0) * np.log(1 + frame_number) / np.log(
        1 + frame_count
    )


def inverse(beta_0, beta_f, frame_number, frame_count):
    return beta_0 / (1 + (beta_0 - beta_f) * frame_number / (frame_count + 1))


def quadratic(beta_0, beta_f, frame_number, frame_count):
    return beta_0 + (beta_f - beta_0) * (frame_number / (frame_count + 1)) ** 2


def power_law(beta_0, beta_f, frame_number, frame_count, power):
    return beta_0 + (beta_f - beta_0) * (frame_number / (frame_count + 1)) ** power
