import numpy as np
from scipy.optimize import least_squares
import math
import matplotlib.pyplot as plt
import random
import time

# Assuming the data has been preprocessed. Irrelevant data has been ignored
# Reads coordinate data from a file

# parameter
radius = 5
angle_start = 0
num = 100

def generate_points(angle_end):
    angles = np.linspace(angle_start, angle_end, num)
    points = np.array([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ]).T

    # 20% of the points are randomly selected for perturbation
    num_points_to_perturb = int(0.2 * num)
    perturb_indices = np.random.choice(num, num_points_to_perturb, replace=False)

    # Range of disturbance
    perturbation_magnitude = 0.005

    # The selected point is perturbed
    for idx in perturb_indices:
        perturbation = np.random.normal(0, perturbation_magnitude, 2)
        points[idx] += perturbation

    return points

# Function to calculate a point on a Bezier curve
def bezier_curve(t, control_points):
    n = len(control_points) - 1
    point = np.zeros(2)
    for i, p in enumerate(control_points):
        bernstein_poly = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        point += bernstein_poly * p
    return point

# Fit a Bezier curve to the points
def fit_bezier(points, degree):
    t_values = np.linspace(0, 1, len(points))
    initial_guess = np.linspace(points[0], points[-1], degree + 1)

    def residuals(control_points, t_values, points):
        control_points = control_points.reshape((degree + 1, 2))
        residuals = []
        for t, point in zip(t_values, points):
            curve_point = bezier_curve(t, control_points)
            residuals.append(curve_point - point)
        return np.ravel(residuals)

    result = least_squares(residuals, initial_guess.ravel(), args=(t_values, points))
    fitted_control_points = result.x.reshape((degree + 1, 2))
    return fitted_control_points

# Calculate points on the Bezier curve
def calculate_bezier_points(control_points, num_points=1000):
    t_values = np.linspace(0, 1, num_points)
    curve_points = np.array([bezier_curve(t, control_points) for t in t_values])
    return curve_points

# Calculate curve length
def curve_length(curve_points):
    diffs = np.diff(curve_points, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    return segment_lengths.sum()

# Calculate curve smoothness (curvature)
def curvature(curve_points):
    diffs = np.diff(curve_points, axis=0)
    second_diffs = np.diff(diffs, axis=0)
    num = np.abs(second_diffs[:, 1] * diffs[:-1, 0] - second_diffs[:, 0] * diffs[:-1, 1])
    denom = (diffs[:-1, 0]**2 + diffs[:-1, 1]**2)**1.5
    curvatures = num / denom
    return curvatures

# Plot the Bezier curve and original trajectory points
def plot_curve(points, curve_points, control_points):
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], 'ro-', markersize=3, label='Original Points')  # Adjusted marker size here
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Bezier Curve')
    plt.plot(control_points[:, 0], control_points[:, 1], 'gx--', label='Control Points')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curve Fitting')
    plt.show()

def doWork(angle_end):
    points = generate_points(angle_end)
    degree = 5
    control_points = fit_bezier(points, degree)
    curve_points = calculate_bezier_points(control_points)
    length = curve_length(curve_points)
    curvatures = curvature(curve_points)
    average_curvature = np.mean(curvatures)

    print("Curve length:", length)
    print("Average curvature:", average_curvature)
    print('Score:', length * 7 - average_curvature * 3)

    plot_curve(points, curve_points, control_points)

for i in range(1, 11):
    print(f"Analysing the curve: {i}")
    angle_end = np.pi / random.uniform(1.25, 2.5)
    doWork(angle_end)
    print("")
    time.sleep(1)
