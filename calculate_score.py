import numpy as np
from scipy.optimize import least_squares
import math
import matplotlib.pyplot as plt

# Assuming the data has been preprocessed. Irrelevant data has been ignored
# Reads coordinate data from a file
points = np.array(
    [[5.00000000e+00, 0.00000000e+00],
     [4.99743108e+00, 1.60257888e-01],
     [4.98972696e+00, 3.20351100e-01],
     [4.97689556e+00, 4.80115130e-01],
     [4.95895007e+00, 6.39385808e-01],
     [4.93590892e+00, 7.97999475e-01],
     [4.90779578e+00, 9.55793144e-01],
     [4.87463956e+00, 1.11260467e+00],
     [4.83647432e+00, 1.26827292e+00],
     [4.79333927e+00, 1.42263793e+00],
     [4.74527874e+00, 1.57554109e+00],
     [4.69234211e+00, 1.72682527e+00],
     [4.63458379e+00, 1.87633502e+00],
     [4.57206312e+00, 2.02391672e+00],
     [4.50484434e+00, 2.16941870e+00],
     [4.43299653e+00, 2.31269145e+00],
     [4.35659352e+00, 2.45358776e+00],
     [4.27571382e+00, 2.59196284e+00],
     [4.19044052e+00, 2.72767451e+00],
     [4.10086127e+00, 2.86058330e+00],
     [4.00706811e+00, 2.99055265e+00],
     [3.90915741e+00, 3.11744901e+00],
     [3.80722979e+00, 3.24114198e+00],
     [3.70138999e+00, 3.36150445e+00],
     [3.59174675e+00, 3.47841275e+00],
     [3.47841275e+00, 3.59174675e+00],
     [3.36150445e+00, 3.70138999e+00],
     [3.24114198e+00, 3.80722979e+00],
     [3.11744901e+00, 3.90915741e+00],
     [2.99055265e+00, 4.00706811e+00],
     [2.86058330e+00, 4.10086127e+00],
     [2.72767451e+00, 4.19044052e+00],
     [2.59196284e+00, 4.28571382e+00],
     [2.45358776e+00, 4.32659352e+00],
     [2.31269145e+00, 4.43299653e+00],
     [2.16941870e+00, 4.50484434e+00],
     [2.02391672e+00, 4.57206312e+00],
     [1.87633502e+00, 4.63458379e+00],
     [1.72682527e+00, 4.69234211e+00],
     [1.57554109e+00, 4.74527874e+00],
     [1.42263793e+00, 4.79333927e+00],
     [1.26827292e+00, 4.83647432e+00],
     [1.11260467e+00, 4.87463956e+00],
     [9.55793144e-01, 4.90779578e+00],
     [7.97999475e-01, 4.93590892e+00],
     [6.39385808e-01, 4.95895007e+00],
     [4.80115130e-01, 4.97689556e+00],
     [3.20351100e-01, 4.98972696e+00],
     [1.60257888e-01, 4.99743108e+00],
     [3.06161700e-16, 5.00000000e+00]])

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
    plt.plot(points[:, 0], points[:, 1], 'ro-', markersize=5, label='Original Points')  # Adjusted marker size here
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Bezier Curve')
    plt.plot(control_points[:, 0], control_points[:, 1], 'gx--', label='Control Points')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curve Fitting')
    plt.show()

degree = 5
control_points = fit_bezier(points, degree)
curve_points = calculate_bezier_points(control_points)
length = curve_length(curve_points)
curvatures = curvature(curve_points)
average_curvature = np.mean(curvatures)

print("Curve length:", length)
print("average curvature:", average_curvature)
print('score:', length * 8 - average_curvature * 2)

plot_curve(points, curve_points, control_points)



