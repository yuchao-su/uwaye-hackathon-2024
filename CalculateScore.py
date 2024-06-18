import numpy as np
import matplotlib.pyplot as plt
import math

def bezier_curve(control_points, num_points=100):
    """
    计算贝塞尔曲线的点。

    :param control_points: 控制点的坐标列表。
    :param num_points: 要生成的曲线点的数量。
    :return: 贝塞尔曲线上的点的坐标列表。
    """
    n = len(control_points) - 1
    t_values = np.linspace(0, 1, num_points)
    curve_points = []

    for t in t_values:
        point = np.zeros(2)
        for i, p in enumerate(control_points):
            bernstein_poly = (math.factorial(n) /
                              (math.factorial(i) * math.factorial(n - i))) * (t ** i) * ((1 - t) ** (n - i))
            point += bernstein_poly * np.array(p)
        curve_points.append(point)

    return np.array(curve_points)


# 示例控制点
control_points_cubic = [(0, 0), (1, 3), (3, 3), (4, 0)]  # 三阶贝塞尔曲线控制点

# 计算贝塞尔曲线
bezier_points_cubic = bezier_curve(control_points_cubic)

# 绘制贝塞尔曲线
plt.figure(figsize=(10, 5))
# 绘制三阶贝塞尔曲线
plt.subplot(1, 2, 2)
plt.plot(bezier_points_cubic[:, 0], bezier_points_cubic[:, 1], label='Cubic Bezier Curve')
plt.plot(*zip(*control_points_cubic), 'ro--', label='Control Points')
plt.title('Cubic Bezier Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
