import numpy as np
import matplotlib.pyplot as plt


def rosenbrock_2D(x1, x2, a=1, b=100):
    return (300 - ((a - x1)**2 + b * (x2 - x1**2)**2)) / 300


# Define the domain
x1 = np.linspace(-2, 2, 500)
x2 = np.linspace(-1, 3, 500)

X1, X2 = np.meshgrid(x1, x2)

# Evaluate function on the grid
Z = rosenbrock_2D(X1, X2)

# Create contour plot
plt.figure(figsize=(8, 6))

contour = plt.contourf(X1, X2, Z, levels=50)
plt.colorbar(contour, label="f(x₁, x₂)")

# Safety threshold f(x₁, x₂) = 0
plt.contour(
    X1,
    X2,
    Z,
    levels=[0],
    colors="red",
    linestyles="dotted",
    linewidths=1,
)
plt.plot([], [], color="red", linestyle="dotted", linewidth=1, label="Safety threshold (f = 0)")

# Add contour lines
plt.contour(X1, X2, Z, levels=20, linewidths=0.5)

# Mark optimum
plt.scatter(
    1,
    1,
    marker="*",
    s=150,
    label="Optimum"
)

plt.xlabel("x₁")
plt.ylabel("x₂")
plt.title("2D Rosenbrock Function")
plt.legend()

plt.savefig("rosenbrock_figure.png", dpi=150)
print("saved")