import numpy as np
import matplotlib.pyplot as plt

# Try to import environment mode from GUI
try:
    from environment_gui import get_mode
except ImportError:
    def get_mode():
        return "normal"  # fallback mode if GUI is not running


class RobotSimulator:
    def __init__(self, dt=0.1):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # robot orientation
        self.dt = dt
        self.path = [(self.x, self.y)]
        self.t = 0.0  # simulation time

    def step(self, v, omega):
        """Single simulation step, affected by environment mode"""
        mode = get_mode()

        if mode == "heavy":
            v *= 0.5  # slower movement under heavy load
        elif mode == "slippery":
            omega *= 0.3  # reduced turning ability
            omega = np.clip(omega, -2.5, 2.5)  # يسمح بتوجيه أوسع عند الانزلاق

        # Apply motion update
        self.x += v * np.cos(self.theta) * self.dt
        self.y += v * np.sin(self.theta) * self.dt
        self.theta += omega * self.dt
        self.path.append((self.x, self.y))
        self.t += self.dt

    def reset(self):
        self.__init__(self.dt)

    def plot_path(self):
        x_vals, y_vals = zip(*self.path)
        plt.figure()
        plt.plot(x_vals, y_vals, label="Robot Path")
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.title("Robot Path")
        plt.show()
