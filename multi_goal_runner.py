import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from simulator.simulation import RobotSimulator
from controller.pid import PIDController
from optimizer.genetic__optimizer import genetic_optimization

try:
    from environment_gui import get_mode
except ImportError:
    def get_mode():
        return "normal"


def get_color_for_mode(mode: str):
    return {
        "normal": "blue",
        "heavy": "green",
        "slippery": "orange"
    }.get(mode, "gray")


def simulate_pid_on_multiple_goals(
    Kp, Ki, Kd, goals,
    visualize=False, save_video=True, loop=True, fps=20
):
    results = []

    trace = []

    pid_cache = {}
    last_mode = None

    if visualize:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("Live Robot Path")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        dot, = ax.plot([], [], 'ro')
        pid_text = ax.text(0.70, 0.02, "", transform=ax.transAxes, fontsize=10,
                           verticalalignment='bottom', bbox=dict(boxstyle="round", fc="w"))
        ax.legend()
        last_pos = None

    sim = RobotSimulator(dt=0.1)
    angle_pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd)
    distance_pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.05)

    while True:
        if visualize and not plt.get_fignums():
            print("🛑 Simulation window closed by user.")
            break

        current_mode = get_mode()

        if current_mode != last_mode:
            if current_mode in pid_cache:
                best_pid = pid_cache[current_mode]
                print(f"🧠 Loaded cached PID for mode: {current_mode}")
            else:
                print(f"🔄 Mode changed to: {current_mode} — optimizing PID...")
                best_pid = genetic_optimization(
                    population_size=6,
                    generations=5,
                    mutation_rate=0.4,
                    crossover_rate=0.7,
                    mode=current_mode
                )
                pid_cache[current_mode] = best_pid
                print(f"💾 Cached PID for mode: {current_mode}")

            angle_pid.Kp, angle_pid.Ki, angle_pid.Kd = best_pid
            last_mode = current_mode

        for goal in goals:
            max_steps = 300
            tolerance = 0.1
            total_time = 0.0

            for _ in range(max_steps):
                if visualize and not plt.get_fignums():
                    print("🛑 Simulation window closed by user.")
                    loop = False
                    break

                x, y, theta = sim.x, sim.y, sim.theta
                direction = np.arctan2(goal[1] - y, goal[0] - x)
                angle_error = (direction - theta + np.pi) % (2 * np.pi) - np.pi

                omega = angle_pid.compute(angle_error, sim.dt)
                distance = np.linalg.norm(goal - np.array([x, y]))
                v = max(0.0, min(distance_pid.compute(distance, sim.dt), 2.0))

                sim.step(v=v, omega=omega)
                total_time += sim.dt

                trace.append((sim.x, sim.y, get_mode()))

                if visualize:
                    current_pos = (sim.x, sim.y)
                    color = get_color_for_mode(get_mode())
                    if last_pos is not None:
                        xs, ys = zip(last_pos, current_pos)
                        ax.plot(xs, ys, color=color, linewidth=2)
                    last_pos = current_pos
                    dot.set_data([sim.x], [sim.y])
                    ax.relim(); ax.autoscale_view()
                    pid_text.set_text(
                        f"Kp={angle_pid.Kp:.2f}, Ki={angle_pid.Ki:.2f}, Kd={angle_pid.Kd:.2f}\nMode: {get_mode()}"
                    )
                    plt.pause(0.001)

                if distance < tolerance:
                    break

            final_error = np.linalg.norm(goal - np.array([sim.x, sim.y]))
            results.append({
                "Goal X": goal[0],
                "Goal Y": goal[1],
                "Time": round(total_time, 2),
                "Final Error": round(final_error, 4)
            })

            angle_pid.reset()
            distance_pid.reset()
            last_pos = None

        if not loop:
            break

    # حفظ النتائج
    os.makedirs("results", exist_ok=True)
    try:
        df = pd.DataFrame(results)
        df.to_excel(f"results/final_results_Kp{angle_pid.Kp}_Ki{angle_pid.Ki}_Kd{angle_pid.Kd}.xlsx", index=False)
        print("✅ Results saved to Excel.")
    except Exception as e:
        print(f"❌ Failed to save Excel: {e}")

    if save_video and len(trace) > 1:
        try:
            save_trace_video(trace, f"results/sim_combined_Kp{angle_pid.Kp}_Ki{angle_pid.Ki}_Kd{angle_pid.Kd}.mp4", fps=fps)
        except Exception as e:
            print(f"❌ Failed to save video: {e}")

    if visualize:
        plt.ioff(); plt.show()

    return results


def save_trace_video(trace, filename, fps=20):
    pts = [(x, y, m) for (x, y, m) in trace if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pts) < 2:
        print("⚠️ Not enough data to render video.")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ms = [p[2] for p in pts]

    fig, ax = plt.subplots()
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True)

    segments = [np.array([[xs[i], ys[i]], [xs[i+1], ys[i+1]]]) for i in range(len(xs) - 1)]
    colors = [get_color_for_mode(ms[i]) for i in range(len(xs) - 1)]

    lc = LineCollection([], linewidths=2)
    ax.add_collection(lc)
    dot, = ax.plot([], [], 'ro')

    def update(i):
        if i <= 0:
            lc.set_segments([]); return lc, dot
        lc.set_segments(segments[:i])
        lc.set_color(colors[:i])
        end_x, end_y = segments[i-1][1]
        dot.set_data([end_x], [end_y])
        return lc, dot

    ani = FuncAnimation(fig, update, frames=len(segments), interval=int(1000 / fps), blit=True)

    try:
        ani.save(filename, fps=fps, writer="ffmpeg")
        print(f"✅ Video saved: {filename}")
    except Exception as e:
        gif_name = filename.replace(".mp4", ".gif")
        print(f"⚠️ MP4 failed ({e}), saving GIF instead → {gif_name}")
        ani.save(gif_name, fps=fps, writer="pillow")
        print(f"✅ GIF saved: {gif_name}")

    plt.close(fig)
