import numpy as np
from simulator.simulation import RobotSimulator
from controller.pid import PIDController
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
from optimizer.genetic__optimizer import genetic_optimization

try:
    from environment_gui import get_mode
except ImportError:
    def get_mode():
        return "normal"

def get_color_for_mode(mode):
    return {
        "normal": "blue",
        "heavy": "green",
        "slippery": "orange"
    }.get(mode, "gray")

def simulate_pid_on_multiple_goals(Kp, Ki, Kd, goals, visualize=False, save_video=False, loop=True):
    results = []
    all_paths = []

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
        full_path = []
        lines = []
        last_pos = None
        last_color = None

    sim = RobotSimulator(dt=0.1)
    angle_pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd)
    distance_pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.05)

    try:
        while True:
            current_mode = get_mode()

            if current_mode != last_mode:
                if current_mode in pid_cache:
                    best_pid = pid_cache[current_mode]
                    print(f"Loaded cached PID for mode: {current_mode}")
                else:
                    print(f"Mode changed to: {current_mode} — optimizing PID...")
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

            for idx, goal in enumerate(goals):
                max_steps = 300
                tolerance = 0.1
                total_time = 0.0

                for _ in range(max_steps):
                    x, y, theta = sim.x, sim.y, sim.theta
                    direction = np.arctan2(goal[1] - y, goal[0] - x)
                    angle_error = (direction - theta + np.pi) % (2 * np.pi) - np.pi

                    omega = angle_pid.compute(angle_error, sim.dt)
                    distance = np.linalg.norm(goal - np.array([x, y]))
                    v = max(0.0, min(distance_pid.compute(distance, sim.dt), 2.0))

                    sim.step(v=v, omega=omega)
                    total_time += sim.dt

                    if visualize:
                        current_pos = (sim.x, sim.y)
                        current_mode = get_mode()
                        current_color = get_color_for_mode(current_mode)

                        if last_pos is not None:
                            xs, ys = zip(last_pos, current_pos)
                            new_line, = ax.plot(xs, ys, color=current_color, linewidth=2)
                            lines.append(new_line)

                        last_pos = current_pos
                        last_color = current_color

                        full_path.append(current_pos)
                        dot.set_data([sim.x], [sim.y])
                        ax.relim()
                        ax.autoscale_view()
                        pid_text.set_text(
                            f"Kp={angle_pid.Kp:.2f}, Ki={angle_pid.Ki:.2f}, Kd={angle_pid.Kd:.2f}\nMode: {current_mode}"
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

                all_paths.append(sim.path.copy())
                angle_pid.reset()
                distance_pid.reset()
                last_pos = None
                last_color = None

            if not loop:
                break

    except KeyboardInterrupt:
        print("🛑 Simulation stopped by user.")

        if visualize or save_video:
            try:
                os.makedirs("results", exist_ok=True)
                df = pd.DataFrame(results)
                df.to_excel(f"results/final_results_Kp{angle_pid.Kp}_Ki{angle_pid.Ki}_Kd{angle_pid.Kd}.xlsx", index=False)
                print("✅ Results saved to Excel.")
            except Exception as e:
                print(f"❌ Failed to save Excel: {e}")

            if save_video:
                try:
                    save_combined_video(all_paths, f"results/sim_combined_Kp{angle_pid.Kp}_Ki{angle_pid.Ki}_Kd{angle_pid.Kd}.mp4")
                except Exception as e:
                    print(f"❌ Failed to save video: {e}")

        if visualize:
            plt.ioff()
            plt.show()

    return results

def save_combined_video(paths, filename_mp4):
    flat_path = []
    for path in paths:
        flat_path.extend(path)
        flat_path.append((None, None))

    numeric_path = [pt for pt in flat_path if isinstance(pt[0], (int, float)) and isinstance(pt[1], (int, float))]
    if not numeric_path:
        print("⚠️ No valid data to plot, skipping video.")
        return

    all_x = [x for x, _ in numeric_path]
    all_y = [y for _, y in numeric_path]

    fig, ax = plt.subplots()
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    line, = ax.plot([], [], 'b-', lw=2)
    dot, = ax.plot([], [], 'ro')

    def update(i):
        current = flat_path[i]
        if current == (None, None):
            return line, dot
        visible_path = [p for p in flat_path[:i+1] if isinstance(p[0], (int, float))]
        x, y = current
        line.set_data(*zip(*visible_path))
        dot.set_data([x], [y])
        return line, dot

    ani = FuncAnimation(fig, update, frames=len(flat_path), interval=50, blit=True)

    try:
        ani.save(filename_mp4, fps=20)
        print(f"✅ Video saved: {filename_mp4}")
    except Exception as e:
        filename_gif = filename_mp4.replace('.mp4', '.gif')
        print(f"⚠️ Failed to save MP4: {e}\n➡️ Saving GIF instead.")
        ani.save(filename_gif, fps=20)
        print(f"✅ GIF saved: {filename_gif}")

    plt.close(fig)
