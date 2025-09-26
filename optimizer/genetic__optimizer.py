import random
import numpy as np
from simulator.simulation import RobotSimulator
from controller.pid import PIDController
import pandas as pd
import os
import matplotlib.pyplot as plt

def evaluate_pid(Kp, Ki, Kd):
    goals = [
        np.array([5.0, 5.0]),
        np.array([-5.0, 5.0]),
        np.array([5.0, -5.0]),
        np.array([-5.0, -5.0]),
        np.array([8.0, 0.0]),
    ]
    
    sim = RobotSimulator(dt=0.1)
    angle_pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd)
    distance_pid = PIDController(Kp=1.0, Ki=0.0, Kd=0.05)

    total_time = 0.0
    total_error = 0.0

    for goal in goals:
        max_steps = 300
        for _ in range(max_steps):
            x, y, theta = sim.x, sim.y, sim.theta
            direction = np.arctan2(goal[1] - y, goal[0] - x)
            angle_error = (direction - theta + np.pi) % (2 * np.pi) - np.pi

            omega = angle_pid.compute(angle_error, sim.dt)
            distance = np.linalg.norm(goal - np.array([x, y]))
            v = max(0.0, min(distance_pid.compute(distance, sim.dt), 2.0))

            sim.step(v=v, omega=omega)
            total_time += sim.dt

            if distance < 0.1:
                break

        final_error = np.linalg.norm(goal - np.array([sim.x, sim.y]))
        total_error += final_error

    avg_error = total_error / len(goals)
    fitness = total_time + 10 * avg_error
    return fitness, total_time, avg_error


def genetic_optimization(
    population_size=6,
    generations=5,
    mutation_rate=0.4,
    crossover_rate=0.7,
    mode="normal"
):
    best_fitness_history = []
    results = []

    population = []
    for _ in range(population_size):
        Kp = round(random.uniform(0.5, 5.0), 2)
        Ki = round(random.uniform(0.0, 1.0), 2)
        Kd = round(random.uniform(0.0, 1.0), 2)
        population.append((Kp, Ki, Kd))

    for gen in range(generations):
        print(f"\n Generation {gen + 1} for mode: {mode}")
        scored = []
        for Kp, Ki, Kd in population:
            fitness, t, error = evaluate_pid(Kp, Ki, Kd)
            results.append((gen + 1, Kp, Ki, Kd, fitness))
            scored.append(((Kp, Ki, Kd), fitness))
            print(f"  PID({Kp}, {Ki}, {Kd}) -> Fitness: {fitness:.2f}, Time: {t:.2f}, Error: {error:.4f}")

        scored.sort(key=lambda x: x[1])
        best = scored[0][0]
        best_fitness = scored[0][1]
        best_fitness_history.append(best_fitness)

        selected = [k for k, _ in scored[:population_size // 2]]
        new_population = selected.copy()

        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                p1, p2 = random.sample(selected, 2)
                child = (
                    round((p1[0] + p2[0]) / 2, 2),
                    round((p1[1] + p2[1]) / 2, 2),
                    round((p1[2] + p2[2]) / 2, 2),
                )
            else:
                child = random.choice(selected)

            if random.random() < mutation_rate:
                idx = random.randint(0, 2)
                delta = random.uniform(-1.0, 1.0)
                child = list(child)
                child[idx] = round(max(0.0, child[idx] + delta), 2)
                child = tuple(child)

            new_population.append(child)

        population = new_population
        save_results_to_excel(results, best, mode)
        plot_fitness(best_fitness_history, mode)

    final_fitness, time, err = evaluate_pid(*best)
    print(f"\n Best PID for mode '{mode}': Kp={best[0]}, Ki={best[1]}, Kd={best[2]}")
    return best

def plot_fitness(best_fitness_history, mode="default", output_dir="results"):
    plt.figure(figsize=(8, 5))
    plt.plot(best_fitness_history, marker='o', color='blue')
    for i, val in enumerate(best_fitness_history):
        plt.text(i, val, f"{val:.3f}", ha='center', va='bottom', fontsize=9)
    plt.title(f'Best Fitness Over Generations ({mode})')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"best_fitness_plot_{mode}.png"))
    plt.close()

def save_results_to_excel(results, best_pid, mode="default"):
    df = pd.DataFrame(results, columns=['Generation', 'Kp', 'Ki', 'Kd', 'Fitness'])
    best_row = pd.DataFrame([["BEST", best_pid[0], best_pid[1], best_pid[2], evaluate_pid(*best_pid)[0]]],
                            columns=['Generation', 'Kp', 'Ki', 'Kd', 'Fitness'])
    df = pd.concat([df, best_row], ignore_index=True)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    df.to_excel(os.path.join(output_dir, f"genetic_results_{mode}.xlsx"), index=False)
