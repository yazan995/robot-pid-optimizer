from optimizer.genetic__optimizer import genetic_optimization
from multi_goal_runner import simulate_pid_on_multiple_goals
import numpy as np

if __name__ == "__main__":
    # Run genetic optimization WITHOUT visualization
    best_pid = genetic_optimization(
        population_size=10,
        generations=15,
        mutation_rate=0.4,
        crossover_rate=0.7
    )

    # Print best result
    Kp, Ki, Kd = best_pid
    print(f"\n✅ Best PID found: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print("🎥 Starting live simulation...")

    # Define goals
    goals = [
        np.array([5.0, 5.0]),
        np.array([-5.0, 5.0]),
        np.array([5.0, -5.0]),
        np.array([-5.0, -5.0]),
        np.array([8.0, 0.0]),
    ]

    # Run one interactive simulation using best PID
    simulate_pid_on_multiple_goals(
        Kp=Kp, Ki=Ki, Kd=Kd,
        goals=goals,
        visualize=True,
        save_video=False,
        loop=True  # ✅ enable infinite loop for interactive display
    )

