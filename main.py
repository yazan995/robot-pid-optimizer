from multi_goal_runner import simulate_pid_on_multiple_goals
import numpy as np

goals = [
    [5, 5],
    [-5, 5],
    [5, -5],
    [-5, -5],
    [8, 0]
]

# قائمة بالقيم التجريبية للـ PID
pid_candidates = [
    (1.0, 0.0, 0.0),
    (1.0, 0.05, 0.0),
    (1.0, 0.0, 0.1),
    (1.0, 0.0, 0.2),
    (1.0, 0.0, 0.15),
]

best_result = None
best_pid = None

print(" Starting classical PID experiments...\n")
for Kp, Ki, Kd in pid_candidates:
    print(f" Testing PID: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    results = simulate_pid_on_multiple_goals(Kp, Ki, Kd, goals, visualize=False, save_video=False)

    total_error = sum([res["Final Error"] for res in results])
    if best_result is None or total_error < best_result:
        best_result = total_error
        best_pid = (Kp, Ki, Kd)

# بعد اختيار الأفضل، نفذ المحاكاة النهائية بالفيديو
print(f"\n The best PID is: Kp={best_pid[0]}, Ki={best_pid[1]}, Kd={best_pid[2]}")
simulate_pid_on_multiple_goals(*best_pid, goals, visualize=False, save_video=True)
