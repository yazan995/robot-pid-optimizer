# Adaptive PID Control for Mobile Robot Simulation using Genetic Algorithms

This project implements a simulation system for a mobile robot that combines a classical PID controller with Genetic Algorithms (GA) for automatic and adaptive parameter optimization.  
The system allows interactive testing in different environments: **Normal**, **Heavy Load**, and **Slippery**.

---

📂 Project Structure

- **pid.py** → Implements the PID controller logic.  
- **simulation.py** → Defines the robot simulator in a 2D environment.  
- **environment_gui.py** → Graphical interface (Tkinter) to switch between environment modes.  
- **genetic__optimizer.py** → Runs the Genetic Algorithm to optimize PID parameters.  
- **multi_goal_runner.py** → Executes the simulation with multiple goals and dynamic environment changes.  
- **test_optimizer.py** → **Main entry point**. Runs the GA, extracts best PID values, and starts the simulation.  

---

▶ How to Run (Windows, Python 3.12)

```bash
# 1) Clone and enter the project
git clone https://github.com/yazan995/robot-pid-optimizer-master.git
cd robot-pid-optimizer-master

# 2) Create a virtual environment and install dependencies
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python -m pip install -r requirements.txt

# 3) Run the main entry point
.venv\Scripts\python test_optimizer.py
```

During simulation:
- Use the GUI to switch between environments: Normal / Heavy Load / Slippery.

---
📊 Outputs

Excel files with best PID parameters and fitness evolution.

Plots showing performance and fitness over generations.

Videos (MP4/GIF) visualizing the robot’s path under different environments.

---
⚙️ Requirements

Python 3.9+ (tested on Python 3.12, Windows 10/11)

Required libraries:

numpy

matplotlib

tkinter (comes with Python standard library; on Linux may require separate installation)

pandas

openpyxl

Install all at once:
pip install numpy matplotlib pandas openpyxl

---
📌 Notes

Make sure to start the project by running test_optimizer.py.

All results (Excel, plots, videos) will be saved in the project folder.

For detailed explanation, please refer to the report (Dok Final).

---
👨‍💻 Authors:

Yazan Hasan

Milad Shafiei Baghbaderani
