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

▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yazan995/robot-pid-ga-simulation.git
   cd robot-pid-ga-simulation
2. Install required Python packages:
pip install -r requirements.txt
3. Run the main script:
python test_optimizer.py
4. During simulation:
Use the GUI to switch between environments (Normal / Heavy Load / Slippery).
The system will automatically re-optimize PID parameters using the Genetic Algorithm.

---
📊 Outputs

Excel files with best PID parameters and fitness evolution.

Plots showing performance and fitness over generations.

Videos (MP4/GIF) visualizing the robot’s path under different environments.

---
⚙️ Requirements

Python 3.9+

Required libraries:

numpy

matplotlib

tkinter (comes with Python standard library)

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
