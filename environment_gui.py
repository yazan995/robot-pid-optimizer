import tkinter as tk

root = tk.Tk()
root.title("Environment Mode Selector")
root.geometry("300x200")

current_mode = tk.StringVar(value="normal")

def set_mode(mode):
    current_mode.set(mode)
    print(f"✅ Environment mode changed to: {mode}")

def get_mode():
    return current_mode.get()  # ✅ نوفر دالة للوصول من الخارج

label = tk.Label(root, text="Select Environment Mode:", font=("Arial", 14))
label.pack(pady=10)

btn_normal = tk.Button(root, text="Normal", width=20, command=lambda: set_mode("normal"))
btn_normal.pack(pady=5)

btn_heavy = tk.Button(root, text="Heavy Load", width=20, command=lambda: set_mode("heavy"))
btn_heavy.pack(pady=5)

btn_slippery = tk.Button(root, text="Slippery", width=20, command=lambda: set_mode("slippery"))
btn_slippery.pack(pady=5)

if __name__ == "__main__":
    root.mainloop()
