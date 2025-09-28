import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Initialize global variables
headers = []
data = None

# GUI setup
root = tk.Tk()
root.title("2D/3D Data Plotter")
start_index = tk.IntVar(value=0)
end_index = tk.IntVar(value=100)  # Default, will be updated after loading data
x_var = tk.StringVar()
y_var = tk.StringVar()
z_var = tk.StringVar()
plot_mode = tk.StringVar(value="2D")

normalize_x = tk.BooleanVar(value=False)
normalize_y = tk.BooleanVar(value=False)
normalize_z = tk.BooleanVar(value=False)

x_menu = ttk.OptionMenu(root, x_var, "")
y_menu = ttk.OptionMenu(root, y_var, "")
z_menu = ttk.OptionMenu(root, z_var, "")

def update_dropdowns():
    x_menu['menu'].delete(0, 'end')
    y_menu['menu'].delete(0, 'end')
    z_menu['menu'].delete(0, 'end')
    for h in headers:
        x_menu['menu'].add_command(label=h, command=tk._setit(x_var, h))
        y_menu['menu'].add_command(label=h, command=tk._setit(y_var, h))
        z_menu['menu'].add_command(label=h, command=tk._setit(z_var, h))
    x_var.set(headers[0])
    y_var.set(headers[1])
    z_var.set(headers[2])

def load_file():
    global headers, data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")])
    if not file_path:
        return
    with open(file_path, 'r') as f:
        headers = f.readline().strip().split(',')
        data = np.loadtxt(f, delimiter=',')
    update_dropdowns()
    end_index.set(len(data) - 1)  # Set max index after loading


def plot_data():
    if data is None:
        return

    start = start_index.get()
    end = end_index.get()
    if start < 0 or end >= len(data) or start >= end:
        print("Invalid index range")
        return

    x_idx = headers.index(x_var.get())
    y_idx = headers.index(y_var.get())
    z_idx = headers.index(z_var.get())

    x = data[start:end+1, x_idx]
    y = data[start:end+1, y_idx]
    z = data[start:end+1, z_idx]

    if normalize_x.get():
        x = x - x[0]
    if normalize_y.get():
        y = y - y[0]
    if normalize_z.get() and plot_mode.get() == "3D":
        z = z - z[0]

    fig = plt.figure()
    if plot_mode.get() == "3D":
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, c='blue', marker='o')
        ax.set_zlabel(z_var.get())
    else:
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'bo-')

    ax.set_xlabel(x_var.get())
    ax.set_ylabel(y_var.get())
    plt.title(f"{plot_mode.get()} Plot of {x_var.get()} vs {y_var.get()}" + (f" vs {z_var.get()}" if plot_mode.get() == "3D" else ""))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Layout
ttk.Button(root, text="Browse File", command=load_file).grid(row=0, column=0, columnspan=2, pady=5)

ttk.Label(root, text="X Axis").grid(row=1, column=0)
x_menu.grid(row=1, column=1)
ttk.Checkbutton(root, text="Normalize X", variable=normalize_x).grid(row=1, column=2)

ttk.Label(root, text="Y Axis").grid(row=2, column=0)
y_menu.grid(row=2, column=1)
ttk.Checkbutton(root, text="Normalize Y", variable=normalize_y).grid(row=2, column=2)

ttk.Label(root, text="Z Axis (for 3D)").grid(row=3, column=0)
z_menu.grid(row=3, column=1)
ttk.Checkbutton(root, text="Normalize Z", variable=normalize_z).grid(row=3, column=2)

ttk.Label(root, text="Plot Type").grid(row=4, column=0)
ttk.OptionMenu(root, plot_mode, "2D", "2D", "3D").grid(row=4, column=1)

ttk.Button(root, text="Plot", command=plot_data).grid(row=5, column=0, columnspan=3, pady=10)

ttk.Label(root, text="Start Index").grid(row=6, column=0)
ttk.Spinbox(root, from_=0, to=100, textvariable=start_index, width=10).grid(row=6, column=1)

ttk.Label(root, text="End Index").grid(row=7, column=0)
ttk.Spinbox(root, from_=0, to=100, textvariable=end_index, width=10).grid(row=7, column=1)

root.mainloop()