# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 09:53:30 2025

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""
# Default libraries
import tkinter as tk
from tkinter import ttk
import threading
import time

# Install libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Explicit libraries


class RealTimePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Distribution Visualizer")

        # State variables
        self.mean = tk.DoubleVar(value=0)
        self.variance = tk.DoubleVar(value=1)
        self.particles = tk.IntVar(value=15)
        self.particle_freq = tk.DoubleVar(value=1.0)
        self.graphic_freq = tk.DoubleVar(value=1.0)
        self.hist_data = np.array([])
        self.running = True

        self.create_widgets()
        self.start_threads()

    def create_widgets(self):
        # Input fields
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=10)

        ttk.Label(input_frame, text="Mean:").grid(row=0, column=0)
        ttk.Entry(input_frame, textvariable=self.mean, width=10).grid(row=0, column=1)

        ttk.Label(input_frame, text="Variance:").grid(row=0, column=2)
        ttk.Entry(input_frame, textvariable=self.variance, width=10).grid(row=0, column=3)

        ttk.Label(input_frame, text="Particles:").grid(row=1, column=0)
        ttk.Entry(input_frame, textvariable=self.particles, width=10).grid(row=1, column=1)

        ttk.Label(input_frame, text="Particle Freq (s):").grid(row=1, column=2)
        ttk.Entry(input_frame, textvariable=self.particle_freq, width=10).grid(row=1, column=3)

        ttk.Label(input_frame, text="Graphic Freq (s):").grid(row=2, column=0)
        ttk.Scale(input_frame, from_=0.2, to=2.0, variable=self.graphic_freq, orient='horizontal', length=200).grid(row=2, column=1, columnspan=3)

        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    def simulate_particles(self):
        while self.running:
            try:
                mean = self.mean.get()
                var = max(self.variance.get(), 0.01)
                count = self.particles.get()
                freq = self.particle_freq.get()
                new_particles = np.random.normal(mean, np.sqrt(var), count)
                self.hist_data = np.concatenate([self.hist_data, new_particles])
                time.sleep(freq)
            except Exception as e:
                print("Particle simulation error:", e)

    def update_plot(self):
        while self.running:
            try:
                mean = self.mean.get()
                var = max(self.variance.get(), 0.01)

                x = np.linspace(mean - 4*np.sqrt(var), mean + 4*np.sqrt(var), 400)
                y = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean)**2 / var))

                self.ax1.clear()
                self.ax1.plot(x, y)
                self.ax1.set_xlim(left=-25, right=25)
                self.ax1.set_title("Normal Distribution")

                self.ax2.clear()
                self.ax2.hist(self.hist_data, bins=30, color='orange', alpha=0.7)
                self.ax2.set_title("Accumulated Particles")

                self.canvas.draw()
                time.sleep(self.graphic_freq.get())
            except Exception as e:
                print("Plot update error:", e)

    def start_threads(self):
        threading.Thread(target=self.simulate_particles, daemon=True).start()
        threading.Thread(target=self.update_plot, daemon=True).start()

    def on_close(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimePlotApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
