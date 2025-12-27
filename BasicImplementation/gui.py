import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import os
import numpy as np
import pandas as pd
from src.data import generate_synthetic_data, normalize_features, split_data
from src.models import train_model, predict, evaluate_model
from src.utils import plot_cost_history, plot_predictions, plot_residuals, plot_feature_importance
from src.data.kaggle_loader import load_kaggle_data
import threading
import time

class DrugUsagePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drug Usage Predictor - Linear Regression")
        self.root.geometry("1100x900")
        self.root.configure(bg="#f0f4f8")
        self.data_mode = tk.StringVar(value="synthetic")
        self.feature_names = ['Patient Count', 'Emergency Cases', 'Is Holiday', 'Previous Day Usage']
        # Add scrollable canvas
        self.canvas = tk.Canvas(self.root, bg="#f0f4f8", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f0f4f8")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.create_widgets()
        self.pipeline_run = False
        self.results = {}
        self.w = None
        self.b = None
        self.mean = None
        self.std = None

    def create_widgets(self):
        self.title_label = tk.Label(self.scrollable_frame, text="Drug Usage Predictor", font=("Segoe UI", 28, "bold"), fg="#2b6777", bg="#f0f4f8")
        self.title_label.pack(pady=20)

        # Data source selection
        mode_frame = tk.Frame(self.scrollable_frame, bg="#f0f4f8")
        mode_frame.pack(pady=5)
        tk.Label(mode_frame, text="Data Source:", font=("Segoe UI", 12, "bold"), bg="#f0f4f8", fg="#2b6777").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Synthetic", variable=self.data_mode, value="synthetic", font=("Segoe UI", 12), bg="#f0f4f8").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Kaggle CSV", variable=self.data_mode, value="kaggle", font=("Segoe UI", 12), bg="#f0f4f8").pack(side=tk.LEFT)
        self.kaggle_path_entry = tk.Entry(mode_frame, width=40, font=("Segoe UI", 12))
        self.kaggle_path_entry.pack(side=tk.LEFT)
        self.kaggle_path_entry.insert(0, "data/Hyper.csv")

        self.run_button = tk.Button(self.scrollable_frame, text="Run Pipeline", command=self.run_pipeline_with_animation, font=("Segoe UI", 14, "bold"), bg="#52ab98", fg="white", activebackground="#388087", activeforeground="white")
        self.run_button.pack(pady=15)

        # Progress bar animation
        self.progress_frame = tk.Frame(self.scrollable_frame, bg="#f0f4f8")
        self.progress_frame.pack(pady=5)
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack()
        self.progress_label = tk.Label(self.progress_frame, text="", font=("Segoe UI", 10), bg="#f0f4f8", fg="#388087")
        self.progress_label.pack()

        self.metrics_frame = tk.LabelFrame(self.scrollable_frame, text="Evaluation Metrics", font=("Segoe UI", 14, "bold"), fg="#2b6777", bg="#e7eff6", bd=2, relief=tk.GROOVE)
        self.metrics_frame.pack(fill="x", padx=20, pady=10)
        self.metrics_text = tk.Text(self.metrics_frame, height=6, width=80, font=("Segoe UI", 11), bg="#e7eff6", fg="#22223b")
        self.metrics_text.pack()

        self.predictions_frame = tk.LabelFrame(self.scrollable_frame, text="Sample Predictions", font=("Segoe UI", 14, "bold"), fg="#2b6777", bg="#e7eff6", bd=2, relief=tk.GROOVE)
        self.predictions_frame.pack(fill="x", padx=20, pady=10)
        self.predictions_text = tk.Text(self.predictions_frame, height=6, width=80, font=("Segoe UI", 11), bg="#e7eff6", fg="#22223b")
        self.predictions_text.pack()

        self.images_frame = tk.LabelFrame(self.scrollable_frame, text="Visualizations", font=("Segoe UI", 14, "bold"), fg="#2b6777", bg="#e7eff6", bd=2, relief=tk.GROOVE)
        self.images_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.image_labels = []
        for i in range(4):
            lbl = tk.Label(self.images_frame, bg="#e7eff6")
            lbl.grid(row=0, column=i, padx=5, pady=5)
            self.image_labels.append(lbl)

    def run_pipeline_with_animation(self):
        self.input_frame.pack_forget()  # Hide prediction part while running pipeline
        self.progress_label.config(text="Running pipeline...")
        self.progress['value'] = 0
        self.root.update_idletasks()
        threading.Thread(target=self._run_pipeline_animated).start()

    def _run_pipeline_animated(self):
        for i in range(1, 101, 5):
            self.progress['value'] = i
            self.root.update_idletasks()
            time.sleep(0.03)
        self.run_pipeline()
        self.progress['value'] = 100
        self.progress_label.config(text="Pipeline complete!")
        self.root.update_idletasks()
        time.sleep(0.5)
        self.progress['value'] = 0
        self.progress_label.config(text="")
        self.input_frame.pack(fill="x", padx=20, pady=10)  # Show prediction part again

    def run_pipeline(self):
        try:
            mode = self.data_mode.get()
            if mode == "synthetic":
                X, y = generate_synthetic_data(days=365, random_state=42)
                self.feature_names = ['Patient Count', 'Emergency Cases', 'Is Holiday', 'Previous Day Usage']
            else:
                csv_path = self.kaggle_path_entry.get()
                X, y, self.feature_names = load_kaggle_data(csv_path)
            X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8, random_state=42)
            X_train_norm, mean, std = normalize_features(X_train)
            X_test_norm = (X_test - mean) / std
            w, b, J_history = train_model(X_train_norm, y_train, alpha=0.01, num_iters=10000)
            metrics = evaluate_model(X_test_norm, y_test, w, b)
            y_pred_test = predict(X_test_norm[:5], w, b)
            y_pred_full = predict(X_test_norm, w, b)
            fig_dir = 'reports/figures'
            os.makedirs(fig_dir, exist_ok=True)
            plot_cost_history(J_history, save_path=os.path.join(fig_dir, 'cost_history.png'))
            plot_predictions(y_test, y_pred_full, save_path=os.path.join(fig_dir, 'predictions.png'))
            plot_residuals(y_test, y_pred_full, save_path=os.path.join(fig_dir, 'residuals.png'))
            plot_feature_importance(w, feature_names=self.feature_names, save_path=os.path.join(fig_dir, 'feature_importance.png'))
            self.metrics_text.delete(1.0, tk.END)
            for k, v in metrics.items():
                self.metrics_text.insert(tk.END, f"{k}: {v:.2f}\n")
            self.predictions_text.delete(1.0, tk.END)
            for i in range(5):
                self.predictions_text.insert(tk.END, f"Actual: {y_test[i]:.0f}, Predicted: {y_pred_test[i]:.0f}\n")
            img_files = ['cost_history.png', 'predictions.png', 'residuals.png', 'feature_importance.png']
            for i, img_name in enumerate(img_files):
                img_path = os.path.join(fig_dir, img_name)
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img = img.resize((180, 180))
                    img_tk = ImageTk.PhotoImage(img)
                    self.image_labels[i].configure(image=img_tk)
                    self.image_labels[i].image = img_tk
                else:
                    self.image_labels[i].configure(image=None, text="No Image")
            # Save model params for prediction
            self.w = w
            self.b = b
            self.mean = mean
            self.std = std
            # Dynamically rebuild input fields for new features
            self.build_input_fields(self.feature_names)
            messagebox.showinfo("Pipeline Complete", "Pipeline completed successfully! Plots saved to reports/figures/")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = DrugUsagePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
