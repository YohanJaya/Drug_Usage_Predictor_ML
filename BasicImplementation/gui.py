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
from functools import partial

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

    def setup_ui_styles(self):
        # Define colors and fonts
        self.primary_color = "#2b6777"
        self.accent_color = "#52ab98"
        self.button_color = "#388087"
        self.bg_color = "#f0f4f8"
        self.card_bg = "#e7eff6"
        self.header_font = ("Segoe UI", 26, "bold")
        self.section_font = ("Segoe UI", 14, "bold")
        self.normal_font = ("Segoe UI", 12)

        # Simple tooltip helper
        class ToolTip:
            def __init__(self, widget, text):
                self.widget = widget
                self.text = text
                self.tip = None
                widget.bind("<Enter>", self.show)
                widget.bind("<Leave>", self.hide)
            def show(self, _=None):
                if self.tip or not self.text:
                    return
                x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0,0,0,0)
                x = self.widget.winfo_rootx() + 20
                y = self.widget.winfo_rooty() + 20
                self.tip = tk.Toplevel(self.widget)
                self.tip.wm_overrideredirect(True)
                self.tip.wm_geometry(f"+{x}+{y}")
                lbl = tk.Label(self.tip, text=self.text, bg="#333", fg="white", bd=1, font=("Segoe UI", 9), padx=6, pady=3)
                lbl.pack()
            def hide(self, _=None):
                if self.tip:
                    self.tip.destroy()
                    self.tip = None
        self.ToolTip = ToolTip

    def _apply_button_hover(self, btn, normal, hover):
        def on_enter(e):
            try:
                btn.config(bg=hover)
            except Exception:
                pass
        def on_leave(e):
            try:
                btn.config(bg=normal)
            except Exception:
                pass
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def create_widgets(self):
        # apply UI styles
        self.setup_ui_styles()
        # Central content frame to center all UI elements
        self.content_frame = tk.Frame(self.scrollable_frame, bg=self.bg_color)
        self.content_frame.pack(expand=True, fill='both', padx=40, pady=10)

        self.title_label = tk.Label(self.content_frame, text="Drug Usage Predictor", font=("Segoe UI", 28, "bold"), fg=self.primary_color, bg=self.bg_color)
        self.title_label.pack(pady=20)
        # start title animation (color pulse)
        self._title_anim_index = 0
        self.animate_title()

        # Data source selection
        mode_frame = tk.Frame(self.content_frame, bg=self.bg_color)
        mode_frame.pack(pady=6)
        tk.Label(mode_frame, text="Data Source:", font=("Segoe UI", 12, "bold"), bg=self.bg_color, fg=self.primary_color).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Synthetic", variable=self.data_mode, value="synthetic", font=("Segoe UI", 12), bg=self.bg_color).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Kaggle CSV", variable=self.data_mode, value="kaggle", font=("Segoe UI", 12), bg=self.bg_color).pack(side=tk.LEFT)
        self.kaggle_path_entry = tk.Entry(mode_frame, width=40, font=("Segoe UI", 12))
        self.kaggle_path_entry.pack(side=tk.LEFT)
        self.kaggle_path_entry.insert(0, "data/Hyper.csv")

        self.run_button = tk.Button(self.content_frame, text="Run Pipeline", command=self.run_pipeline_with_animation, font=("Segoe UI", 14, "bold"), bg=self.accent_color, fg="white", activebackground=self.button_color, activeforeground="white")
        self.run_button.pack(pady=15)
        self._apply_button_hover(self.run_button, self.accent_color, self.button_color)
        self.ToolTip(self.run_button, "Run full pipeline: load data, train model, and show results")

        # Progress bar animation
        self.progress_frame = tk.Frame(self.content_frame, bg=self.bg_color)
        self.progress_frame.pack(pady=6)
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack()
        self.progress_label = tk.Label(self.progress_frame, text="", font=("Segoe UI", 10), bg=self.bg_color, fg=self.button_color)
        self.progress_label.pack()

        self.metrics_frame = tk.LabelFrame(self.content_frame, text="Evaluation Metrics", font=self.section_font, fg=self.primary_color, bg=self.card_bg, bd=0, relief=tk.FLAT)
        self.metrics_frame.pack(padx=20, pady=10, fill='x')
        self.metrics_text = tk.Text(self.metrics_frame, height=6, width=80, font=self.normal_font, bg=self.card_bg, fg="#22223b")
        self.metrics_text.pack()

        self.predictions_frame = tk.LabelFrame(self.content_frame, text="Sample Predictions", font=self.section_font, fg=self.primary_color, bg=self.card_bg, bd=0, relief=tk.FLAT)
        self.predictions_frame.pack(padx=20, pady=10, fill='x')
        self.predictions_text = tk.Text(self.predictions_frame, height=6, width=80, font=self.normal_font, bg=self.card_bg, fg="#22223b")
        self.predictions_text.pack()

        self.images_frame = tk.LabelFrame(self.content_frame, text="Visualizations", font=self.section_font, fg=self.primary_color, bg=self.card_bg, bd=0, relief=tk.FLAT)
        self.images_frame.pack(padx=20, pady=10)
        self.image_labels = []
        for i in range(4):
            lbl = tk.Label(self.images_frame, bg=self.card_bg, text="")
            lbl.grid(row=0, column=i, padx=5, pady=5)
            self.image_labels.append(lbl)

        # Prediction input part
        # Use a dedicated container for input fields so we can clear/rebuild them safely
        self.input_frame = tk.LabelFrame(self.content_frame, text="Predict Drug Usage", font=self.section_font, fg=self.primary_color, bg=self.card_bg, bd=0, relief=tk.FLAT)
        self.input_frame.pack(padx=20, pady=10, fill='x')
        # container for the dynamic input field widgets
        self.fields_container = tk.Frame(self.input_frame, bg=self.card_bg)
        self.fields_container.pack(side=tk.LEFT, fill="x", expand=True)
        self.input_entries = []
        self.input_labels = []
        # Create labeled input fields using helper to ensure consistent layout
        self.build_input_fields(self.feature_names)

        # Right-side frame for Predict button + animated result
        right_side = tk.Frame(self.input_frame, bg=self.card_bg)
        right_side.pack(side=tk.RIGHT, padx=10, pady=10)
        self.predict_button = tk.Button(right_side, text="Predict", command=self.on_predict, font=("Segoe UI", 13, "bold"), bg=self.button_color, fg="white", activebackground=self.accent_color, activeforeground="white", bd=0, padx=12, pady=6, cursor='hand2')
        self.predict_button.pack(pady=(0,6))
        self._apply_button_hover(self.predict_button, self.button_color, self.accent_color)
        self.ToolTip(self.predict_button, "Make a prediction from the input values")

        # Prediction result label (starts hidden)
        self.prediction_result = tk.Label(right_side, text="", font=("Segoe UI", 12, "bold"), bg=self.card_bg, fg=self.bg_color)
        self.prediction_result.pack()

    def build_input_fields(self, feature_names):
        # Clear any existing widgets in the fields container
        container = getattr(self, 'fields_container', None)
        if container is None:
            # fallback to input_frame or scrollable_frame which are valid widget masters
            container = getattr(self, 'input_frame', getattr(self, 'scrollable_frame', None))
        if container is None:
            raise RuntimeError("No container available to build input fields")

        for child in container.winfo_children():
            child.destroy()

        self.input_entries = []
        self.input_labels = []
        for fname in feature_names:
            frame = tk.Frame(container, bg=self.card_bg, highlightbackground="#52ab98", highlightcolor="#52ab98", highlightthickness=2, bd=1, relief=tk.SOLID)
            frame.pack(side=tk.LEFT, padx=8, pady=8)
            label = tk.Label(frame, text=fname+":", font=("Segoe UI", 13, "bold"), bg=self.card_bg, fg="#22223b")
            label.pack(pady=4)
            entry = tk.Entry(frame, width=18, font=("Segoe UI", 13), bg="#ffffff", relief=tk.FLAT, highlightbackground="#388087", highlightcolor="#388087", highlightthickness=1)
            entry.pack(pady=4)
            entry.config(state="normal")
            # bind focus animations
            entry.bind("<FocusIn>", lambda e, en=entry: self._on_entry_focus_in(en))
            entry.bind("<FocusOut>", lambda e, en=entry: self._on_entry_focus_out(en))
            self.input_entries.append(entry)
            self.input_labels.append(label)

    def on_predict(self):
        try:
            if self.w is None or self.mean is None or self.std is None:
                messagebox.showwarning("Warning", "Please run the pipeline first.")
                return
            # Validate and collect inputs
            values = []
            for e in self.input_entries[:len(self.feature_names)]:
                v = e.get().strip()
                if v == "":
                    messagebox.showwarning("Warning", "Please fill all input fields before predicting.")
                    return
                values.append(float(v))
            x = np.array(values, dtype=float)
            # Normalize using stored mean/std. Handle possible shape mismatch by slicing mean/std
            if hasattr(self.mean, 'shape') and self.mean.shape == x.shape:
                x_norm = (x - self.mean) / self.std
            else:
                x_norm = (x - self.mean[:x.shape[0]]) / self.std[:x.shape[0]]
            y_pred = predict(x_norm.reshape(1, -1), self.w, self.b)
            text = f"Predicted Usage: {y_pred[0]:.2f}"
            # Use animated reveal so text becomes visible (prediction_result initially uses bg-colored fg)
            try:
                self.animate_prediction_result(text)
            except Exception:
                # Fallback: directly set visible color
                self.prediction_result.config(text=text, fg=self.primary_color)
            # Clear input fields shortly after showing result so user can read it
            self.root.after(900, lambda: [entry.delete(0, tk.END) for entry in self.input_entries])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_pipeline_with_animation(self):
        # Removed self.input_frame.pack_forget() since input frame is gone
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
        # Removed self.input_frame.pack(fill="x", padx=20, pady=10) since input frame is gone

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

    # --- Animation / UX helpers ---
    def animate_title(self):
        # simple color pulse between primary and accent
        try:
            colors = [self.primary_color, self.accent_color]
            color = colors[self._title_anim_index % len(colors)]
            self.title_label.config(fg=color)
            self._title_anim_index += 1
            self.root.after(700, self.animate_title)
        except Exception:
            pass

    def reveal_images_with_animation(self):
        # stagger loading of images to make UI feel alive
        fig_dir = 'reports/figures'
        img_files = ['cost_history.png', 'predictions.png', 'residuals.png', 'feature_importance.png']
        for i, lbl in enumerate(self.image_labels):
            def _reveal(idx=i, label=lbl):
                img_path = os.path.join(fig_dir, img_files[idx])
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).resize((180, 180))
                        img_tk = ImageTk.PhotoImage(img)
                        label.configure(image=img_tk, text="")
                        label.image = img_tk
                    except Exception:
                        label.configure(text="Image Err")
                else:
                    label.configure(text="No Image")
            self.root.after(250 * i, _reveal)

    def animate_prediction_result(self, text):
        # fade-in-like effect by changing fg from bg to primary
        try:
            steps = 6
            r1, g1, b1 = self._hex_to_rgb(self.bg_color)
            r2, g2, b2 = self._hex_to_rgb(self.primary_color)
            self.prediction_result.config(text=text)
            for i in range(steps):
                def _step(i=i):
                    t = (i + 1) / steps
                    nr = int(r1 + (r2 - r1) * t)
                    ng = int(g1 + (g2 - g1) * t)
                    nb = int(b1 + (b2 - b1) * t)
                    self.prediction_result.config(fg=f"#{nr:02x}{ng:02x}{nb:02x}")
                self.root.after(60 * i, _step)
        except Exception:
            try:
                self.prediction_result.config(fg=self.primary_color, text=text)
            except Exception:
                pass

    def _hex_to_rgb(self, hexcol):
        hexcol = hexcol.lstrip('#')
        return tuple(int(hexcol[i:i+2], 16) for i in (0, 2, 4))

    def _on_entry_focus_in(self, entry):
        try:
            entry.config(highlightthickness=2, highlightbackground=self.accent_color)
        except Exception:
            pass

    def _on_entry_focus_out(self, entry):
        try:
            entry.config(highlightthickness=1, highlightbackground="#388087")
        except Exception:
            pass

def main():
    root = tk.Tk()
    app = DrugUsagePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
