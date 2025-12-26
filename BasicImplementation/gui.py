import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import os
import numpy as np
from src.data import generate_synthetic_data, normalize_features, split_data
from src.models import train_model, predict, evaluate_model
from src.utils import plot_cost_history, plot_predictions, plot_residuals, plot_feature_importance

class DrugUsagePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drug Usage Predictor - Linear Regression")
        self.root.geometry("800x600")
        self.create_widgets()
        self.pipeline_run = False
        self.results = {}

    def create_widgets(self):
        self.title_label = tk.Label(self.root, text="Drug Usage Predictor", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=10)

        self.run_button = tk.Button(self.root, text="Run Pipeline", command=self.run_pipeline, font=("Arial", 14))
        self.run_button.pack(pady=10)

        self.metrics_frame = tk.LabelFrame(self.root, text="Evaluation Metrics", font=("Arial", 12))
        self.metrics_frame.pack(fill="x", padx=20, pady=10)
        self.metrics_text = tk.Text(self.metrics_frame, height=6, width=80, font=("Arial", 10))
        self.metrics_text.pack()

        self.predictions_frame = tk.LabelFrame(self.root, text="Sample Predictions", font=("Arial", 12))
        self.predictions_frame.pack(fill="x", padx=20, pady=10)
        self.predictions_text = tk.Text(self.predictions_frame, height=6, width=80, font=("Arial", 10))
        self.predictions_text.pack()

        self.images_frame = tk.LabelFrame(self.root, text="Visualizations", font=("Arial", 12))
        self.images_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.image_labels = []
        for i in range(4):
            lbl = tk.Label(self.images_frame)
            lbl.grid(row=0, column=i, padx=5, pady=5)
            self.image_labels.append(lbl)

    def run_pipeline(self):
        try:
            # 1. Generate synthetic data
            X, y = generate_synthetic_data(days=365, random_state=42)
            # 2. Split data
            X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8, random_state=42)
            # 3. Normalize features
            X_train_norm, mean, std = normalize_features(X_train)
            X_test_norm = (X_test - mean) / std
            # 4. Train model
            w, b, J_history = train_model(X_train_norm, y_train, alpha=0.01, num_iters=10000)
            # 5. Evaluate model
            metrics = evaluate_model(X_test_norm, y_test, w, b)
            # 6. Sample predictions
            y_pred_test = predict(X_test_norm[:5], w, b)
            # 7. Visualizations
            feature_names = ['Patient Count', 'Emergency Cases', 'Is Holiday', 'Previous Day Usage']
            y_pred_full = predict(X_test_norm, w, b)
            fig_dir = 'reports/figures'
            os.makedirs(fig_dir, exist_ok=True)
            plot_cost_history(J_history, save_path=os.path.join(fig_dir, 'cost_history.png'))
            plot_predictions(y_test, y_pred_full, save_path=os.path.join(fig_dir, 'predictions.png'))
            plot_residuals(y_test, y_pred_full, save_path=os.path.join(fig_dir, 'residuals.png'))
            plot_feature_importance(w, feature_names=feature_names, save_path=os.path.join(fig_dir, 'feature_importance.png'))
            # Display metrics
            self.metrics_text.delete(1.0, tk.END)
            for k, v in metrics.items():
                self.metrics_text.insert(tk.END, f"{k}: {v:.2f}\n")
            # Display predictions
            self.predictions_text.delete(1.0, tk.END)
            for i in range(5):
                self.predictions_text.insert(tk.END, f"Actual: {y_test[i]:.0f}, Predicted: {y_pred_test[i]:.0f}\n")
            # Display images
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
            messagebox.showinfo("Pipeline Complete", "Pipeline completed successfully! Plots saved to reports/figures/")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = DrugUsagePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
