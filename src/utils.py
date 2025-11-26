import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

class ModelEvaluator:
    """
    Handles the evaluation and visualization of model performance.
    Generates plots for Loss, Accuracy, ROC Curves, and Confusion Matrices.
    """

    def __init__(self, output_dir='results'):
        """
        Args:
            output_dir (str): Directory where plots will be saved.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_training_history(self, history):
        """
        Plots training and validation loss/accuracy over epochs.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot Loss
        ax1.plot(history.history['loss'], label='Train Loss', color='tab:blue')
        ax1.plot(history.history['val_loss'], label='Val Loss', color='tab:orange', linestyle='--')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss (Binary Crossentropy)')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        # Plot Accuracy
        ax2.plot(history.history['accuracy'], label='Train Acc', color='tab:green')
        ax2.plot(history.history['val_accuracy'], label='Val Acc', color='tab:red', linestyle='--')
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()

        # Save
        save_path = os.path.join(self.output_dir, 'training_history.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Training history saved to {save_path}")
        plt.show()

    def plot_roc_curve(self, y_true, y_pred_scores):
        """
        Plots the Receiver Operating Characteristic (ROC) curve.
        
        Args:
            y_true: True binary labels.
            y_pred_scores: Predicted probabilities (output of sigmoid).
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Background Efficiency)')
        plt.ylabel('True Positive Rate (Signal Efficiency)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save
        save_path = os.path.join(self.output_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=300)
        print(f"ROC Curve saved to {save_path}")
        plt.show()
        
        return roc_auc

    def plot_confusion_matrix(self, y_true, y_pred_scores, threshold=0.5):
        """
        Plots a confusion matrix heatmap.
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_scores > threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred_binary)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Background', 'Signal'],
                    yticklabels=['Background', 'Signal'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Threshold={threshold})')
        
        # Save
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300)
        print(f"Confusion Matrix saved to {save_path}")
        plt.close()

    def print_classification_report(self, y_true, y_pred_scores, threshold=0.5):
        """
        Prints standard classification metrics (Precision, Recall, F1).
        """
        y_pred_binary = (y_pred_scores > threshold).astype(int)
        print("\n--- Classification Report ---")
        print(classification_report(y_true, y_pred_binary, target_names=['Background', 'Signal']))