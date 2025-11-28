import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class PhysicsVisualizer:
    def __init__(self, style='darkgrid'):
        sns.set_theme(style=style)
        self.colors = ['#1f77b4', '#d62728'] # Blue (Signal), Red (Background)

    def plot_distributions(self, X, y, feature_names, n_cols=3):
        """
        Plots comparison histograms for Signal vs Background.
        Automatically detects PT/Energy columns and applies Log Scale.
        """
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        
        sig = df[df['label'] == 1]
        bkg = df[df['label'] == 0]
        
        n_features = len(feature_names)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(feature_names):
            ax = axes[i]
            
            # Physics Plotting Standard: Step plots with fill
            sns.histplot(sig[col], ax=ax, color=self.colors[0], label='Signal', 
                         element="step", fill=True, stat="density", alpha=0.3)
            sns.histplot(bkg[col], ax=ax, color=self.colors[1], label='Background', 
                         element="step", fill=True, stat="density", alpha=0.3)
            
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.legend()
            
            # Auto-Log scale for Energy/Momentum (High Dynamic Range)
            lower_name = col.lower()
            if any(x in lower_name for x in ['pt', 'energy', 'met', '_e', '_m']):
                ax.set_yscale('log')
                
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    def plot_correlations(self, X, feature_names, threshold=0.0):
        """
        Plots a heatmap of feature correlations. 
        Good for identifying multicollinearity (e.g., MET vs SumET).
        """
        df = pd.DataFrame(X, columns=feature_names)
        corr = df.corr()
        
        # Mask the upper triangle (redundant info)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    annot=False) # Turn annot=True if you have few features
        plt.title('Feature Correlation Matrix')
        plt.show()

    def plot_training_history(self, history):
        """
        Plots Loss and Accuracy curves from the Keras/TensorFlow history object.
        Essential for diagnosing Overfitting vs Underfitting.
        """
        metrics = ['loss', 'accuracy']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for i, metric in enumerate(metrics):
            if metric in history.history:
                train_metric = history.history[metric]
                val_metric = history.history[f'val_{metric}']
                epochs = range(1, len(train_metric) + 1)
                
                axes[i].plot(epochs, train_metric, 'b-', label=f'Training {metric}')
                axes[i].plot(epochs, val_metric, 'r--', label=f'Validation {metric}')
                axes[i].set_title(f'Training and Validation {metric.title()}')
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel(metric.title())
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()