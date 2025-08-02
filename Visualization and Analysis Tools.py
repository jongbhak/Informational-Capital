# ic_visualization.py
"""
Visualization tools for IC analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ICVisualizer:
    """Visualization tools for IC data"""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    def plot_ic_components(self, ic_data: pd.DataFrame,
                          title: str = "IC Components Over Time"):
        """Plot IC components separately"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Information
        axes[0, 0].plot(ic_data.index, ic_data['information'], 
                       color=self.colors[0], linewidth=2)
        axes[0, 0].set_ylabel('Information (bits)')
        axes[0, 0].set_title('Information Content')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Utility
        axes[0, 1].plot(ic_data.index, ic_data['utility'],
                       color=self.colors[3], linewidth=2)
        axes[0, 1].set_ylabel('Utility')
        axes[0, 1].set_title('Utility Function')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Stability
        axes[1, 0].plot(ic_data.index, ic_data['stability'],
                       color=self.colors[6], linewidth=2)
        axes[1, 0].set_ylabel('Stability')
        axes[1, 0].set_title('Thermodynamic Stability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total IC
        axes[1, 1].plot(ic_data.index, ic_data['ic_value'],
                       color=self.colors[9], linewidth=3)
        axes[1, 1].set_ylabel('IC Value')
        axes[1, 1].set_title('Total Informational Capital')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_phase_diagram(self, 
                          temperature: np.ndarray,
                          energy: np.ndarray,
                          ic_values: np.ndarray):
        """Plot IC phase diagram"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create contour plot
        T_mesh, E_mesh = np.meshgrid(temperature, energy)
        contour = ax.contourf(T_mesh, E_mesh, ic_values, levels=20, cmap='RdBu_r')
        
        # Add contour lines
        lines = ax.contour(
