# ic_core.py
"""
Core module for Informational Capital calculations
"""

import numpy as np
from scipy import stats, integrate
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
R = 8.314462618     # Gas constant (J/mol·K)
h = 6.62607015e-34  # Planck constant (J·s)

@dataclass
class ICComponents:
    """Container for IC components"""
    information: float  # bits
    utility: float      # dimensionless [0,1]
    stability: float    # dimensionless [0,1]
    energy: float       # joules
    temperature: float  # kelvin
    
    @property
    def ic_value(self) -> float:
        """Calculate total IC"""
        return self.information * self.utility * self.stability
    
    @property
    def ic_density(self) -> float:
        """Calculate IC per unit energy"""
        return self.ic_value / self.energy if self.energy > 0 else 0

class InformationalCapital:
    """Base class for IC calculations"""
    
    def __init__(self, temperature: float = 298.15):
        self.temperature = temperature
        self.beta = 1 / (k_B * temperature)
    
    def shannon_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy in bits"""
        # Remove zero probabilities
        p = probabilities[probabilities > 0]
        if len(p) == 0:
            return 0.0
        # Normalize
        p = p / p.sum()
        # Calculate entropy
        return -np.sum(p * np.log2(p))
    
    def boltzmann_factor(self, energy: float) -> float:
        """Calculate Boltzmann factor exp(-E/kT)"""
        return np.exp(-self.beta * energy)
    
    def partition_function(self, energies: np.ndarray) -> float:
        """Calculate partition function Z"""
        return np.sum(np.exp(-self.beta * energies))
    
    def free_energy(self, energies: np.ndarray) -> float:
        """Calculate Helmholtz free energy"""
        Z = self.partition_function(energies)
        return -k_B * self.temperature * np.log(Z)
    
    def calculate_ic(self, 
                    information: Union[float, np.ndarray],
                    utility: Union[float, np.ndarray],
                    energy: Union[float, np.ndarray],
                    normalize: bool = True) -> float:
        """
        Calculate total IC for a system
        
        Parameters:
        -----------
        information : Shannon information content (bits)
        utility : Utility function values [0,1]
        energy : Energy costs (joules)
        normalize : Whether to normalize by partition function
        
        Returns:
        --------
        ic : Total informational capital
        """
        # Convert to arrays
        I = np.atleast_1d(information)
        U = np.atleast_1d(utility)
        E = np.atleast_1d(energy)
        
        # Calculate Boltzmann factors
        boltzmann = self.boltzmann_factor(E)
        
        # Calculate IC
        if normalize:
            Z = np.sum(boltzmann)
            ic = np.sum(I * U * boltzmann) / Z
        else:
            ic = np.sum(I * U * boltzmann)
        
        return ic
    
    def ic_derivative(self, ic_values: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Calculate dIC/dt using finite differences"""
        return np.gradient(ic_values, time)
    
    def ic_autocorrelation(self, ic_series: np.ndarray, max_lag: int = 100) -> np.ndarray:
        """Calculate IC autocorrelation function"""
        ic_norm = ic_series - np.mean(ic_series)
        autocorr = np.correlate(ic_norm, ic_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag]
