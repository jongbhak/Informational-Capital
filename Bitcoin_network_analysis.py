# ic_bitcoin.py
"""
IC calculations for Bitcoin and cryptocurrency networks
"""

import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib

class BitcoinIC(InformationalCapital):
    """Calculate IC for Bitcoin network"""
    
    def __init__(self, temperature: float = 300.0):
        super().__init__(temperature)
        self.bits_per_hash = 256  # SHA-256
        
    def block_information(self, difficulty: float) -> float:
        """
        Calculate information content of a valid block
        
        Parameters:
        -----------
        difficulty : Current network difficulty
        
        Returns:
        --------
        information : Bits of information in block
        """
        # Number of leading zeros required
        leading_zeros = np.log2(difficulty)
        # Information content
        return self.bits_per_hash - leading_zeros
    
    def mining_energy(self, hashrate: float, 
                     efficiency: float = 30e-12) -> float:
        """
        Calculate energy consumption
        
        Parameters:
        -----------
        hashrate : Network hashrate (H/s)
        efficiency : J/hash (default: 30 TH/s per kW)
        
        Returns:
        --------
        power : Power consumption (W)
        """
        return hashrate * efficiency
    
    def network_utility(self, price: float, volume: float,
                       market_cap: float) -> float:
        """Calculate network utility from market metrics"""
        # Normalized price momentum
        price_utility = np.tanh(price / 10000)  # Normalize around $10k
        
        # Volume/market cap ratio (liquidity)
        liquidity = volume / market_cap if market_cap > 0 else 0
        
        # Combined utility
        return 0.7 * price_utility + 0.3 * liquidity
    
    def effective_temperature(self, n_miners: int,
                            block_reward: float) -> float:
        """
        Calculate effective temperature of mining competition
        
        Parameters:
        -----------
        n_miners : Number of active miners
        block_reward : Current block reward (BTC)
        
        Returns:
        --------
        t_eff : Effective temperature (K)
        """
        # Competition increases effective temperature
        competition_factor = np.log(n_miners + 1)
        # Reward decreases effective temperature
        reward_factor = block_reward / 6.25  # Normalized to 2020 reward
        
        return self.temperature * competition_factor / reward_factor
    
    def calculate_network_ic(self, 
                           blockchain_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate IC for Bitcoin network over time
        
        Parameters:
        -----------
        blockchain_data : DataFrame with columns:
            - block_height
            - timestamp
            - difficulty
            - hashrate
            - price
            - volume
            - market_cap
            - n_miners
            - block_reward
        
        Returns:
        --------
        results : DataFrame with IC calculations
        """
        results = []
        cumulative_energy = 0
        cumulative_information = 0
        
        for idx, row in blockchain_data.iterrows():
            # Information per block
            block_info = self.block_information(row['difficulty'])
            cumulative_information += block_info
            
            # Energy per block (10 min average)
            block_energy = self.mining_energy(row['hashrate']) * 600
            cumulative_energy += block_energy
            
            # Utility
            utility = self.network_utility(
                row['price'], 
                row['volume'],
                row['market_cap']
            )
            
            # Effective temperature
            t_eff = self.effective_temperature(
                row['n_miners'],
                row['block_reward']
            )
            
            # Stability factor
            stability = np.exp(-cumulative_energy / (k_B * t_eff))
            
            # Calculate IC
            ic = cumulative_information * utility * stability
            
            results.append({
                'block_height': row['block_height'],
                'timestamp': row['timestamp'],
                'cumulative_information': cumulative_information,
                'cumulative_energy': cumulative_energy,
                'utility': utility,
                'stability': stability,
                'ic_value': ic,
                'ic_density': ic / cumulative_energy,
                'ic_rate': block_info * utility * stability / 600  # per second
            })
        
        return pd.DataFrame(results)
    
    def fetch_blockchain_data(self, 
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """
        Fetch blockchain data from APIs
        
        Parameters:
        -----------
        start_date : Start date (YYYY-MM-DD)
        end_date : End date (YYYY-MM-DD)
        
        Returns:
        --------
        data : DataFrame with blockchain metrics
        """
        # This is a simplified example - in practice, use proper API keys
        # and handle rate limiting
        
        data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end:
            # Placeholder for API calls
            # In reality, fetch from blockchain.info, coinmetrics, etc.
            
            # Simulated data for example
            data.append({
                'timestamp': current_date,
                'block_height': 700000 + len(data) * 144,  # ~144 blocks/day
                'difficulty': 25e12 * (1 + 0.001 * len(data)),
                'hashrate': 150e18 * (1 + 0.001 * len(data)),
                'price': 30000 * (1 + 0.1 * np.random.randn()),
                'volume': 20e9 * (1 + 0.3 * np.random.randn()),
                'market_cap': 600e9,
                'n_miners': 1000000,
                'block_reward': 6.25
            })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)

    def halving_analysis(self, ic_series: pd.DataFrame,
                        halving_dates: List[str]) -> Dict[str, float]:
        """Analyze IC behavior around halving events"""
        results = {}
        
        for halving_date in halving_dates:
            # Get IC before and after halving
            halving_idx = ic_series[
                ic_series['timestamp'] == halving_date
            ].index[0]
            
            before = ic_series.iloc[halving_idx-30:halving_idx]['ic_value'].mean()
            after = ic_series.iloc[halving_idx:halving_idx+30]['ic_value'].mean()
            
            results[halving_date] = {
                'ic_before': before,
                'ic_after': after,
                'ic_change': (after - before) / before,
                'ic_volatility': ic_series.iloc[
                    halving_idx-30:halving_idx+30
                ]['ic_value'].std()
            }
        
        return results
