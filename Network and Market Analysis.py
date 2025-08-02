# ic_networks.py
"""
IC calculations for complex networks and markets
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import linalg
from typing import Dict, List, Tuple, Optional
import warnings

class NetworkIC(InformationalCapital):
    """Calculate IC for network systems"""
    
    def __init__(self, temperature: float = 298.15):
        super().__init__(temperature)
    
    def network_entropy(self, G: nx.Graph) -> float:
        """Calculate network structure entropy"""
        # Degree distribution entropy
        degrees = np.array([d for n, d in G.degree()])
        degree_probs = degrees / degrees.sum()
        degree_entropy = self.shannon_entropy(degree_probs)
        
        # Edge weight entropy (if weighted)
        if nx.is_weighted(G):
            weights = np.array([d['weight'] for u, v, d in G.edges(data=True)])
            weight_probs = weights / weights.sum()
            weight_entropy = self.shannon_entropy(weight_probs)
        else:
            weight_entropy = 0
        
        return degree_entropy + weight_entropy
    
    def algebraic_connectivity(self, G: nx.Graph) -> float:
        """Calculate second smallest eigenvalue of Laplacian"""
        if not nx.is_connected(G):
            # Return connectivity of largest component
            G = G.subgraph(max(nx.connected_components(G), key=len))
        
        L = nx.laplacian_matrix(G).astype(float)
        eigenvalues = linalg.eigvalsh(L.toarray())
        # Second smallest eigenvalue
        return sorted(eigenvalues)[1]
    
    def network_utility(self, G: nx.Graph,
                       node_values: Optional[Dict] = None) -> float:
        """Calculate network utility from structure and flow"""
        # Clustering coefficient (local efficiency)
        clustering = nx.average_clustering(G)
        
        # Path efficiency
        if nx.is_connected(G):
            avg_path = nx.average_shortest_path_length(G)
            efficiency = 1 / avg_path
        else:
            efficiency = 0
        
        # Node value contribution
        if node_values:
            value_sum = sum(node_values.values())
            value_var = np.var(list(node_values.values()))
            value_utility = value_sum / (value_sum + value_var)
        else:
            value_utility = 1
        
        return clustering * efficiency * value_utility
    
    def calculate_network_ic(self, G: nx.Graph,
                           node_values: Optional[Dict] = None,
                           edge_flows: Optional[Dict] = None) -> ICComponents:
        """
        Calculate IC for a network
        
        Parameters:
        -----------
        G : NetworkX graph
        node_values : Dict mapping nodes to values
        edge_flows : Dict mapping edges to flow rates
        
        Returns:
        --------
        ic_components : ICComponents object
        """
        # Information content
        info = self.network_entropy(G) * G.number_of_nodes()
        
        # Utility
        utility = self.network_utility(G, node_values)
        
        # Stability (from algebraic connectivity)
        lambda_2 = self.algebraic_connectivity(G)
        stability = 1 - np.exp(-lambda_2)
        
        # Energy (from edge flows or degree sum)
        if edge_flows:
            energy = sum(edge_flows.values()) * 1e-9  # Convert to joules
        else:
            energy = G.number_of_edges() * 1e-9
        
        return ICComponents(
            information=info,
            utility=utility,
            stability=stability,
            energy=energy,
            temperature=self.temperature
        )
    
    def market_network_ic(self, 
                         trades: pd.DataFrame,
                         time_window: str = '1H') -> pd.DataFrame:
        """
        Calculate IC for trading network over time
        
        Parameters:
        -----------
        trades : DataFrame with columns:
            - timestamp
            - buyer_id
            - seller_id  
            - price
            - volume
        time_window : Pandas time window (e.g., '1H', '1D')
        
        Returns:
        --------
        ic_series : Time series of IC values
        """
        # Group by time window
        trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        grouped = trades.set_index('timestamp').groupby(pd.Grouper(freq=time_window))
        
        results = []
        
        for time, group in grouped:
            if len(group) == 0:
                continue
                
            # Build network for this time window
            G = nx.Graph()
            
            for _, trade in group.iterrows():
                if G.has_edge(trade['buyer_id'], trade['seller_id']):
                    G[trade['buyer_id']][trade['seller_id']]['weight'] += trade['volume']
                else:
                    G.add_edge(trade['buyer_id'], trade['seller_id'], 
                              weight=trade['volume'])
            
            if G.number_of_nodes() < 2:
                continue
            
            # Node values (total volume)
            node_values = {}
            for node in G.nodes():
                volume = sum([d['weight'] for u, v, d in G.edges(node, data=True)])
                node_values[node] = volume
            
            # Calculate IC
            ic_comp = self.calculate_network_ic(G, node_values)
            
            results.append({
                'timestamp': time,
                'n_traders': G.number_of_nodes(),
                'n_trades': G.number_of_edges(),
                'total_volume': group['volume'].sum(),
                'avg_price': group['price'].mean(),
                'information': ic_comp.information,
                'utility': ic_comp.utility,
                'stability': ic_comp.stability,
                'ic_value': ic_comp.ic_value,
                'ic_density': ic_comp.ic_density
            })
        
        return pd.DataFrame(results)

class EcologicalIC(InformationalCapital):
    """Calculate IC for ecological communities"""
    
    def __init__(self, temperature: float = 298.15):
        super().__init__(temperature)
    
    def species_diversity_ic(self, 
                           abundance_matrix: np.ndarray,
                           interaction_matrix: np.ndarray) -> float:
        """
        Calculate IC from species abundances and interactions
        
        Parameters:
        -----------
        abundance_matrix : Species x Time abundance matrix
        interaction_matrix : Species x Species interaction strengths
        
        Returns:
        --------
        ic : Community IC value
        """
        n_species, n_time = abundance_matrix.shape
        
        # Shannon diversity at each time
        diversities = []
        for t in range(n_time):
            abundances = abundance_matrix[:, t]
            if abundances.sum() > 0:
                probs = abundances / abundances.sum()
                diversities.append(self.shannon_entropy(probs))
        
        # Average diversity
        avg_diversity = np.mean(diversities)
        
        # Network stability from interaction matrix
        eigenvalues = np.linalg.eigvals(interaction_matrix)
        max_real = np.max(np.real(eigenvalues))
        stability = np.exp(-max_real) if max_real > 0 else 1
        
        # Metabolic utility (from total abundance)
        total_abundance = abundance_matrix.sum()
        utility = np.tanh(total_abundance / n_species)
        
        # Energy (simplified - proportional to metabolism)
        energy = total_abundance * 1e-15  # joules
        
        return self.calculate_ic(
            information=avg_diversity * n_species,
            utility=utility,
            energy=energy,
            normalize=False
        ) * stability
