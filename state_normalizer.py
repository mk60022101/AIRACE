import numpy as np

class StateNormalizer:
    """Handles state normalization with running statistics"""
    
    def __init__(self, state_dim, epsilon=1e-8, n_cells=None):
        """
        Args:
            state_dim: Total state dimension
            epsilon: Small value for numerical stability
            n_cells: Number of cells (MUST be provided, no default!)
        """
        if n_cells is None:
            raise ValueError("n_cells must be explicitly provided!")
        
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.n_cells = n_cells
        
        # Verify state_dim matches expected structure
        expected_dim = 17 + 14 + (n_cells * 12)
        if state_dim != expected_dim:
            raise ValueError(
                f"State dimension mismatch! Expected {expected_dim} "
                f"(17 sim + 14 net + {n_cells}×12 cell), got {state_dim}"
            )

        # Simulation features normalization bounds (first 17 features)
        self.simulation_bounds = {
            'totalCells': [1, 100],              # FIXED: Support up to 100 cells
            'totalUEs': [1, 500],                
            'simTime': [100, 1000],              # FIXED: Match config range
            'timeStep': [1, 10],                 
            'timeProgress': [0, 1],              
            'carrierFrequency': [700e6, 100e9],  # FIXED: Support mmWave
            'isd': [50, 5000],                   # FIXED: Wider range
            'minTxPower': [0, 46],               
            'maxTxPower': [0, 46],              
            'basePower': [100, 200000],          # FIXED: Support high power
            'idlePower': [50, 100000],           # FIXED: Proportional to base
            'dropCallThreshold': [0.1, 10],      # FIXED: Allow < 1%
            'latencyThreshold': [10, 200],       # FIXED: Wider range
            'cpuThreshold': [70, 99],            
            'prbThreshold': [70, 99],            
            'trafficLambda': [0.1, 50],          # FIXED: Support λ=30
            'peakHourMultiplier': [1, 5]         
        }
        
        # Network features normalization bounds (next 14 features)
        self.network_bounds = {
            'totalEnergy': [0, 100000],          # FIXED: Much higher for long sims
            'activeCells': [0, 100],             # FIXED: Match totalCells
            'avgDropRate': [0, 20],              
            'avgLatency': [0, 200],              
            'totalTraffic': [0, 50000],          # FIXED: Higher traffic capacity
            'connectedUEs': [0, 500],            
            'connectionRate': [0, 100],         
            'cpuViolations': [0, 100000],        # FIXED: Long simulation
            'prbViolations': [0, 100000],        
            'maxCpuUsage': [0, 100],             
            'maxPrbUsage': [0, 100],             
            'kpiViolations': [0, 100000],        
            'totalTxPower': [0, 10000],          # FIXED: 100 cells × 46dBm
            'avgPowerRatio': [0, 1]              
        }
        
        # Cell features normalization bounds (12 features per cell)
        self.cell_bounds = {
            'cpuUsage': [0, 100],                
            'prbUsage': [0, 100],                
            'currentLoad': [0, 5000],            # FIXED: Higher capacity
            'maxCapacity': [0, 5000],            
            'numConnectedUEs': [0, 100],         # FIXED: More UEs per cell
            'txPower': [0, 46],                  
            'energyConsumption': [0, 10000],     # FIXED: Higher consumption
            'avgRSRP': [-140, -70],              
            'avgRSRQ': [-20, 0],                 
            'avgSINR': [-10, 30],                
            'totalTrafficDemand': [0, 2000],     # FIXED: Higher demand
            'loadRatio': [0, 1]                  
        }
    
    def normalize(self, state_vector):
        """
        Normalize state vector to [0, 1] range
        
        State structure:
        [sim_1, ..., sim_17,              # Index 0-16 (17 features)
         net_1, ..., net_14,              # Index 17-30 (14 features)
         c1_f1, c2_f1, ..., cn_f1,       # cpuUsage for all cells
         c1_f2, c2_f2, ..., cn_f2,       # prbUsage for all cells
         ...                              # etc for all 12 cell features
         c1_f12, c2_f12, ..., cn_f12]    # loadRatio for all cells
        """
        if len(state_vector) != self.state_dim:
            raise ValueError(
                f"State vector size mismatch! Expected {self.state_dim}, "
                f"got {len(state_vector)}"
            )
        
        normalized = np.zeros_like(state_vector, dtype=np.float32)
        
        # Normalize simulation features (indices 0-16)
        simulation_keys = list(self.simulation_bounds.keys())
        for i, key in enumerate(simulation_keys):
            min_val, max_val = self.simulation_bounds[key]
            normalized[i] = self._normalize_value(state_vector[i], min_val, max_val)
        
        # Normalize network features (indices 17-30)
        network_keys = list(self.network_bounds.keys())
        for i, key in enumerate(network_keys):
            global_idx = 17 + i
            min_val, max_val = self.network_bounds[key]
            normalized[global_idx] = self._normalize_value(
                state_vector[global_idx], min_val, max_val
            )
        
        # Normalize cell features (indices 31 onwards)
        cell_keys = list(self.cell_bounds.keys())
        start_idx = 31  # After simulation (17) and network (14) features
        
        for feat_idx, key in enumerate(cell_keys):
            min_val, max_val = self.cell_bounds[key]
            
            # Normalize all cells for this feature
            for cell_idx in range(self.n_cells):
                global_idx = start_idx + feat_idx * self.n_cells + cell_idx
                normalized[global_idx] = self._normalize_value(
                    state_vector[global_idx], min_val, max_val
                )
        
        return normalized
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize single value to [0, 1] range with safety checks"""
        if max_val == min_val:
            return 0.5  # Default middle value
        
        normalized = (value - min_val) / (max_val - min_val)
        
        # Warn if value is out of expected range (but still clip)
        if normalized < -0.1 or normalized > 1.1:
            print(f"Warning: Value {value} outside bounds [{min_val}, {max_val}]")
        
        return np.clip(normalized, 0.0, 1.0)
    
    def update_stats(self, state_vector):
        """Update running statistics (placeholder for online normalization)"""
        pass
