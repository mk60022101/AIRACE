import numpy as np

class StateNormalizer:
    """Handles state normalization with running statistics"""
    
    def __init__(self, state_dim, epsilon=1e-8, n_cells=10):
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.n_cells = n_cells

        # Simulation features normalization bounds (first 17 features)
        self.simulation_bounds = {
            'totalCells': [12, 60],               # Max 57 (Rural) -> Round up to 60
            'totalUEs': [100, 500],               # Max 300 (Dense Urban) -> Giữ nguyên 500 để linh hoạt
            'simTime': [300, 500],                # Giả định max simTime là 500s (hoặc theo cấu hình kịch bản)
            'timeStep': [1, 1],
            'timeProgress': [0, 1],
            'carrierFrequency': [700e6, 4e9],     # Min 700e6 (Rural), Max 4e9
            'isd': [20, 2000],                    # Min 20m (Indoor), Max 1732m (Rural)
            'minTxPower': [10, 40],               # Min 10 dBm (Indoor), Max 35 dBm (Rural)
            'maxTxPower': [23, 50],               # Min 23 dBm (Indoor), Max 49 dBm (Rural) -> Round up to 50
            'basePower': [50, 1200],              # Min 50 W (Indoor), Max 1200 W (Rural)
            'idlePower': [15, 300],               # Min 15 W (Indoor), Max 300 W (Rural)
            'dropCallThreshold': [1.0, 2.0],
            'latencyThreshold': [50.0, 100.0],
            'cpuThreshold': [90.0, 95.0],
            'prbThreshold': [90.0, 95.0],
            'trafficLambda': [10.0, 30.0],            # Min 10 (Rural), Max 25 (Urban Macro) -> Round up to 30
            'peakHourMultiplier': [1.2, 1.5],     # Min 1.2 (Rural), Max 1.5 (Indoor)
        }
        
        # Network features normalization bounds (next 14 features)
        self.network_bounds = {
            'totalEnergy': [0, 100000],           # kWh
            'activeCells': [0, self.n_cells],              # number of cells
            'avgDropRate': [0.0, 5.0],              # percentage
            'avgLatency': [0.0, 500.0],              # ms
            'totalTraffic': [0, 10000],           # traffic units
            'connectedUEs': [0, self.n_cells * 20],            # number of UEs
            'connectionRate': [0, 100],         # percentage
            'cpuViolations': [0, 500],            # number of violations
            'prbViolations': [0, 500],            # number of violations
            'maxCpuUsage': [0, 100],             # percentage
            'maxPrbUsage': [0, 100],             # percentage
            'kpiViolations': [0, 10000],          # number of violations
            'totalTxPower': [0, 5000],           # total power
            'avgPowerRatio': [0, 1]              # ratio
        }
        
        # Cell features normalization bounds (12 features per cell)
        self.cell_bounds = {
            'cpuUsage': [0.0, 100.0],                # percentage
            'prbUsage': [0.0, 100.0],                # percentage
            'currentLoad': [0.0, 5.0],            # load units
            'maxCapacity': [0.0, 20.0],            # capacity units
            'numConnectedUEs': [0.0, 50.0],          # number of UEs
            'txPower': [10.0, 50.0],                  # dBm
            'energyConsumption': [0, 2000.0],      # watts
            'avgRSRP': [-150.0, -50.0],              # dBm
            'avgRSRQ': [-20.0, -3.0],                 # dB
            'avgSINR': [-15.0, 40.0],                # dB
            'totalTrafficDemand': [0.0, 10.0],      # traffic units
            'loadRatio': [0.0, 2.0]                  # ratio
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
        normalized = np.zeros_like(state_vector)
        
        # Normalize simulation features (indices 0-16)
        simulation_keys = list(self.simulation_bounds.keys())
        for i, key in enumerate(simulation_keys):
            if i < len(state_vector):
                min_val, max_val = self.simulation_bounds[key]
                normalized[i] = self._normalize_value(state_vector[i], min_val, max_val)
        
        # Normalize network features (indices 17-30)
        network_keys = list(self.network_bounds.keys())
        for i, key in enumerate(network_keys):
            global_idx = 17 + i
            if global_idx < len(state_vector):
                min_val, max_val = self.network_bounds[key]
                normalized[global_idx] = self._normalize_value(state_vector[global_idx], min_val, max_val)
        
        # Normalize cell features (indices 31 onwards)
        cell_keys = list(self.cell_bounds.keys())
        start_idx = 31  # After simulation (17) and network (14) features
        
        for feat_idx, key in enumerate(cell_keys):
            min_val, max_val = self.cell_bounds[key]
            
            # Normalize all cells for this feature
            for cell_idx in range(self.n_cells):
                global_idx = start_idx + feat_idx * self.n_cells + cell_idx
                if global_idx < len(state_vector):
                    normalized[global_idx] = self._normalize_value(
                        state_vector[global_idx], min_val, max_val)
        
        return normalized
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize single value to [0, 1] range"""
        if max_val == min_val:
            return 0.5  # Default middle value
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
    
    def update_stats(self, state_vector):
        pass