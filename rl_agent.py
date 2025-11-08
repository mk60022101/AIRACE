import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
import os
from datetime import datetime

from .transition import Transition, TransitionBuffer
from .models import Actor
from .models import Critic
from .state_normalizer import StateNormalizer


class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        """
        Initialize PPO agent for 5G energy saving
        
        Args:
            n_cells (int): Number of cells to control
            n_ues (int): Number of UEs in network
            max_time (int): Maximum simulation time steps
            log_file (str): Path to log file
            use_gpu (bool): Whether to use GPU acceleration
        """
        print("Initializing RL Agent")
        
        # FIXED: Validate inputs
        if n_cells <= 0 or n_ues <= 0 or max_time <= 0:
            raise ValueError(f"Invalid parameters: n_cells={n_cells}, n_ues={n_ues}, max_time={max_time}")
        
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # State dimensions: 17 simulation features + 14 network features + (n_cells * 12) cell features
        self.state_dim = 17 + 14 + (n_cells * 12)
        self.action_dim = n_cells  # Power ratio for each cell
        
        # FIXED: Pass n_cells explicitly
        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=n_cells)
        
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim=256).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # PPO hyperparameters
        self.gamma = 0.99  # Discount factor
        self.lambda_gae = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clipping parameter
        self.ppo_epochs = 10  # Number of PPO update epochs
        
        # FIXED: Adaptive buffer/batch size based on max_time
        self.batch_size = min(64, max_time // 4)  # At least 4 batches per episode
        self.buffer_size = max(max_time * 2, 1024)  # 2 episodes worth
        
        # Experience buffer
        self.buffer = TransitionBuffer(self.buffer_size)
        
        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        
        # FIXED: Track previous power for energy rate calculation
        self.prev_power = None
        
        self.setup_logging(log_file)
        
        self.logger.info(f"PPO Agent initialized: {n_cells} cells, {n_ues} UEs")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
        self.logger.info(f"Device: {self.device}")
    
    def normalize_state(self, state):
        """Normalize state vector with validation"""
        # FIXED: Validate state dimension
        if len(state) != self.state_dim:
            raise ValueError(
                f"State dimension mismatch! Expected {self.state_dim}, got {len(state)}"
            )
        return self.state_normalizer.normalize(state)
    
    def setup_logging(self, log_file):
        """Setup logging configuration"""
        self.logger = logging.getLogger('PPOAgent')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_scenario(self):
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.prev_power = None  # FIXED: Reset for new episode
        self.logger.info(f"Starting episode {self.total_episodes}")
    
    def end_scenario(self):
        self.episode_rewards.append(self.current_episode_reward)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        self.logger.info(f"Episode {self.total_episodes} ended: "
                        f"Steps={self.episode_steps}, "
                        f"Reward={self.current_episode_reward:.2f}, "
                        f"Avg100={avg_reward:.2f}")
        
        # FIXED: Train even if buffer not full (for short episodes)
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()
    
    def get_action(self, state):
        """
        Get action from policy network with validation
        
        Args:
            state: State vector from MATLAB interface (must match state_dim)
            
        Returns:
            action: Power ratios for each cell [0, 1]
        """
        # FIXED: Validate and normalize with error handling
        try:
            state = np.array(state).flatten()
            if len(state) != self.state_dim:
                raise ValueError(f"State size mismatch: expected {self.state_dim}, got {len(state)}")
            
            state = self.normalize_state(state)
        except Exception as e:
            self.logger.error(f"State processing error: {e}")
            raise
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_logstd = self.actor(state_tensor)
            
            if self.training_mode:
                # Sample from policy during training
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                # Use mean during evaluation
                action = action_mean
                log_prob = torch.zeros(1).to(self.device)
        
        # Clamp actions to [0, 1]
        action = torch.clamp(action, 0.0, 1.0)
        
        # Store for experience replay
        self.last_state = state
        self.last_action = action.cpu().numpy().flatten()
        self.last_log_prob = log_prob.cpu().numpy().flatten()
        
        return self.last_action
    
    def calculate_reward(self, prev_state, action, current_state):
        """
        FIXED: Calculate reward with proper state indexing and energy rate
        
        State structure (from StateNormalizer):
        [0-16]:   Simulation features (17)
        [17-30]:  Network features (14)
        [31+]:    Cell features (n_cells * 12)
        
        Key indices:
        - [17]: totalEnergy (cumulative kWh)
        - [18]: activeCells
        - [22]: connectedUEs (FIXED: was using index 5)
        - [11]: avgDropRate (simulation feature)
        - [12]: avgLatency (simulation feature)
        """
        if prev_state is None:
            return 0.0
        
        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()
        
        # FIXED: Use correct indices from state structure
        # Simulation features
        prev_drop_rate = prev_state[11]  # avgDropRate
        curr_drop_rate = current_state[11]
        prev_latency = prev_state[12]    # avgLatency
        curr_latency = current_state[12]
        
        # Network features (start at index 17)
        prev_energy = prev_state[17]      # totalEnergy (cumulative)
        curr_energy = current_state[17]
        prev_connected = prev_state[22]   # connectedUEs
        curr_connected = current_state[22]
        active_cells = current_state[18]  # activeCells
        
        # FIXED: Calculate energy consumption RATE (not cumulative)
        energy_rate = curr_energy - prev_energy  # Energy consumed this timestep
        
        # FIXED: Normalize by number of active cells (penalize high consumption)
        if active_cells > 0:
            energy_per_cell = energy_rate / active_cells
        else:
            energy_per_cell = energy_rate
        
        # FIXED: Energy reward (negative consumption, scaled appropriately)
        energy_reward = -energy_per_cell * 0.01  # Small scale to balance with other rewards
        
        # FIXED: Use config thresholds for penalties
        # dropCallThreshold: 1-2%, latencyThreshold: 50-100ms
        drop_threshold = 2.0   # Adaptive threshold
        latency_threshold = 60.0
        
        # FIXED: Progressive penalties (quadratic growth)
        drop_violation = max(0, curr_drop_rate - drop_threshold)
        drop_penalty = -(drop_violation ** 2) * 2.0  # Quadratic penalty
        
        latency_violation = max(0, (curr_latency - latency_threshold) / 10)
        latency_penalty = -(latency_violation ** 2) * 1.0
        
        # FIXED: Connection stability (only penalize large drops)
        connection_change = curr_connected - prev_connected
        if connection_change < -10:  # Only penalize significant drops
            connection_penalty = connection_change * 0.1
        else:
            connection_penalty = 0
        
        # FIXED: Reward for KPI improvements
        drop_improvement = max(0, prev_drop_rate - curr_drop_rate) * 3.0
        latency_improvement = max(0, (prev_latency - curr_latency) / 10) * 1.0
        
        # FIXED: Bonus for maintaining service with low energy
        if curr_drop_rate < drop_threshold and curr_latency < latency_threshold:
            efficiency_bonus = 2.0 / (energy_per_cell + 1.0)  # Higher bonus for lower energy
        else:
            efficiency_bonus = 0
        
        # Total reward with balanced components
        reward = (
            energy_reward +           # -inf to 0 (small scale)
            drop_penalty +            # -inf to 0
            latency_penalty +         # -inf to 0
            connection_penalty +      # -inf to 0
            drop_improvement +        # 0 to +inf
            latency_improvement +     # 0 to +inf
            efficiency_bonus          # 0 to 2.0
        )
        
        # FIXED: Better clipping range (symmetric, smaller)
        reward = float(np.clip(reward, -20, 20))
        
        # FIXED: Log detailed reward breakdown periodically
        if self.episode_steps % 50 == 0:
            self.logger.debug(
                f"Reward breakdown: energy={energy_reward:.2f}, "
                f"drop_pen={drop_penalty:.2f}, lat_pen={latency_penalty:.2f}, "
                f"conn_pen={connection_penalty:.2f}, drop_imp={drop_improvement:.2f}, "
                f"lat_imp={latency_improvement:.2f}, eff_bonus={efficiency_bonus:.2f}, "
                f"total={reward:.2f}"
            )
        
        return reward
    
    def update(self, state, action, next_state, done):
        """
        Update agent with experience
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Next state
            done: Whether episode is done
        """
        if not self.training_mode:
            return
        
        # Calculate actual reward
        actual_reward = self.calculate_reward(state, action, next_state)

        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += actual_reward
        
        # Convert inputs to numpy if needed
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()
        
        # Ensure proper shapes
        state = self.normalize_state(np.array(state).flatten())
        action = np.array(action).flatten()
        next_state = self.normalize_state(np.array(next_state).flatten())
        
        # Get value estimates
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy().flatten()[0]
            next_value = self.critic(next_state_tensor).cpu().numpy().flatten()[0]
        
        # Create transition
        transition = Transition(
            state=state,
            action=action,
            reward=actual_reward,
            next_state=next_state,
            done=done,
            log_prob=getattr(self, 'last_log_prob', np.array([0.0]))[0],
            value=value
        )
        
        self.buffer.add(transition)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        FIXED: Compute Generalized Advantage Estimation with proper bounds
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # FIXED: For last timestep, use 0 as next value if done
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t] if t < len(next_values) else 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lambda_gae * next_non_terminal * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def train(self):
        """Train the PPO agent"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Get all transitions
        transitions = self.buffer.get_all()
        
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions])
        values = np.array([t.value for t in transitions])
        
        # Compute next values
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_values = self.critic(next_states_tensor).cpu().numpy().flatten()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO training loop
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Compute current policy
                action_mean, action_logstd = self.actor(batch_states)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        # Clear buffer after training
        self.buffer.clear()
        
        self.logger.info(f"Training completed: Actor loss={actor_loss:.4f}, "
                        f"Critic loss={critic_loss:.4f}")
    
    def save_model(self, filepath=None):
        """Save model parameters"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_model_{timestamp}.pth"
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'n_cells': self.n_cells,  # FIXED: Save architecture info
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """FIXED: Load model with architecture validation"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # FIXED: Validate architecture compatibility
        if 'n_cells' in checkpoint and checkpoint['n_cells'] != self.n_cells:
            raise ValueError(
                f"Model architecture mismatch! "
                f"Model trained with {checkpoint['n_cells']} cells, "
                f"but current config has {self.n_cells} cells"
            )
        
        if 'state_dim' in checkpoint and checkpoint['state_dim'] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch! "
                f"Model expects {checkpoint['state_dim']}, "
                f"but current is {self.state_dim}"
            )
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def set_training_mode(self, training):
        """Set training mode"""
        self.training_mode = training
        self.actor.train(training)
        self.critic.train(training)
        self.logger.info(f"Training mode set to {training}")
    
    def get_stats(self):
        """Get training statistics"""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_reward': avg_reward,
            'buffer_size': len(self.buffer),
            'training_mode': self.training_mode,
            'episode_steps': self.episode_steps,
            'current_episode_reward': self.current_episode_reward
        }
