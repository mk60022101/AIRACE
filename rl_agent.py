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
    """
    PPO Agent for 5G Energy Optimization
    
    Objective: Minimize total energy E_total = ΣΣ P_tx,i(t)
    Constraints:
        - avgDropRate ≤ dropCallThreshold
        - avgLatency ≤ latencyThreshold  
        - cpuUsage_i ≤ cpuThreshold
        - prbUsage_i ≤ prbThreshold
    """
    
    def __init__(self, n_cells, n_ues, max_time, 
                 drop_threshold=None, latency_threshold=None,
                 cpu_threshold=None, prb_threshold=None,
                 log_file='rl_agent.log', use_gpu=False):
        """
        Initialize PPO agent
        
        Args:
            n_cells: Number of cells
            n_ues: Number of UEs
            max_time: Simulation time steps
            drop_threshold: Drop call threshold (%)
            latency_threshold: Latency threshold (ms)
            cpu_threshold: CPU usage threshold (%)
            prb_threshold: PRB usage threshold (%)
        """
        print("Initializing RL Agent")
        
        if n_cells <= 0 or n_ues <= 0 or max_time <= 0:
            raise ValueError(f"Invalid parameters: n_cells={n_cells}, n_ues={n_ues}, max_time={max_time}")
        
        self.n_cells = n_cells
        self.n_ues = n_ues
        self.max_time = max_time
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Constraint thresholds (from config)
        self.drop_threshold = drop_threshold if drop_threshold else 2.0
        self.latency_threshold = latency_threshold if latency_threshold else 50.0
        self.cpu_threshold = cpu_threshold if cpu_threshold else 95.0
        self.prb_threshold = prb_threshold if prb_threshold else 95.0
        
        # State: 17 sim + 14 net + (n_cells * 12) cell features
        self.state_dim = 17 + 14 + (n_cells * 12)
        self.action_dim = n_cells  # r_i ∈ [0,1] for each cell
        
        self.state_normalizer = StateNormalizer(self.state_dim, n_cells=n_cells)
        
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=256).to(self.device)
        self.critic = Critic(self.state_dim, hidden_dim=256).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 10
        
        # Adaptive buffer/batch size
        self.batch_size = min(64, max_time // 4)
        self.buffer_size = max(max_time * 2, 1024)
        self.buffer = TransitionBuffer(self.buffer_size)
        
        # Training stats
        self.training_mode = True
        self.total_episodes = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        
        # Tracking for evaluation metrics
        self.episode_total_energy = 0.0
        self.episode_constraint_violations = 0
        self.prev_energy = 0.0
        
        self.setup_logging(log_file)
        
        self.logger.info(f"PPO Agent initialized with constraints:")
        self.logger.info(f"  Drop threshold: {self.drop_threshold}%")
        self.logger.info(f"  Latency threshold: {self.latency_threshold} ms")
        self.logger.info(f"  CPU threshold: {self.cpu_threshold}%")
        self.logger.info(f"  PRB threshold: {self.prb_threshold}%")
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Device: {self.device}")
    
    def normalize_state(self, state):
        """Normalize state with validation"""
        if len(state) != self.state_dim:
            raise ValueError(f"State dimension mismatch! Expected {self.state_dim}, got {len(state)}")
        return self.state_normalizer.normalize(state)
    
    def setup_logging(self, log_file):
        """Setup logging"""
        self.logger = logging.getLogger('PPOAgent')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_scenario(self):
        """Start new episode"""
        self.total_episodes += 1
        self.episode_steps = 0
        self.current_episode_reward = 0.0
        self.episode_total_energy = 0.0
        self.episode_constraint_violations = 0
        self.prev_energy = 0.0
        self.logger.info(f"Starting episode {self.total_episodes}")
    
    def end_scenario(self):
        """End episode and train"""
        self.episode_rewards.append(self.current_episode_reward)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        
        self.logger.info(
            f"Episode {self.total_episodes} ended: "
            f"Steps={self.episode_steps}, "
            f"Reward={self.current_episode_reward:.2f}, "
            f"TotalEnergy={self.episode_total_energy:.2f} kWh, "
            f"Violations={self.episode_constraint_violations}, "
            f"Avg100={avg_reward:.2f}"
        )
        
        if self.training_mode and len(self.buffer) >= self.batch_size:
            self.train()
    
    def get_action(self, state):
        """
        Get action from policy: r_i ∈ [0,1] for each cell i
        
        Returns:
            action: Array of power ratios [r_1, r_2, ..., r_N]
        """
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
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                action = action_mean
                log_prob = torch.zeros(1).to(self.device)
        
        # Clamp to [0, 1]
        action = torch.clamp(action, 0.0, 1.0)
        
        self.last_state = state
        self.last_action = action.cpu().numpy().flatten()
        self.last_log_prob = log_prob.cpu().numpy().flatten()
        
        return self.last_action
    
    def check_constraints(self, state):
        """
        Check if constraints are violated
        
        Constraints:
            1. avgDropRate ≤ dropCallThreshold
            2. avgLatency ≤ latencyThreshold
            3. cpuUsage_i ≤ cpuThreshold for all i
            4. prbUsage_i ≤ prbThreshold for all i
            
        Returns:
            violations: Number of constraint violations
            penalty: Constraint violation penalty
        """
        state = np.array(state).flatten()
        
        violations = 0
        penalty = 0.0
        
        # Quality constraints (from simulation features)
        drop_rate = state[11]  # avgDropRate
        latency = state[12]    # avgLatency
        
        if drop_rate > self.drop_threshold:
            violation_amount = drop_rate - self.drop_threshold
            penalty += violation_amount * 10.0  # Heavy penalty
            violations += 1
        
        if latency > self.latency_threshold:
            violation_amount = (latency - self.latency_threshold) / 10.0
            penalty += violation_amount * 5.0
            violations += 1
        
        # Resource constraints (from cell features)
        # Cell features start at index 31
        cell_start = 31
        
        for i in range(self.n_cells):
            # cpuUsage is 1st cell feature
            cpu_idx = cell_start + i
            cpu_usage = state[cpu_idx]
            
            if cpu_usage > self.cpu_threshold:
                violation_amount = cpu_usage - self.cpu_threshold
                penalty += violation_amount * 2.0
                violations += 1
            
            # prbUsage is 2nd cell feature  
            prb_idx = cell_start + self.n_cells + i
            prb_usage = state[prb_idx]
            
            if prb_usage > self.prb_threshold:
                violation_amount = prb_usage - self.prb_threshold
                penalty += violation_amount * 2.0
                violations += 1
        
        return violations, penalty
    
    def calculate_reward(self, prev_state, action, current_state):
        """
        Calculate reward based on energy minimization with constraint penalties
        
        Objective: Minimize E_total = ΣΣ P_tx,i(t)
        
        Reward formulation:
            r(t) = -ΔE(t) - λ * penalty_constraints(t)
            
        where:
            ΔE(t) = energy consumed in timestep t
            penalty_constraints = sum of all constraint violations
            λ = penalty weight
        """
        if prev_state is None:
            return 0.0
        
        prev_state = np.array(prev_state).flatten()
        current_state = np.array(current_state).flatten()
        
        # Energy from network features (index 17: totalEnergy in kWh)
        prev_total_energy = prev_state[17]
        curr_total_energy = current_state[17]
        
        # Energy consumed this timestep (in kWh)
        delta_energy = curr_total_energy - prev_total_energy
        
        # Track for evaluation
        self.episode_total_energy = curr_total_energy
        
        # Primary objective: Minimize energy
        # Negative because we want to minimize (lower energy = higher reward)
        energy_reward = -delta_energy
        
        # Constraint penalties
        violations, constraint_penalty = self.check_constraints(current_state)
        self.episode_constraint_violations += violations
        
        # Total reward: Energy minimization - Constraint penalties
        # λ = 1.0 (equal weight for constraints)
        reward = energy_reward - constraint_penalty
        
        # Log detailed breakdown every 50 steps
        if self.episode_steps % 50 == 0:
            drop_rate = current_state[11]
            latency = current_state[12]
            active_cells = current_state[18]
            
            self.logger.debug(
                f"Step {self.episode_steps}: "
                f"ΔE={delta_energy:.3f} kWh, "
                f"Violations={violations}, "
                f"Penalty={constraint_penalty:.2f}, "
                f"Reward={reward:.2f} | "
                f"Drop={drop_rate:.2f}%, Lat={latency:.2f}ms, "
                f"ActiveCells={active_cells:.0f}"
            )
        
        # Clip reward to prevent extreme values
        return float(np.clip(reward, -50, 10))
    
    def calculate_mape(self, E_opt):
        """
        Calculate MAPE for this scenario
        
        MAPE = |E_thisinh - E_opt| / E_opt
        
        Args:
            E_opt: Optimal energy consumption (from benchmark)
            
        Returns:
            mape: Mean Absolute Percentage Error
        """
        if E_opt == 0:
            return 1.0  # Maximum penalty if optimal is 0
        
        mape = abs(self.episode_total_energy - E_opt) / E_opt
        
        # Constraint violation penalty: Set MAPE = 1 if constraints violated
        if self.episode_constraint_violations > 0:
            self.logger.warning(
                f"Constraint violations detected ({self.episode_constraint_violations}). "
                f"Setting MAPE = 1.0"
            )
            return 1.0
        
        return mape
    
    def update(self, state, action, next_state, done):
        """Update agent with experience"""
        if not self.training_mode:
            return
        
        actual_reward = self.calculate_reward(state, action, next_state)
        
        self.episode_steps += 1
        self.total_steps += 1
        self.current_episode_reward += actual_reward
        
        # Convert to numpy
        if hasattr(state, 'numpy'):
            state = state.numpy()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if hasattr(next_state, 'numpy'):
            next_state = next_state.numpy()
        
        state = self.normalize_state(np.array(state).flatten())
        action = np.array(action).flatten()
        next_state = self.normalize_state(np.array(next_state).flatten())
        
        # Get value estimates
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(state_tensor).cpu().numpy().flatten()[0]
            next_value = self.critic(next_state_tensor).cpu().numpy().flatten()[0]
        
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
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
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
        """Train PPO agent"""
        if len(self.buffer) < self.batch_size:
            return
        
        transitions = self.buffer.get_all()
        
        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions])
        values = np.array([t.value for t in transitions])
        
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        with torch.no_grad():
            next_values = self.critic(next_states_tensor).cpu().numpy().flatten()
        
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                action_mean, action_logstd = self.actor(batch_states)
                action_std = torch.exp(action_logstd)
                dist = torch.distributions.Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
        
        self.buffer.clear()
        
        self.logger.info(f"Training: Actor loss={actor_loss:.4f}, Critic loss={critic_loss:.4f}")
    
    def save_model(self, filepath=None):
        """Save model"""
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
            'n_cells': self.n_cells,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model with validation"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if 'n_cells' in checkpoint and checkpoint['n_cells'] != self.n_cells:
            raise ValueError(
                f"Architecture mismatch! Model: {checkpoint['n_cells']} cells, Current: {self.n_cells}"
            )
        
        if 'state_dim' in checkpoint and checkpoint['state_dim'] != self.state_dim:
            raise ValueError(
                f"State dim mismatch! Model: {checkpoint['state_dim']}, Current: {self.state_dim}"
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
        self.logger.info(f"Training mode: {training}")
    
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
            'current_episode_reward': self.current_episode_reward,
            'episode_total_energy': self.episode_total_energy,
            'episode_violations': self.episode_constraint_violations
        }
