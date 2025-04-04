import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F


class DQNAgent:
    """
    Deep Q-Network (DQN) agent for learning drug recommendation policies from tabular patient data.

    Attributes:
    - state_size (int): Dimensionality of patient state input.
    - action_size (int): Number of possible drug actions.
    - memory (deque): Experience replay buffer.
    - gamma (float): Discount factor for future rewards.
    - epsilon (float): Exploration probability.
    - epsilon_min (float): Minimum exploration threshold.
    - epsilon_decay (float): Decay rate for epsilon after each episode.
    - model (nn.Module): Neural network predicting Q-values for actions.
    - optimizer (torch.optim): Optimizer for training the model.
    - device (torch.device): CPU or GPU.

    Methods:
    - act(state: np.ndarray, evaluate: bool) -> int:
        Returns an action using the epsilon-greedy strategy.

    - remember(state, action, reward, next_state, done):
        Stores experience in replay buffer.

    - replay(batch_size: int, gamma: float):
        Samples a batch and trains the Q-network using smooth L1 loss.

    - save(path: str):
        Saves the model state and optimizer to file.

    - load(path: str):
        Loads model and optimizer from a saved checkpoint.
    """
    
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def _build_model(self):
        """Neural Network for Deep Q Learning"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, evaluate=False):
        """Return action for given state using epsilon-greedy policy"""
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return np.argmax(act_values.cpu().data.numpy())
    
    def replay(self, batch_size, gamma):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([i[0] for i in minibatch]).to(self.device)
        actions = torch.LongTensor([[i[1]] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([[i[2]] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor([i[3] for i in minibatch]).to(self.device)
        dones = torch.FloatTensor([[i[4]] for i in minibatch]).to(self.device)
        
        # Current Q values
        curr_q = self.model(states).gather(1, actions)
        
        # Next Q values
        next_q = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = F.smooth_l1_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model from file"""
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
