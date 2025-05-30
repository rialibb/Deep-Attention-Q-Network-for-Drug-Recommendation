import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal positional encodings to a sequence of input embeddings.

    Purpose:
    - Enables the model to capture the order of elements in a sequence, which is crucial for modeling time dependencies.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        max_len (int): Maximum sequence length to support (default is 100).

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

    Returns:
        torch.Tensor: The input tensor with positional encodings added, same shape as input.
    """
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd, one dimension left
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    
    
    
    
    
class DAQN(nn.Module):
    """
    Deep Attention Q-Network (DAQN) model combining a Transformer-based attention mechanism over patient observation 
    sequences and static patient features for drug recommendation.

    Purpose:
    - Learns to predict Q-values (expected returns) for drug choices based on temporal patient data and static profile.

    Args:
        obs_size (int): Dimension of each observation in the sequence.
        static_size (int): Dimension of static patient features.
        model_dim (int): Embedding dimension used throughout the network.
        action_size (int): Number of discrete drug actions.
        seq_len (int): Number of time steps in the patient observation sequence.
        n_layers (int): Number of Transformer decoder layers.
        n_heads (int): Number of attention heads in each decoder layer.
        ff_dim (int): Dimension of feedforward layers in the Transformer.
        dropout (float): Dropout rate used in the Transformer layers.

    Inputs:
        obs_history (torch.Tensor): Tensor of shape (batch_size, seq_len, obs_size) representing time series data.
        static_info (torch.Tensor): Tensor of shape (batch_size, static_size) for static patient data.

    Returns:
        torch.Tensor: Q-values for each possible action, shape (batch_size, action_size).
    """
    
    def __init__(self, obs_size, static_size, model_dim, action_size,
                 seq_len, n_layers=4, n_heads=4, ff_dim=128, dropout=0.1):
        
        super(DAQN, self).__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim

        # Linear projection for each observation
        self.obs_embedding = nn.Linear(obs_size, model_dim)

        # Positional encoding for observation sequence
        self.pos_encoder = PositionalEncoding(model_dim, max_len=seq_len)

        # Learnable fixed start token (query) to attend over the history.
        # Initialized as a parameter with shape (1, model_dim)
        self.query_token = nn.Parameter(torch.randn(1, model_dim))

        # Transformer decoder layers (we treat the query as the target, and the embedded history as memory)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Combine the output with static patient info (if available)
        self.static_embedding = nn.Linear(static_size, model_dim)
        # calculate action value function
        self.advantage_stream = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, action_size)
        )

    def forward(self, obs_history, static_info):
        """
        obs_history: Tensor of shape (batch_size, seq_len, obs_size)
        static_info: Tensor of shape (batch_size, static_size)
        Returns:
            Q-values: Tensor of shape (batch_size, action_size)
        """
        batch_size = obs_history.size(0)
        # Embed observations
        embedded = self.obs_embedding(obs_history)  # (B, seq_len, model_dim)
        embedded = self.pos_encoder(embedded)  # add positional encoding

        # Transformer expects input shape: (seq_len, batch, model_dim)
        memory = embedded.transpose(0, 1)  # (seq_len, B, model_dim)

        # Prepare the query token for each batch: shape (1, B, model_dim)
        query = self.query_token.unsqueeze(1).repeat(1, batch_size, 1)

        # Decoder: query attends to the memory
        dec_output = self.transformer_decoder(tgt=query, memory=memory)  # (1, B, model_dim)
        dec_output = dec_output.squeeze(0)  # (B, model_dim)

        # Process static patient information
        static_emb = self.static_embedding(static_info)  # (B, model_dim)

        # Concatenate decoder output and static embedding
        joint = torch.cat([dec_output, static_emb], dim=1)  # (B, model_dim*2)

        # calculate action value function
        q_vals = self.advantage_stream(joint)  # (B, action_size)
        return q_vals
    
    
    
    
    

class DAQNAgent:
    """
    Deep Attention Q-Network Agent for Reinforcement Learning-based Drug Recommendation.

    Purpose:
    - Implements experience replay, epsilon-greedy exploration, and training using temporal patient data with attention.

    Args:
        obs_size (int): Dimension of each observation vector in the sequence.
        static_size (int): Dimension of static patient data.
        action_size (int): Number of discrete actions (drugs).
        model_dim (int): Embedding size used in the Transformer (default: 64).
        seq_len (int): Length of the input observation history (default: 5).
        n_layers (int): Number of decoder layers in the Transformer.
        n_heads (int): Number of attention heads per layer.
        ff_dim (int): Feedforward layer dimension inside Transformer.
        dropout (float): Dropout rate (default: 0.1).
        epsilon (float): Initial exploration rate (default: 1.0).
        epsilon_min (float): Minimum exploration rate (default: 0.05).
        epsilon_decay (float): Decay factor for epsilon (default: 0.995).

    Methods:
        - remember(state, action, reward, next_state, done): stores transition in memory.
        - act(obs_features, static_features, evaluate=False): selects action using ε-greedy strategy.
        - replay(batch_size, gamma): samples and trains on a mini-batch of transitions.
        - save(path): saves model weights and optimizer state.
        - load(path): loads model weights and optimizer state.

    Inputs:
        state (Tuple[Tensor, Tensor]): Tuple containing observation sequence and static info.
        action (int): Selected drug index.
        reward (float): Received reward.
        next_state (Tuple[Tensor, Tensor]): Next patient state.
        done (bool): Episode termination flag.

    Returns:
        - Action index or updated weights depending on method.
    """
    
    def __init__(self,  obs_size, static_size, action_size, model_dim=64, seq_len=5, n_layers=4, n_heads=4, ff_dim=128, dropout=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DAQN(obs_size, static_size, model_dim, action_size, seq_len, n_layers, n_heads, ff_dim, dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, obs_features, static_features, evaluate=False):
        """Return action for given state using epsilon-greedy policy"""
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        self.model.eval()
        with torch.no_grad():
            obs_features = obs_features.to(self.device)
            static_features = static_features.to(self.device)
            act_values = self.model(obs_features, static_features)
            return np.argmax(act_values.cpu().data.numpy())
    
    def replay(self, batch_size, gamma):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch data
        obs_features = torch.stack([i[0][0] for i in minibatch]).to(self.device)       
        static_features = torch.stack([i[0][1] for i in minibatch]).to(self.device)  
        
        actions = torch.LongTensor([[i[1]] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([[i[2]] for i in minibatch]).to(self.device)
        
        next_obs_features = torch.stack([i[3][0] for i in minibatch]).to(self.device)
        next_static_features = torch.stack([i[3][1] for i in minibatch]).to(self.device)
        
        dones = torch.FloatTensor([[i[4]] for i in minibatch]).to(self.device)
        
        # Current Q values
        self.model.train()
        curr_q = self.model(obs_features, static_features).gather(1, actions)
        
        # Next Q values
        next_q = self.model(next_obs_features, next_static_features).detach().max(1)[0].unsqueeze(1)
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
