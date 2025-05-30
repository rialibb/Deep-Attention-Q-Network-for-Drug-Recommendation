from DQN.environment import DrugRecommendationEnvDQN
from DQN.dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os




def train_DQNAgent(num_episodes=1500, batch_size=32, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
    """
    Train a Deep Q-Network (DQN) agent on the drug recommendation environment.

    Parameters:
    - num_episodes (int): Total number of episodes for training.
    - batch_size (int): Size of mini-batches used for replay training.
    - gamma (float): Discount factor for future rewards.
    - epsilon (float): Initial exploration rate for epsilon-greedy policy.
    - epsilon_min (float): Minimum value for epsilon after decay.
    - epsilon_decay (float): Multiplicative decay factor for epsilon per episode.

    Returns:
    - agent (DQNAgent): The trained DQN agent.
    - rewards (list): Episode-wise list of total rewards collected during training.

    Description:
    This function trains a DQN agent on a drug recommendation task using an epsilon-greedy strategy
    and experience replay. It periodically saves the model based on the epsilon configuration.
    """
    
    # Initialize environment and agent
    env = DrugRecommendationEnvDQN()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, 
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,  # Increased minimum exploration
                    epsilon_decay=epsilon_decay)  # Slower decay
    
    # Training loop
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train on batch if enough samples
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, gamma)
        
        rewards.append(total_reward)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward:.3f}, Epsilon: {agent.epsilon:.2f}")
    
    save_dir = 'trained_models/DQN'
    os.makedirs(save_dir, exist_ok=True)

    # Save the trained model
    if epsilon == epsilon_min or epsilon_decay == 1:
        model_path = os.path.join(save_dir, 'DQN_fix_eps_model.pth')
        agent.save(model_path)
        print(f"\nModel saved to {model_path}")
    else:
        model_path = os.path.join(save_dir, 'DQN_var_eps_model.pth')
        agent.save(model_path)
        print(f"\nModel saved to {model_path}")
    
    return agent, rewards






def evaluate_DQNAgent(agent, env, n_episodes=100):
    """
    Evaluate a trained DQN agent on a given environment.

    Parameters:
    - agent (DQNAgent): The trained DQN agent to be evaluated.
    - env (gym.Env): The drug recommendation environment.
    - n_episodes (int): Number of evaluation episodes.

    Returns:
    - None

    Description:
    This function runs the trained DQN agent on the environment for a specified number of episodes.
    It calculates and prints the average reward obtained over all episodes, giving insight into
    the agent's generalization performance.
    """
    
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"\nEvaluation Results:")
    print(f"Average Reward over {n_episodes} episodes: {avg_reward:.2f}")
    