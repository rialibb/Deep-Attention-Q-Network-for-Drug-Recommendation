from DAQN.environment import DrugRecommendationEnvDAQN
from DAQN.daqn_agent import DAQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os




def train_DAQNAgent(num_episodes=1500, batch_size=32, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
    """
    Trains the DAQN (Deep Attention Q-Network) agent in the custom drug recommendation environment.

    Parameters:
    - num_episodes (int): Total number of training episodes.
    - batch_size (int): Number of experiences used per training batch.
    - gamma (float): Discount factor for future rewards.
    - epsilon (float): Initial exploration rate for epsilon-greedy policy.
    - epsilon_min (float): Minimum exploration rate.
    - epsilon_decay (float): Decay rate of epsilon per episode.

    Returns:
    - agent (DAQNAgent): The trained DAQN agent.
    - rewards (list[float]): List of total rewards obtained per episode.

    Description:
    - Initializes the DAQN agent and environment.
    - For each episode, the agent interacts with the environment using an epsilon-greedy policy.
    - Stores transitions in memory and trains using experience replay.
    - Periodically prints training progress and saves the trained model at the end.
    """
    
    # Initialize environment and agent
    env = DrugRecommendationEnvDAQN()
    obs_size, static_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DAQNAgent(obs_size=obs_size, 
                      static_size=static_size, 
                      action_size=action_size, 
                      seq_len=5, 
                      epsilon=epsilon, 
                      epsilon_min=epsilon_min, 
                      epsilon_decay=epsilon_decay)
    
    # Training loop
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        obs_features, static_features = state
        total_reward = 0
        done = False
        
        while not done:
            # Get action
            action = agent.act(obs_features, static_features)

            # Take action
            next_state, reward, done, _ = env.step(action)
            next_obs_features, next_static_features = next_state
            # Store experience
            agent.remember((obs_features.squeeze(0), static_features.squeeze(0)), action, reward, (next_obs_features.squeeze(0), next_static_features.squeeze(0)), done)
            
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
    
    save_dir = 'trained_models/DAQN'
    os.makedirs(save_dir, exist_ok=True)

    # Save the trained model
    if epsilon == epsilon_min or epsilon_decay == 1:
        model_path = os.path.join(save_dir, 'DAQN_fix_eps_model.pth')
        agent.save(model_path)
        print(f"\nModel saved to {model_path}")
    else:
        model_path = os.path.join(save_dir, 'DAQN_var_eps_model.pth')
        agent.save(model_path)
        print(f"\nModel saved to {model_path}")
        
    return agent, rewards







def evaluate_DAQNAgent(agent, env, n_episodes=1000):
    """
    Evaluates a trained DAQN agent over a number of test episodes.

    Parameters:
    - agent (DAQNAgent): The trained DAQN agent to evaluate.
    - env (DrugRecommendationEnvDAQN): The drug recommendation environment.
    - n_episodes (int): Number of episodes to run for evaluation.

    Returns:
    - None (prints average reward over evaluation episodes).

    Description:
    - Runs the agent in the environment for `n_episodes` using a greedy policy.
    - Tracks and prints the average reward to assess performance on unseen patients.
    """
    
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        obs_features, static_features = state
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(obs_features, static_features)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"\nEvaluation Results:")
    print(f"Average Reward over {n_episodes} episodes: {avg_reward:.2f}")
    
