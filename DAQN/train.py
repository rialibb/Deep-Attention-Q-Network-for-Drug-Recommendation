from environment import DrugRecommendationEnv
from dqn_agent import DAQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm





def train_DAQNAgent(num_episodes=1500, batch_size=32, gamma=0.95):
    """Train the DQN agent"""
    # Initialize environment and agent
    env = DrugRecommendationEnv()
    obs_size, static_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DAQNAgent(obs_size=obs_size, static_size=static_size, action_size=action_size, seq_len=5)
    
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
    
    # Save the trained model
    agent.save('DAQN/DAQN_model.pth')
    print("\nModel saved to DAQN_model.pth")
    
    return agent, rewards







def evaluate_DAQNAgent(agent, env, n_episodes=1000):
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
    
    
    
    
    
    
if __name__ == "__main__":
    # Train the agent
    tot_reward=[]
    for run_ in tqdm(range(100)):
        
        trained_agent, rewards = train_DAQNAgent()
        tot_reward.append(rewards)

    # Compute average rewards over runs
    tot_reward = np.array(tot_reward)
    mean_reward = np.mean(tot_reward, axis=0)
    std_reward = np.std(tot_reward, axis=0)

    # Plot with confidence interval
    plt.figure(figsize=(10, 6))
    plt.plot(mean_reward, label='Average Reward per Episode', color='blue')
    plt.fill_between(range(len(mean_reward)), mean_reward - std_reward, mean_reward + std_reward,
                     color='blue', alpha=0.2, label='±1 Std Dev')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Performance of DAQN Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("DAQN/DAQN_reward.png")
    
    # Create new environment for evaluation
    eval_env = DrugRecommendationEnv()
    
    # Evaluate the trained agent
    evaluate_DAQNAgent(trained_agent, eval_env)
