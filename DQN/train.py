from environment import DrugRecommendationEnv
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




def train_DQNAgent(num_episodes=1500, batch_size=32, gamma=0.95):
    """Train the DQN agent"""
    # Initialize environment and agent
    env = DrugRecommendationEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, 
                    epsilon=1.0,
                    epsilon_min=0.05,  # Increased minimum exploration
                    epsilon_decay=0.995)  # Slower decay
    
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
    
    # Save the trained model
    agent.save('DQN/DQN_model.pth')
    print("\nModel saved to DQN_model.pth")
    
    return agent, rewards





def evaluate_DQNAgent(agent, env, n_episodes=100):
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
    
    
    
    
    
    
if __name__ == "__main__":
    # Train the agent
    tot_reward=[]
    for run_ in tqdm(range(100)):
        
        trained_agent, rewards = train_DQNAgent()
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
    plt.title('Training Performance of DQN Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("DQN/DQN_reward.png")
    
    # Create new environment for evaluation
    eval_env = DrugRecommendationEnv()
    
    # Evaluate the trained agent
    evaluate_DQNAgent(trained_agent, eval_env)
