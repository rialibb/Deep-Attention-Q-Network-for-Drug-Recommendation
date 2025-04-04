import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from DAQN.train import train_DAQNAgent
from DQN.train import train_DQNAgent

import torch
from DAQN.environment import DrugRecommendationEnvDAQN
from DAQN.daqn_agent import DAQNAgent
from DAQN.data_processor import DataProcessorDAQN

from DQN.environment import DrugRecommendationEnvDQN
from DQN.dqn_agent import DQNAgent
from DQN.data_processor import DataProcessorDQN

import pandas as pd





def comapre_DAQN_vs_DQN(num_runs = 100, num_episodes=1500):
    # Train the agents
    tot_reward_daqn = []
    tot_reward_dqn = []

    for run_ in tqdm(range(num_runs)):
        # fixed epsilon
        _, rewards_daqn = train_DAQNAgent(num_episodes=num_episodes)
        tot_reward_daqn.append(rewards_daqn)

        # variable epsilon
        _, rewards_dqn = train_DQNAgent(num_episodes=num_episodes)
        tot_reward_dqn.append(rewards_dqn)

    # Compute average rewards over runs
    tot_reward_daqn = np.array(tot_reward_daqn)
    tot_reward_dqn = np.array(tot_reward_dqn)

    mean_daqn = np.mean(tot_reward_daqn, axis=0)
    std_daqn = np.std(tot_reward_daqn, axis=0)

    mean_dqn = np.mean(tot_reward_dqn, axis=0)
    std_dqn = np.std(tot_reward_dqn, axis=0)

    # Plot with confidence interval
    plt.figure(figsize=(10, 6))

    # Fixed epsilon plot
    plt.plot(mean_daqn, label='DAQN', color='blue')
    plt.fill_between(range(len(mean_daqn)), mean_daqn - std_daqn, mean_daqn + std_daqn,
                     color='blue', alpha=0.2)

    # Variable epsilon plot
    plt.plot(mean_dqn, label='DQN', color='green')
    plt.fill_between(range(len(mean_dqn)), mean_dqn - std_dqn, mean_dqn + std_dqn,
                     color='green', alpha=0.2)

    # Labels and layout
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Performance of DAQN vs DQN')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/DAQN_vs_DQN.png")










def run_DAQN_epsilon_comparator(fix_eps = 0.1,
                                var_eps_start = 1.0,
                                var_eps_min = 0.05,
                                var_eps_decay = 0.995,
                                num_runs = 100,
                                num_episodes=1500):
    # Train the agents
    tot_reward_fix_eps = []
    tot_reward_var_eps = []

    for run_ in tqdm(range(num_runs)):
        # fixed epsilon
        _, rewards_fix_eps = train_DAQNAgent(num_episodes=num_episodes, epsilon=fix_eps, epsilon_decay=1)
        tot_reward_fix_eps.append(rewards_fix_eps)

        # variable epsilon
        _, rewards_var_eps = train_DAQNAgent(num_episodes=num_episodes, epsilon=var_eps_start, epsilon_min=var_eps_min, epsilon_decay=var_eps_decay)
        tot_reward_var_eps.append(rewards_var_eps)

    # Compute average rewards over runs
    tot_reward_fix_eps = np.array(tot_reward_fix_eps)
    tot_reward_var_eps = np.array(tot_reward_var_eps)

    mean_fix = np.mean(tot_reward_fix_eps, axis=0)
    std_fix = np.std(tot_reward_fix_eps, axis=0)

    mean_var = np.mean(tot_reward_var_eps, axis=0)
    std_var = np.std(tot_reward_var_eps, axis=0)

    # Plot with confidence interval
    plt.figure(figsize=(10, 6))

    # Fixed epsilon plot
    plt.plot(mean_fix, label='Fixed Epsilon (ε=0.1)', color='blue')
    plt.fill_between(range(len(mean_fix)), mean_fix - std_fix, mean_fix + std_fix,
                     color='blue', alpha=0.2)

    # Variable epsilon plot
    plt.plot(mean_var, label='Variable Epsilon (ε-decay)', color='green')
    plt.fill_between(range(len(mean_var)), mean_var - std_var, mean_var + std_var,
                     color='green', alpha=0.2)

    # Labels and layout
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('DAQN: Fixed vs decreasing Epsilon Comparison')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/DAQN/DAQN_epsilon_comparison.png")









def run_DAQN_gamma_comparator(gamma_values=[1.0, 0.95, 0.9],
                              num_runs=100,
                              num_episodes=1500):
    all_means = {}
    all_stds = {}

    for gamma in gamma_values:
        print(f"\nTraining with gamma = {gamma}")
        rewards_per_gamma = []

        for run in tqdm(range(num_runs), desc=f"Gamma={gamma}"):
            _, rewards = train_DAQNAgent(
                gamma=gamma,
                num_episodes=num_episodes
            )
            rewards_per_gamma.append(rewards)

        rewards_per_gamma = np.array(rewards_per_gamma)
        all_means[gamma] = np.mean(rewards_per_gamma, axis=0)
        all_stds[gamma] = np.std(rewards_per_gamma, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, gamma in enumerate(gamma_values):
        mean = all_means[gamma]
        std = all_stds[gamma]
        plt.plot(mean, label=f'γ = {gamma}', color=colors[i % len(colors)])
        plt.fill_between(range(len(mean)), mean - std, mean + std,
                         color=colors[i % len(colors)], alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('DAQN: Comparison Across Discount Factors (γ)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/DAQN/DAQN_gamma_comparison.png")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
def test_model(model_path='trained_models/DAQN/DAQN_var_eps_model.pth', test_file='DAQN/data/DAQN_test_input_format.txt'):
    """Test the trained model with patient data from file"""
    try:
        # Load test data
        test_data = load_test_data(test_file)
        ids = test_data['id'].unique().tolist()
        print(f"\nLoaded {len(ids)} patient(s) from test file.")

        
        if test_file.startswith('DAQN'): 
            # Initialize environment and agent
            env = DrugRecommendationEnvDAQN()
            state_size = env.observation_space.shape
            action_size = env.action_space.n
        
            obs_size, static_size = state_size
            agent = DAQNAgent(obs_size=obs_size, static_size=static_size, action_size=action_size, seq_len=5)
            
            data_processor = DataProcessorDAQN()
        
        else:
            # Initialize environment and agent
            env = DrugRecommendationEnvDQN()
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = DQNAgent(state_size=state_size, action_size=action_size)
            
            data_processor = DataProcessorDQN()
        
        # Load trained model
        agent.load(model_path)
        print("Loaded trained model successfully.")
        
        # Get drug data for recommendations
        drug_data = pd.read_csv('DAQN/data/drug_data_expanded.csv')
        
        # Process each patient
        for id in ids:
            patient = test_data[test_data['id']==id]
            print(f"\nAnalyzing patient {id}:")
            print(f"Age: {patient['age'].iloc[0]}, Gender: {patient['gender'].iloc[0]}")
            print(f"Conditions: {patient['condition_1'].iloc[0]}, {patient['condition_2'].iloc[0]}, {patient['condition_3'].iloc[0]}")
            print(f"Allergies: {patient['allergies'].iloc[0]}")
            
            # Create temporary patient entry in data processor
            state = data_processor.process_patient_data(patient)
                
                
            if test_file.startswith('DAQN'): 
                agent.model.eval()
                # Get top N recommendations from model
                with torch.no_grad():
                    obs_features, static_features = state
                    obs_features = torch.tensor(obs_features, dtype=torch.float32).unsqueeze(0)
                    static_features = torch.tensor(static_features.toarray(), dtype=torch.float32)  
                    q_values = agent.model(obs_features, static_features)
                    
            else:
                # Get top N recommendations from model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = agent.model(state_tensor)
                    
            top_actions = q_values.topk(5).indices[0]
            
            # Evaluate each action with our criteria
            best_action = None
            best_score = -float('inf')
            best_drug = None
            
            for action in top_actions:
                action = action.item()
                drug_id = env.available_drugs[action]
                drug = drug_data[drug_data['drug_id'] == drug_id].iloc[0]
                
                # Calculate score based on condition matching
                score = 0
                conditions = [patient['condition_1'].iloc[0], patient['condition_2'].iloc[0], patient['condition_3'].iloc[0]]
                conditions = [c for c in conditions if c != 'none']
                
                # Primary condition match
                if drug['primary_condition'] in conditions:
                    score += 2.0
                
                # Secondary condition match
                if drug['secondary_condition'] in conditions:
                    score += 1.0
                
                # Penalize for no condition match
                if not any(cond in [drug['primary_condition'], drug['secondary_condition']] 
                            for cond in conditions):
                    score -= 1.0
                
                # Check contraindications
                contraindications = drug['contraindications'].split('|')
                if any(contra in conditions for contra in contraindications):
                    score -= 2.0
                
                # Check allergies
                if str(patient['allergies'].iloc[0]).lower() != 'none' and \
                    str(patient['allergies'].iloc[0]).lower() in drug['drug_name'].lower():
                    score -= 2.0
                
                # Consider side effects
                side_effects = drug['side_effects'].split('|')
                score -= len(side_effects) * 0.1
                
                # Update best action if this score is higher
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_drug = drug
            
            # Get effectiveness for best drug
            effectiveness = data_processor.get_drug_effectiveness(
                env.available_drugs[best_action], static_features if test_file.startswith('DAQN') else state)
            
            print("\nRecommended Medication:")
            print(f"Drug: {best_drug['drug_name']}")
            print(f"Primary Condition: {best_drug['primary_condition']}")
            print(f"Secondary Condition: {best_drug['secondary_condition']}")
            print(f"Predicted Effectiveness: {effectiveness:.2f}")
            print(f"Contraindications: {best_drug['contraindications']}")
            print(f"Possible Side Effects: {best_drug['side_effects']}")
            
            print("\n" + "="*50)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease make sure:")
        print("1. The input file exists and follows the correct format")
        print("2. The trained model file exists")
        print("3. All required data files are present")









def load_test_data(file_path):
    """Load patient data from test file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out comments and empty lines
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    
    if not data_lines:
        raise ValueError("No patient data found in the input file!")
    
    # Create DataFrame with proper columns
    columns = ['id', 'age', 'gender', 'blood_pressure', 'heart_rate', 'temperature', 
              'condition_1', 'condition_2', 'condition_3', 'allergies']
    if file_path.startswith('DAQN'):
        columns.append('time_step')
    
    # Parse each line into a list of values
    data = [line.split(',') for line in data_lines]
    return pd.DataFrame(data, columns=columns)