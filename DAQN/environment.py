import gym
import numpy as np
import random
from gym import spaces
from data_processor import DataProcessor
import torch

class DrugRecommendationEnv(gym.Env):
    """Custom Environment for Drug Recommendation System"""
    def __init__(self):
        super(DrugRecommendationEnv, self).__init__()
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Get available drugs
        self.available_drugs = self.data_processor.get_available_drugs()
        self.n_drugs = len(self.available_drugs)
        
        # get number of unique patients
        self.unique_ids = self.data_processor.patient_data['patient_id'].unique().tolist()
        
        # Calculate state size from preprocessed data
        pa_id = self.data_processor.patient_data['patient_id'].iloc[0]
        data_pa_id = self.data_processor.patient_data[self.data_processor.patient_data['patient_id']==pa_id].sort_values(by=['time_step'])
        obs_features, static_features = self.data_processor.process_patient_data(data_pa_id[self.data_processor.numeric_features+self.data_processor.categorical_features])
        obs_size = obs_features.shape[1]
        static_size = static_features.shape[1]
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_drugs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size, static_size),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Randomly select a patient
        random_patient_id = random.choice(self.unique_ids)
        self.state_data = self.data_processor.patient_data[self.data_processor.patient_data['patient_id']==random_patient_id].sort_values(by=['time_step'])\
                                    [self.data_processor.numeric_features+self.data_processor.categorical_features]
        
        # Process patient data
        self.obs_features, self.static_features= self.data_processor.process_patient_data(self.state_data)
        return (
                torch.tensor(self.obs_features, dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.static_features.toarray(), dtype=torch.float32)  
            )
    
    def step(self, action):
        """Execute action and return new state"""
        # Get drug and patient info
        drug_id = self.available_drugs[action]
        drug = self.data_processor.drug_data[self.data_processor.drug_data['drug_id'] == drug_id].iloc[0]
        
        # Get patient conditions
        condition_features = [f for f in self.data_processor.categorical_features if f.startswith('condition_')]
        patient_conditions = []
        for condition in condition_features:
            value = self.state_data[condition].iloc[0]
            if value != 'none':
                patient_conditions.append(value)

        # Calculate base reward from effectiveness
        reward = self.data_processor.get_drug_effectiveness(drug_id, self.static_features)
        
        # Boost reward for matching conditions
        if drug['primary_condition'] in patient_conditions:
            reward *= 2.0
        if drug['secondary_condition'] in patient_conditions:
            reward *= 1.5
        
        # Penalize for mismatched conditions
        if not any(cond in [drug['primary_condition'], drug['secondary_condition']] 
                  for cond in patient_conditions):
            reward *= 0.3
        
        # Penalize for contraindications
        contraindications = drug['contraindications'].split('|')
        if any(contra in patient_conditions for contra in contraindications):
            reward *= 0.1
        
        # Get patient allergies
        allergies = str(self.state_data['allergies'].iloc[0]).lower()
        if allergies != 'none' and allergies in drug['drug_name'].lower():
            reward *= 0.1
        
        # Consider side effects
        side_effects = drug['side_effects'].split('|')
        if len(side_effects) > 3:
            reward *= 0.9
        
        # Ensure reward is between 0 and 1
        reward = min(max(reward, 0.0), 1.0)
        
        # Move to next state (in this case, same state as it's a single-step environment)
        done = True
        info = {
            'drug_name': drug['drug_name'],
            'primary_condition': drug['primary_condition'],
            'secondary_condition': drug['secondary_condition'],
            'contraindications': drug['contraindications'],
            'side_effects': drug['side_effects']
        }
        
        return (
            torch.tensor(self.obs_features, dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.static_features.toarray(), dtype=torch.float32)
        ), reward, done, info
