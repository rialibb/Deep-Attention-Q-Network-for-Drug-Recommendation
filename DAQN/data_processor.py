import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessorDAQN:
    def __init__(self):
        # Load data from data directory
        self.patient_data = pd.read_csv('DAQN/data/patient_data_1000_timeseries.csv')
        self.drug_data = pd.read_csv('DAQN/data/drug_data_expanded.csv')
        self.setup_preprocessing()
    
    def setup_preprocessing(self):
        # Define numeric and categorical columns
        self.numeric_features = ['age', 'heart_rate', 'temperature', 'systolic', 'diastolic']
        self.categorical_features = ['gender', 'condition_1', 'condition_2', 'condition_3', 'allergies']
        
        # Extract blood pressure components for all patients
        bp_components = self.patient_data['blood_pressure'].str.split('/', expand=True)
        self.patient_data['systolic'] = bp_components[0].astype(float)
        self.patient_data['diastolic'] = bp_components[1].astype(float)
        
        # Create individual transformers
        self.preprocessor_num = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor_cat = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Fit both preprocessors separately
        num_df = self.patient_data[self.numeric_features]
        cat_df = self.patient_data[self.categorical_features]
        
        self.preprocessor_num.fit(num_df)
        self.preprocessor_cat.fit(cat_df)
    
    def process_patient_data(self, patient_data):
        """Process a single patient time series"""
        # Convert to DataFrame if Series
        if isinstance(patient_data, pd.Series):
            patient_df = pd.DataFrame([patient_data.to_dict()])
        else:
            patient_df = patient_data.copy()
        
        # Select and order features
        bp_components = patient_df['blood_pressure'].str.split('/', expand=True)
        patient_df['systolic'] = bp_components[0].astype(float)
        patient_df['diastolic'] = bp_components[1].astype(float)
        obs_df = patient_df[self.numeric_features]
        
        static_df = patient_df[self.categorical_features].drop_duplicates()
        
        
        # Transform features
        obs_features = self.preprocessor_num.transform(obs_df)
        static_features = self.preprocessor_cat.transform(static_df)
        
        
        return obs_features, static_features
    
    def get_available_drugs(self):
        """Get list of available drugs"""
        return self.drug_data['drug_id'].tolist()
    
    def get_drug_effectiveness(self, drug_id, static_features):
        """Calculate drug effectiveness score considering conditions and contraindications"""
        drug = self.drug_data[self.drug_data['drug_id'] == drug_id].iloc[0]
        base_score = drug['effectiveness_score']
        
        # Get patient conditions from state
        patient_conditions = []
        condition_features = [f for f in self.categorical_features if f.startswith('condition_')]
        
        # Access OneHotEncoder directly
        onehot = self.preprocessor_cat.named_steps['onehot']
        feature_names = onehot.get_feature_names_out(self.categorical_features)

        # Loop over each condition feature to find its encoded active value
        for condition in condition_features:
            condition_idx = self.categorical_features.index(condition)
            
            # Determine the index range in one-hot vector for this condition
            feature_start = sum(len(cat) for cat in onehot.categories_[:condition_idx])
            feature_end = feature_start + len(onehot.categories_[condition_idx])
            
            # Check which category is active (value == 1.0) in static_features
            for i in range(feature_start, feature_end):
                if static_features[0, i] == 1.0:  # static_features is shape (1, N)
                    condition_value = feature_names[i].split('_', 1)[1]
                    if condition_value != 'none':
                        patient_conditions.append(condition_value)
        
        # Check primary condition match
        if any(cond == drug['primary_condition'] for cond in patient_conditions):
            base_score *= 1.5  # Boost score for primary condition match
        
        # Check secondary condition match
        if any(cond == drug['secondary_condition'] for cond in patient_conditions):
            base_score *= 1.2  # Smaller boost for secondary condition match
        
        # Penalize for no condition match
        if not any(cond in [drug['primary_condition'], drug['secondary_condition']] for cond in patient_conditions):
            base_score *= 0.5
        
        # Check contraindications
        contraindications = drug['contraindications'].split('|')
        if any(contra in patient_conditions for contra in contraindications):
            base_score *= 0.1  # Heavy penalty for contraindications
        
        return min(max(base_score, 0.0), 1.0)  # Ensure score is between 0 and 1
