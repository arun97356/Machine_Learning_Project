import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 50000

# Generate synthetic data with realistic patterns for depression/anxiety
data = {
    # Behavioral Data
    'sleep_quality': np.random.normal(6.5, 2.5, n_samples),  # 0-10 scale, lower = worse sleep
    'physical_activity': np.random.normal(15000, 6000, n_samples),  # steps per day (will be clipped to 1000-35000)
    'social_interaction': np.random.normal(5, 3, n_samples),  # hours per week
    'screen_time': np.random.normal(6, 2.5, n_samples),  # hours per day
    'stress_level': np.random.normal(5, 2.5, n_samples),  # 0-10 scale
    
    # EEG Data (microvolts²)
    'eeg_alpha': np.random.normal(45, 15, n_samples),  # Alpha waves
    'eeg_beta': np.random.normal(25, 10, n_samples),   # Beta waves
    'eeg_theta': np.random.normal(35, 12, n_samples),  # Theta waves
    
    # fMRI Data (BOLD signal change %)
    'fmri_prefrontal': np.random.normal(2.5, 1.2, n_samples),  # Prefrontal cortex activity
    'fmri_amygdala': np.random.normal(1.8, 0.9, n_samples),    # Amygdala activity
}

# Create DataFrame
df = pd.DataFrame(data)

# Apply realistic correlations with depression/anxiety
def apply_depression_patterns(df):
    """Apply realistic patterns that correlate with depression/anxiety"""
    
    # Generate underlying depression/anxiety factor (hidden variable)
    depression_factor = np.random.normal(0, 1, n_samples)
    
    # Apply correlations - these are based on real research findings
    df['sleep_quality'] = np.clip(df['sleep_quality'] - 0.8 * depression_factor, 0, 10)
    df['physical_activity'] = np.clip(df['physical_activity'] - 700 * depression_factor, 1000, 35000)
    df['social_interaction'] = np.clip(df['social_interaction'] - 0.9 * depression_factor, 0, 15)
    df['screen_time'] = np.clip(df['screen_time'] + 0.6 * depression_factor, 0, 15)
    df['stress_level'] = np.clip(df['stress_level'] + 0.9 * depression_factor, 0, 10)
    
    # EEG patterns: Depression often shows altered alpha asymmetry and theta activity
    df['eeg_alpha'] = np.clip(df['eeg_alpha'] - 0.5 * depression_factor, 10, 80)
    df['eeg_beta'] = np.clip(df['eeg_beta'] + 0.4 * depression_factor, 5, 50)
    df['eeg_theta'] = np.clip(df['eeg_theta'] + 0.6 * depression_factor, 10, 60)
    
    # fMRI patterns: Depression shows prefrontal hypoactivity and amygdala hyperactivity
    df['fmri_prefrontal'] = np.clip(df['fmri_prefrontal'] - 0.7 * depression_factor, 0.5, 5)
    df['fmri_amygdala'] = np.clip(df['fmri_amygdala'] + 0.8 * depression_factor, 0.5, 4)
    
    return df, depression_factor

# Apply depression patterns
df, depression_factor = apply_depression_patterns(df)

# Create target variables based on the patterns
def create_target_variables(df, depression_factor):
    """Create depression and anxiety labels based on the synthetic data"""
    
    # Calculate depression score (PHQ-9 like, 0-27)
    depression_score = (
        (10 - df['sleep_quality']) * 0.7 +
    (35000 - df['physical_activity']) * 0.0001 +  # steps per day, higher is better
        (15 - df['social_interaction']) * 0.8 +
        df['stress_level'] * 1.2 +
        (df['fmri_amygdala'] - df['fmri_prefrontal']) * 3 +
        np.random.normal(0, 2, n_samples)
    )
    
    depression_score = np.clip(depression_score, 0, 27)
    
    # Calculate anxiety score (GAD-7 like, 0-21)
    anxiety_score = (
        df['stress_level'] * 1.5 +
        (10 - df['sleep_quality']) * 0.6 +
        df['eeg_beta'] * 0.1 +
        df['fmri_amygdala'] * 2 +
        np.random.normal(0, 1.5, n_samples)
    )
    anxiety_score = np.clip(anxiety_score, 0, 21)
    
    # Create binary classifications
    depression_label = (depression_score >= 10).astype(int)  # PHQ-9 threshold
    anxiety_label = (anxiety_score >= 8).astype(int)         # GAD-7 threshold
    
    # Create severity categories
    def get_depression_severity(score):
        if score < 5: return 'None'
        elif score < 10: return 'Mild'
        elif score < 15: return 'Moderate'
        elif score < 20: return 'Moderately severe'
        else: return 'Severe'
    
    def get_anxiety_severity(score):
        if score < 5: return 'None'
        elif score < 10: return 'Mild'
        elif score < 15: return 'Moderate'
        else: return 'Severe'
    
    depression_severity = [get_depression_severity(score) for score in depression_score]
    anxiety_severity = [get_anxiety_severity(score) for score in anxiety_score]
    
    return depression_score, anxiety_score, depression_label, anxiety_label, depression_severity, anxiety_severity

# Create target variables
(df['phq9_score'], df['gad7_score'], 
 df['depression_label'], df['anxiety_label'],
 df['depression_severity'], df['anxiety_severity']) = create_target_variables(df, depression_factor)

# Add participant ID
df['participant_id'] = [f'P{str(i+1).zfill(3)}' for i in range(n_samples)]

# Reorder columns to put IDs and targets first
column_order = ['participant_id', 'phq9_score', 'gad7_score', 'depression_label', 
                'anxiety_label', 'depression_severity', 'anxiety_severity',
                'sleep_quality', 'physical_activity', 'social_interaction', 
                'screen_time', 'stress_level', 'eeg_alpha', 'eeg_beta', 
                'eeg_theta', 'fmri_prefrontal', 'fmri_amygdala']

df = df[column_order]

# Add some noise to make it more realistic
for col in df.select_dtypes(include=[np.number]).columns:
    if col not in ['participant_id', 'depression_label', 'anxiety_label']:
        df[col] += np.random.normal(0, df[col].std() * 0.1, n_samples)


# Round numerical values for readability, and ensure physical_activity is integer
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].round(3)
df['physical_activity'] = df['physical_activity'].round().astype(int)
df['physical_activity'] = np.clip(df['physical_activity'], 1000, 35000)  # Ensure range and integer

print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn descriptions:")
print(df.describe())
print(f"\nDepression distribution: {df['depression_label'].value_counts()}")
print(f"Anxiety distribution: {df['anxiety_label'].value_counts()}")

# Save to Excel
file_path = 'depression_anxiety_dataset.xlsx'
df.to_excel(file_path, index=False, sheet_name='Mental_Health_Data')

print(f"\nDataset saved as '{file_path}'")

# Create a description sheet
description_data = {
    'Column Name': [
        'participant_id', 'phq9_score', 'gad7_score', 'depression_label', 
        'anxiety_label', 'depression_severity', 'anxiety_severity',
        'sleep_quality', 'physical_activity', 'social_interaction', 
        'screen_time', 'stress_level', 'eeg_alpha', 'eeg_beta', 
        'eeg_theta', 'fmri_prefrontal', 'fmri_amygdala'
    ],
    'Description': [
        'Unique participant identifier',
        'PHQ-9 depression score (0-27, higher = more severe)',
        'GAD-7 anxiety score (0-21, higher = more severe)',
        'Binary depression classification (0: No, 1: Yes)',
        'Binary anxiety classification (0: No, 1: Yes)',
        'Depression severity category',
        'Anxiety severity category',
        'Self-reported sleep quality (0-10, higher = better)',
    'Physical activity in steps per day (1000-35000, integer)',
        'Social interaction hours per week',
        'Screen time hours per day',
        'Self-reported stress level (0-10, higher = more stress)',
        'EEG alpha wave power (microvolts²)',
        'EEG beta wave power (microvolts²)',
        'EEG theta wave power (microvolts²)',
        'fMRI prefrontal cortex activity (BOLD signal change %)',
        'fMRI amygdala activity (BOLD signal change %)'
    ],
    'Range/Values': [
        'P001-P500',
        '0-27',
        '0-21',
        '0, 1',
        '0, 1',
        'None, Mild, Moderate, Moderately severe, Severe',
        'None, Mild, Moderate, Severe',
        '0-10',
    '1000-35000',
        '0-15',
        '0-15',
        '0-10',
        '10-80',
        '5-50',
        '10-60',
        '0.5-5',
        '0.5-4'
    ]
}

description_df = pd.DataFrame(description_data)

# Save description to second sheet
with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
    description_df.to_excel(writer, sheet_name='Column_Descriptions', index=False)

print("Description sheet added to the Excel file!")
print("\nDataset is ready for your ML project!")