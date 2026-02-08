import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MentalHealthPredictor:
    def __init__(self):
        self.df = None
        self.models = {}
        self.scalers = {}
        self.accuracy_scores = {}
        
    def load_dataset(self, file_path='depression_anxiety_dataset.xlsx'):
        """Load dataset from Excel file"""
        try:
            self.df = pd.read_excel(file_path)
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Check if required columns exist
            required_columns = ['sleep_quality', 'physical_activity', 'social_interaction', 
                              'screen_time', 'stress_level', 'eeg_alpha', 'eeg_beta', 
                              'eeg_theta', 'fmri_prefrontal', 'fmri_amygdala',
                              'depression_label', 'anxiety_label']
            
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Display dataset info
            print(f"\nDepression distribution:")
            print(self.df['depression_label'].value_counts())
            print(f"\nAnxiety distribution:")
            print(self.df['anxiety_label'].value_counts())
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def train_models(self):
        """Train Random Forest and SVM models for depression and anxiety using entire dataset"""
        # Prepare features and targets
        features = ['sleep_quality', 'physical_activity', 'social_interaction', 
                   'screen_time', 'stress_level', 'eeg_alpha', 'eeg_beta', 
                   'eeg_theta', 'fmri_prefrontal', 'fmri_amygdala']
        
        X = self.df[features]
        
        targets = {
            'depression': self.df['depression_label'],
            'anxiety': self.df['anxiety_label']
        }
        
        # Train models for each target using the entire dataset
        for target_name, y in targets.items():
            # Use entire dataset for training (no test split for final models)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.scalers[target_name] = scaler
            
            # Initialize models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(probability=True, random_state=42)
            }
            
            self.models[target_name] = {}
            self.accuracy_scores[target_name] = {}
            
            # For accuracy evaluation, we'll do a quick train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale the evaluation data
            X_train_eval = scaler.transform(X_train)
            X_test_eval = scaler.transform(X_test)
            
            for model_name, model in models.items():
                # Train model on evaluation split
                model.fit(X_train_eval, y_train)
                
                # Make predictions for evaluation
                y_pred = model.predict(X_test_eval)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                self.accuracy_scores[target_name][model_name] = accuracy
                
                # Now retrain on entire dataset for final model
                final_model = model.__class__(**model.get_params())
                final_model.fit(X_scaled, y)
                
                # Store the final model trained on entire dataset
                self.models[target_name][model_name] = final_model
                
                print(f"{target_name.capitalize()} - {model_name}: {accuracy:.4f}")
                
                # Save confusion matrix
                self.plot_confusion_matrix(y_test, y_pred, target_name, model_name)
        
        # Save models and scalers
        joblib.dump(self.models, 'mental_health_models.pkl')
        joblib.dump(self.scalers, 'mental_health_scalers.pkl')
        
        print("\nModels saved successfully!")
        
    def plot_confusion_matrix(self, y_true, y_pred, target_name, model_name):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low', 'High'], 
                   yticklabels=['Low', 'High'])
        plt.title(f'Confusion Matrix - {target_name.capitalize()} ({model_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{target_name}_{model_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: confusion_matrix_{target_name}_{model_name.replace(' ', '_')}.png")
        
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison for all models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Depression accuracy
        depression_acc = list(self.accuracy_scores['depression'].values())
        models = list(self.accuracy_scores['depression'].keys())
        bars1 = ax1.bar(models, depression_acc, color=['skyblue', 'lightcoral'])
        ax1.set_title('Depression Prediction Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, depression_acc):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Anxiety accuracy
        anxiety_acc = list(self.accuracy_scores['anxiety'].values())
        bars2 = ax2.bar(models, anxiety_acc, color=['skyblue', 'lightcoral'])
        ax2.set_title('Anxiety Prediction Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        for bar, acc in zip(bars2, anxiety_acc):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Accuracy comparison plot saved: model_accuracy_comparison.png")
        
    def predict_mental_health(self, input_data):
        """Predict depression and anxiety levels for any input combination"""
        features = ['sleep_quality', 'physical_activity', 'social_interaction', 
                   'screen_time', 'stress_level', 'eeg_alpha', 'eeg_beta', 
                   'eeg_theta', 'fmri_prefrontal', 'fmri_amygdala']
        
        input_df = pd.DataFrame([input_data], columns=features)
        
        predictions = {}
        
        for target_name in ['depression', 'anxiety']:
            # Scale input using the dataset scaler
            scaler = self.scalers[target_name]
            input_scaled = scaler.transform(input_df)
            
            # Get best model (based on accuracy)
            best_model_name = max(self.accuracy_scores[target_name], 
                                key=self.accuracy_scores[target_name].get)
            best_model = self.models[target_name][best_model_name]
            
            # Predict using the model trained on entire dataset
            prediction = best_model.predict(input_scaled)[0]
            probability = best_model.predict_proba(input_scaled)[0][1]
            
            predictions[target_name] = {
                'level': 'Low' if prediction == 0 else 'High',
                'probability': probability,
                'model_used': best_model_name
            }
            
        return predictions

# Main execution
if __name__ == "__main__":
    print("Loading dataset and training models...")
    
    # Create predictor and load existing dataset
    predictor = MentalHealthPredictor()
    
    try:
        # Load your existing Excel file
        predictor.load_dataset('depression_anxiety_dataset.xlsx')
        
        # Train models
        predictor.train_models()
        predictor.plot_accuracy_comparison()
        
        print("\nTraining completed!")
        print("Confusion matrices and accuracy plots saved as images.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure 'depression_anxiety_dataset.xlsx' exists in the same directory.")