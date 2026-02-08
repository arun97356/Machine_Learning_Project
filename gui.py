import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import warnings
warnings.filterwarnings('ignore')

class MentalHealthPredictorGUI:
    def __init__(self):
        self.models = None
        self.scalers = None
        
        # Load pre-trained models
        self.load_models()
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Depression & Anxiety Prediction")
        self.root.geometry("500x600")
        self.root.configure(bg='#f0f0f0')
        
        self.setup_gui()
    
    def load_models(self):
        """Load pre-trained models and scalers from .pkl files"""
        try:
            print("Loading pre-trained models...")
            
            # Load models
            self.models = joblib.load('mental_health_models.pkl')
            print("✓ Models loaded successfully")
            
            # Load scalers
            self.scalers = joblib.load('mental_health_scalers.pkl')
            print("✓ Scalers loaded successfully")
            
            # Print model information
            print("\nModel Information:")
            for target in ['depression', 'anxiety']:
                if target in self.models:
                    model_names = list(self.models[target].keys())
                    print(f"  {target.capitalize()}: {', '.join(model_names)}")
            
        except FileNotFoundError:
            print("❌ Model files not found! Please make sure these files exist:")
            print("   - mental_health_models.pkl")
            print("   - mental_health_scalers.pkl")
            raise
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def setup_gui(self):
        # Main title
        title_label = tk.Label(self.root, text="Depression & Anxiety Prediction", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, text="Enter Feature Values:", 
                                font=('Arial', 12, 'bold'), bg='#f0f0f0')
        subtitle_label.pack(pady=5)
        
        # Create input frame
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(pady=10, padx=20, fill='both')
        
        # Feature inputs - EMPTY by default
        self.entries = {}
        features = [
            ('Sleep Quality (0-10):', 'sleep_quality'),
            ('Physical Activity (steps):', 'physical_activity'),
            ('Social Interaction (0-10):', 'social_interaction'),
            ('Screen Time (hours):', 'screen_time'),
            ('Stress Level (0-10):', 'stress_level'),
            ('EEG Alpha:', 'eeg_alpha'),
            ('EEG Beta:', 'eeg_beta'),
            ('EEG Theta:', 'eeg_theta'),
            ('fMRI Prefrontal:', 'fmri_prefrontal'),
            ('fMRI Amygdala:', 'fmri_amygdala')
        ]
        
        for i, (label, key) in enumerate(features):
            row_frame = tk.Frame(input_frame, bg='#f0f0f0')
            row_frame.pack(fill='x', pady=3)
            
            # Label
            tk.Label(row_frame, text=label, width=25, anchor='w', 
                    bg='#f0f0f0', font=('Arial', 10)).pack(side='left')
            
            # Entry field (EMPTY)
            entry = tk.Entry(row_frame, width=15, font=('Arial', 10))
            entry.pack(side='left', padx=5)
            self.entries[key] = entry
        
        # Separator
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x', pady=20, padx=20)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # Predict button
        predict_btn = tk.Button(button_frame, text="Predict", 
                               command=self.predict, 
                               font=('Arial', 12, 'bold'),
                               bg='#4CAF50', fg='white',
                               width=12, height=1)
        predict_btn.pack(side='left', padx=5)
        
        # Clear button
        clear_btn = tk.Button(button_frame, text="Clear All", 
                             command=self.clear_all, 
                             font=('Arial', 12),
                             bg='#f44336', fg='white',
                             width=12, height=1)
        clear_btn.pack(side='left', padx=5)
        
    def clear_all(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
    
    def predict(self):
        try:
            # Get input values
            input_data = []
            empty_fields = []
            
            feature_keys = ['sleep_quality', 'physical_activity', 'social_interaction', 
                          'screen_time', 'stress_level', 'eeg_alpha', 'eeg_beta', 
                          'eeg_theta', 'fmri_prefrontal', 'fmri_amygdala']
            
            for key in feature_keys:
                value = self.entries[key].get().strip()
                if not value:
                    empty_fields.append(key)
                else:
                    input_data.append(float(value))
            
            if empty_fields:
                messagebox.showerror("Input Error", 
                                   f"Please fill all fields. Missing: {', '.join(empty_fields)}")
                return
            
            # Make prediction
            predictions = self.predict_mental_health(input_data)
            
            # Show results
            self.show_results(predictions)
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
    
    def predict_mental_health(self, input_data):
        """Predict depression and anxiety levels using pre-trained models"""
        features = ['sleep_quality', 'physical_activity', 'social_interaction', 
                   'screen_time', 'stress_level', 'eeg_alpha', 'eeg_beta', 
                   'eeg_theta', 'fmri_prefrontal', 'fmri_amygdala']
        
        input_df = pd.DataFrame([input_data], columns=features)
        
        predictions = {}
        
        for target_name in ['depression', 'anxiety']:
            if target_name not in self.models:
                continue
                
            scaler = self.scalers[target_name]
            input_scaled = scaler.transform(input_df)
            available_models = list(self.models[target_name].keys())
            if available_models:
                model_name = available_models[0]
                model = self.models[target_name][model_name]
                prediction = model.predict(input_scaled)[0]
                probability = 0.5
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_scaled)[0][1]
                # Set level based on probability
                if probability < 0.5:
                    level = 'Low'
                elif probability < 0.75:
                    level = 'Moderate'
                else:
                    level = 'High'
                predictions[target_name] = {
                    'level': level,
                    'probability': probability,
                    'model_used': model_name
                }
        
        return predictions
    
    def show_results(self, predictions):
        # Create results window
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction Results")
        result_window.geometry("400x350")
        result_window.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(result_window, text="Prediction Results", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Depression results
        if 'depression' in predictions:
            dep_frame = tk.Frame(result_window, bg='#f0f0f0')
            dep_frame.pack(fill='x', pady=15, padx=20)
            
            tk.Label(dep_frame, text="Depression Level:", font=('Arial', 12, 'bold'), 
                    bg='#f0f0f0').pack(anchor='w')
            
            dep_level = predictions['depression']['level']
            if dep_level == 'Low':
                dep_color = '#28a745'  # Green
            elif dep_level == 'Moderate':
                dep_color = '#FFD600'  # Yellow
            else:
                dep_color = '#dc3545'  # Red
            tk.Label(dep_frame, 
                    text=f"{dep_level}",
                    font=('Arial', 14, 'bold'), bg='#f0f0f0', fg=dep_color).pack(anchor='w')
            
            tk.Label(dep_frame, 
                    text=f"Probability: {predictions['depression']['probability']:.2f}",
                    font=('Arial', 11), bg='#f0f0f0').pack(anchor='w')
            tk.Label(dep_frame, 
                    text=f"Model Used: {predictions['depression']['model_used']}",
                    font=('Arial', 10), bg='#f0f0f0', fg='gray').pack(anchor='w')
            
            sep = ttk.Separator(result_window, orient='horizontal')
            sep.pack(fill='x', pady=10, padx=20)
        
        # Anxiety results
        if 'anxiety' in predictions:
            anx_frame = tk.Frame(result_window, bg='#f0f0f0')
            anx_frame.pack(fill='x', pady=15, padx=20)
            
            tk.Label(anx_frame, text="Anxiety Level:", font=('Arial', 12, 'bold'), 
                    bg='#f0f0f0').pack(anchor='w')
            
            anx_level = predictions['anxiety']['level']
            if anx_level == 'Low':
                anx_color = '#28a745'  # Green
            elif anx_level == 'Moderate':
                anx_color = '#FFD600'  # Yellow
            else:
                anx_color = '#dc3545'  # Red
            tk.Label(anx_frame, 
                    text=f"{anx_level}",
                    font=('Arial', 14, 'bold'), bg='#f0f0f0', fg=anx_color).pack(anchor='w')
            
            tk.Label(anx_frame, 
                    text=f"Probability: {predictions['anxiety']['probability']:.2f}",
                    font=('Arial', 11), bg='#f0f0f0').pack(anchor='w')
            tk.Label(anx_frame, 
                    text=f"Model Used: {predictions['anxiety']['model_used']}",
                    font=('Arial', 10), bg='#f0f0f0', fg='gray').pack(anchor='w')
        
        ok_btn = tk.Button(result_window, text="OK", 
                          command=result_window.destroy,
                          font=('Arial', 12),
                          bg='#2196F3', fg='white',
                          width=10)
        ok_btn.pack(pady=20)
    
    def run(self):
        self.root.mainloop()

# Alternative version if you have different .pkl file names
class MentalHealthPredictorCustom:
    def __init__(self, model_file='mental_health_models.pkl', scaler_file='mental_health_scalers.pkl'):
        self.model_file = model_file
        self.scaler_file = scaler_file
        self.models = None
        self.scalers = None
        
        self.load_models_custom()
        self.setup_gui()
    
    def load_models_custom(self):
        """Load models with custom file names"""
        try:
            print(f"Loading models from: {self.model_file}")
            self.models = joblib.load(self.model_file)
            
            print(f"Loading scalers from: {self.scaler_file}")
            self.scalers = joblib.load(self.scaler_file)
            
            print("✓ All models and scalers loaded successfully!")
            
        except FileNotFoundError:
            print("❌ Model files not found! Looking for:")
            print(f"   - {self.model_file}")
            print(f"   - {self.scaler_file}")
            print("\nAvailable .pkl files in directory:")
            import os
            pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
            for f in pkl_files:
                print(f"   - {f}")
            raise
    
    def setup_gui(self):
        # Same GUI setup as above
        self.root = tk.Tk()
        self.root.title("Depression & Anxiety Prediction")
        self.root.geometry("500x600")
        self.root.configure(bg='#f0f0f0')
        
        # ... (same GUI code as above)
        
    # ... (same methods as above)

def main():
    """Main function to launch the GUI"""
    print("=" * 50)
    print("MENTAL HEALTH PREDICTION GUI")
    print("Using Pre-trained Models")
    print("=" * 50)
    
    try:
        # Try the default version first
        app = MentalHealthPredictorGUI()
        print("✓ GUI initialized successfully")
        print("✓ Ready for predictions!")
        print("=" * 50)
        
        app.run()
        
    except Exception as e:
        print(f"❌ Error with default model files: {e}")
        print("\nTrying with custom file search...")
        
        # Try to find any .pkl files
        import os
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        
        if pkl_files:
            print(f"Found .pkl files: {pkl_files}")
            # You can modify this to use specific files
            # For example, if you have different file names:
            # app = MentalHealthPredictorCustom('your_models.pkl', 'your_scalers.pkl')
        else:
            print("No .pkl files found in the directory.")

if __name__ == "__main__":
    main()