"""
PHASE 3: INFERENCE & LLM INTEGRATION
=====================================

Real-time fault detection with LLM-based explanations
"""

import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_path': 'models/yamnet_finetuned.h5',
    'classes': ['bearing', 'healthy', 'propeller'],
    'input_sr': 16000,
    'input_length': 16000,  # 1 second at 16kHz
    
    # Confidence thresholds
    'confidence_threshold': 0.7,  # Minimum confidence for classification
    'fault_threshold': 0.5,        # Probability above which it's considered a fault
    
    # Fault descriptions
    'fault_descriptions': {
        'bearing': {
            'severity_high': {
                'name': 'BEARING WEAR - CRITICAL',
                'description': 'Severe bearing wear detected. Ball/roller degradation evident.',
                'symptoms': ['Grinding noise', 'Irregular vibration', 'High-frequency rattling'],
                'action': 'Schedule immediate maintenance. Continued operation risks catastrophic failure.'
            },
            'severity_medium': {
                'name': 'BEARING WEAR - WARNING',
                'description': 'Moderate bearing wear detected. Early stage degradation.',
                'symptoms': ['Persistent humming', 'Slight vibration increase', 'Temperature rise'],
                'action': 'Schedule maintenance within 24-48 hours. Monitor closely.'
            },
            'severity_low': {
                'name': 'BEARING - MINOR ISSUE',
                'description': 'Potential bearing wear detected. Early warning signs present.',
                'symptoms': ['Subtle noise change', 'Micro-vibrations', 'Thermal changes'],
                'action': 'Monitor condition. Maintenance recommended within 1 week.'
            }
        },
        'propeller': {
            'severity_high': {
                'name': 'PROPELLER DAMAGE - CRITICAL',
                'description': 'Severe propeller damage detected. Structural integrity compromised.',
                'symptoms': ['Loud rattling', 'Aerodynamic noise', 'Severe imbalance'],
                'action': 'Stop vehicle immediately. Propeller replacement required.'
            },
            'severity_medium': {
                'name': 'PROPELLER DAMAGE - WARNING',
                'description': 'Moderate propeller damage detected. Performance degradation evident.',
                'symptoms': ['Abnormal whistling', 'Occasional rattling', 'Slight imbalance'],
                'action': 'Reduce speed and schedule maintenance. Avoid extreme conditions.'
            },
            'severity_low': {
                'name': 'PROPELLER - MINOR DAMAGE',
                'description': 'Minor propeller damage detected. Functionality uncompromised.',
                'symptoms': ['Subtle noise change', 'Minor vibration', 'Frequency shifts'],
                'action': 'Monitor performance. Repair recommended during next maintenance.'
            }
        },
        'healthy': {
            'severity_none': {
                'name': 'VEHICLE OPERATING NORMALLY',
                'description': 'All systems functioning within normal parameters.',
                'symptoms': [],
                'action': 'No action required. Continue normal operation.'
            }
        }
    }
}

# ============================================================================
# MODEL INFERENCE
# ============================================================================

class HearAIPredictor:
    """
    Main inference class for fault detection
    """
    
    def __init__(self, model_path=CONFIG['model_path']):
        """Initialize model and load weights"""
        self.model_path = model_path
        self.model = None
        self.classes = CONFIG['classes']
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"📦 Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("✅ Model loaded successfully!")
    
    def preprocess_audio(self, audio_path, sr=16000, duration=1.0):
        """
        Load and preprocess audio file
        """
        try:
            # Load audio
            y, _ = librosa.load(audio_path, sr=sr, duration=duration)
            
            # Ensure correct length
            expected_length = int(sr * duration)
            if len(y) < expected_length:
                y = np.pad(y, (0, expected_length - len(y)), mode='constant')
            else:
                y = y[:expected_length]
            
            # Normalize
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            return y.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error loading audio: {str(e)}")
            return None
    
    def predict(self, audio_input, return_confidence=True):
        """
        Predict fault class from audio
        
        Args:
            audio_input: Path to audio file or numpy array
            return_confidence: Whether to return confidence scores
        
        Returns:
            {
                'prediction': 'bearing',
                'confidence': 0.95,
                'probabilities': {'bearing': 0.95, 'healthy': 0.03, 'propeller': 0.02},
                'is_healthy': False
            }
        """
        # Load audio if path provided
        if isinstance(audio_input, str):
            audio = self.preprocess_audio(audio_input)
        else:
            audio = audio_input
        
        if audio is None:
            return None
        
        # Ensure correct shape
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=0)
        
        # Predict
        probabilities = self.model.predict(audio, verbose=0)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                self.classes[i]: float(probabilities[i])
                for i in range(len(self.classes))
            },
            'is_healthy': predicted_class == 'healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        return result

# ============================================================================
# SEVERITY DETERMINATION
# ============================================================================

def determine_severity(prediction_result):
    """
    Determine fault severity based on confidence and class
    
    Args:
        prediction_result: Output from HearAIPredictor.predict()
    
    Returns:
        severity: 'none', 'low', 'medium', 'high'
        confidence_factor: 0.0-1.0
    """
    confidence = prediction_result['confidence']
    prediction = prediction_result['prediction']
    
    # Healthy vehicle
    if prediction == 'healthy':
        return 'none', confidence
    
    # For faults, use confidence to determine severity
    # High confidence (>0.85) = high severity
    # Medium confidence (0.7-0.85) = medium severity
    # Lower confidence (threshold-0.7) = low severity
    
    if confidence >= 0.85:
        return 'high', confidence
    elif confidence >= 0.70:
        return 'medium', confidence
    else:
        return 'low', confidence

def get_diagnostic_info(prediction_result):
    """
    Get diagnostic information for display
    
    Returns:
        {
            'status': 'HEALTHY' | 'WARNING' | 'CRITICAL',
            'name': 'VEHICLE OPERATING NORMALLY',
            'description': 'All systems functioning...',
            'severity': 'none' | 'low' | 'medium' | 'high',
            'confidence': 0.95,
            'symptoms': [],
            'action': 'No action required...',
            'color': 'green' | 'yellow' | 'red'
        }
    """
    prediction = prediction_result['prediction']
    severity, confidence = determine_severity(prediction_result)
    
    # Map severity to status
    severity_map = {
        'none': 'HEALTHY',
        'low': 'WARNING',
        'medium': 'WARNING',
        'high': 'CRITICAL'
    }
    
    # Map to color
    color_map = {
        'HEALTHY': 'green',
        'WARNING': 'yellow',
        'CRITICAL': 'red'
    }
    
    status = severity_map[severity]
    color = color_map[status]
    
    # Get fault description
    if severity == 'none':
        fault_info = CONFIG['fault_descriptions']['healthy']['severity_none']
    else:
        fault_info = CONFIG['fault_descriptions'][prediction][f'severity_{severity}']
    
    return {
        'status': status,
        'name': fault_info['name'],
        'description': fault_info['description'],
        'severity': severity,
        'confidence': round(confidence * 100, 1),
        'symptoms': fault_info['symptoms'],
        'action': fault_info['action'],
        'color': color,
        'probabilities': prediction_result['probabilities']
    }

# ============================================================================
# CONTINUOUS MONITORING
# ============================================================================

class ContinuousMonitor:
    """
    Simulates continuous vehicle monitoring with 1-minute audio samples
    """
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.history = []
    
    def process_sample(self, audio_path):
        """
        Process a single audio sample
        """
        prediction = self.predictor.predict(audio_path)
        
        if prediction is None:
            return None
        
        diagnostic = get_diagnostic_info(prediction)
        
        self.history.append({
            'timestamp': prediction['timestamp'],
            'prediction': prediction['prediction'],
            'diagnostic': diagnostic
        })
        
        return diagnostic
    
    def get_health_trend(self, window_size=5):
        """
        Analyze recent health trend
        """
        if len(self.history) < window_size:
            recent = self.history
        else:
            recent = self.history[-window_size:]
        
        fault_count = sum(1 for h in recent if not h['prediction'] == 'healthy')
        fault_ratio = fault_count / len(recent) if recent else 0
        
        return {
            'recent_samples': len(recent),
            'fault_detections': fault_count,
            'fault_ratio': round(fault_ratio, 2),
            'trend': 'degrading' if fault_ratio > 0.5 else 'stable'
        }

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def save_prediction_log(predictions, output_path='reports/predictions_log.json'):
    """
    Save prediction history for analysis
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    print(f"✅ Predictions saved to: {output_path}")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_inference():
    """
    Example inference on test data
    """
    print("\n" + "="*70)
    print("INFERENCE EXAMPLE")
    print("="*70)
    
    # Initialize predictor
    predictor = HearAIPredictor()
    
    # Get test files
    test_dir = Path('data/processed/test')
    all_test_files = []
    
    for class_name in CONFIG['classes']:
        class_dir = test_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob('*.wav'))[:2]  # 2 samples per class
            all_test_files.extend(files)
    
    if not all_test_files:
        print("❌ No test files found!")
        return
    
    predictions = []
    
    print(f"\n🔍 Running inference on {len(all_test_files)} test samples...\n")
    
    for audio_file in all_test_files:
        print(f"Processing: {audio_file.name}")
        
        # Predict
        prediction = predictor.predict(str(audio_file))
        
        if prediction:
            diagnostic = get_diagnostic_info(prediction)
            
            print(f"  Prediction: {diagnostic['name']}")
            print(f"  Status: {diagnostic['status']}")
            print(f"  Confidence: {diagnostic['confidence']}%")
            print(f"  Action: {diagnostic['action']}")
            print()
            
            predictions.append(prediction)
    
    # Save logs
    if predictions:
        save_prediction_log(predictions)

if __name__ == "__main__":
    example_inference()
