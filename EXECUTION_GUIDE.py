#!/usr/bin/env python3
"""
HearAI-EV Execution Guide
==========================

This file documents how to run the complete system.
"""

# ============================================================================
# OPTION 1: INTERACTIVE MENU (Recommended for first-time users)
# ============================================================================

"""
Step 1: Open PowerShell in the project directory
Step 2: Run the following command:

    python quickstart.py

Step 3: Select from the menu:
    [1] Data Processing (already done - skip this)
    [2] Model Training (train YAMNet model)
    [3] Inference Demo (run predictions)
    [4] Continuous Monitoring (simulate monitoring)
    [5] Complete End-to-End (all of the above)
    [0] Exit

Step 4: Check reports/ directory for results

Time to complete:
- Data Processing: ~10-15 minutes
- Model Training: ~5-10 minutes
- Inference: ~2-5 minutes
- Total: ~30-40 minutes for complete run
"""

# ============================================================================
# OPTION 2: COMMAND LINE (For automated/scripted execution)
# ============================================================================

"""
Complete Demo (all phases):
    python main.py --mode demo

Process audio files only:
    python main.py --mode process --audio-dir data/processed/test --limit 10

Continuous monitoring simulation:
    python main.py --mode monitor

Train model from scratch:
    python yamnet_training.py

Process data:
    python data_processing.py
"""

# ============================================================================
# OPTION 3: PROGRAMMATIC (For integration in other code)
# ============================================================================

"""
from main import HearAISystem
from inference import HearAIPredictor, get_diagnostic_info
from llm_explanations import DiagnosticReport
from ui_interface import AlertDisplay, DiagnosticDashboard

# Initialize
system = HearAISystem()

# Process single file
report = system.process_audio_file('path/to/audio.wav')

# Process batch
results = system.process_batch('data/processed/test', limit=10)

# View results
print(report['vehicle_status'])
print(report['diagnostic_explanation'])
print(report['recommendations'])

# Generate visualizations
system.dashboard.generate_dashboard()
system.dashboard.generate_html_dashboard()

# Get system report
system.generate_report()
"""

# ============================================================================
# SYSTEM REQUIREMENTS VERIFICATION
# ============================================================================

import sys
import os
from pathlib import Path

def verify_system():
    """Verify system is ready to run"""
    
    print("\n" + "="*70)
    print("🔍 SYSTEM VERIFICATION")
    print("="*70)
    
    checks = {
        "Python Version": sys.version_info >= (3, 8),
        "Required Directories Exist": all([
            Path('data/processed').exists(),
            Path('dataset').exists(),
            Path('reports').exists(),
        ]),
        "Core Modules Present": all([
            Path('data_processing.py').exists(),
            Path('yamnet_training.py').exists(),
            Path('inference.py').exists(),
            Path('llm_explanations.py').exists(),
            Path('ui_interface.py').exists(),
            Path('main.py').exists(),
        ]),
        "Dependencies File Present": Path('requirements.txt').exists(),
    }
    
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
    
    if not all(checks.values()):
        print("\n⚠️  Some checks failed!")
        print("Please ensure:")
        print("1. Python 3.8+ is installed")
        print("2. All data directories exist")
        print("3. All Python modules are present")
        print("4. requirements.txt exists")
        return False
    
    print("\n✅ System ready!")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    
    print("\n" + "="*70)
    print("📦 DEPENDENCY CHECK")
    print("="*70)
    
    required_packages = [
        'tensorflow',
        'librosa',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n⚠️  Some packages are missing!")
        print("Install with:")
        print("    pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

# ============================================================================
# EXECUTION PATHS
# ============================================================================

def path_1_interactive():
    """Run interactive menu"""
    print("\n🎬 Starting interactive menu...")
    exec(open('quickstart.py').read())

def path_2_training():
    """Run training only"""
    print("\n🤖 Starting model training...")
    exec(open('yamnet_training.py').read())

def path_3_inference():
    """Run inference demo"""
    print("\n🔍 Starting inference demo...")
    from main import HearAISystem
    system = HearAISystem()
    results = system.process_batch('data/processed/test', limit=10)
    system.dashboard.generate_dashboard()
    system.dashboard.generate_html_dashboard()

def path_4_monitoring():
    """Run continuous monitoring"""
    print("\n⏱️  Starting monitoring simulation...")
    from main import HearAISystem
    system = HearAISystem()
    system.continuous_monitoring_demo(duration_samples=10)

def path_5_complete():
    """Run complete end-to-end"""
    print("\n🎬 Starting complete end-to-end demo...")
    from main import run_demo
    run_demo()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Verify system
    if not verify_system():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Install dependencies and try again:")
        print("    pip install -r requirements.txt")
        sys.exit(1)
    
    # Show execution paths
    print("\n" + "="*70)
    print("🚀 HOW TO RUN THE SYSTEM")
    print("="*70)
    
    print("""
    Path 1: Interactive Menu (Recommended)
    └─ python quickstart.py
    
    Path 2: Command Line - Complete Demo
    └─ python main.py --mode demo
    
    Path 3: Command Line - Process Audio
    └─ python main.py --mode process --limit 10
    
    Path 4: Command Line - Monitoring
    └─ python main.py --mode monitor
    
    Path 5: Direct Training
    └─ python yamnet_training.py
    
    Path 6: Programmatic Usage
    └─ from main import HearAISystem
       system = HearAISystem()
       report = system.process_audio_file('audio.wav')
    """)
    
    print("\n" + "="*70)
    print("📊 EXPECTED OUTPUTS")
    print("="*70)
    
    print("""
    After running, check these files:
    
    Models:
    ✓ models/yamnet_finetuned.h5
    
    Training Results:
    ✓ reports/training_history.csv
    ✓ reports/model_evaluation.json
    ✓ reports/model_evaluation.png
    
    Inference Results:
    ✓ reports/predictions_log.json
    ✓ reports/alert_display.png
    ✓ reports/diagnostic_dashboard.png
    ✓ reports/dashboard.html (open in browser!)
    ✓ reports/system_report.json
    """)
    
    print("\n" + "="*70)
    print("⏱️  ESTIMATED EXECUTION TIMES")
    print("="*70)
    
    print("""
    Data Processing: 10-15 minutes
    Model Training: 5-10 minutes
    Inference (10 files): 1-2 minutes
    Monitoring Simulation: 2-3 minutes
    Visualization: 1-2 minutes
    ─────────────────────────────
    Total (complete run): 20-40 minutes
    
    (Times vary based on hardware and batch size)
    """)
    
    print("\n✅ System is ready!")
    print("\n👉 Choose your execution path above and run it!")
    print("\n" + "="*70 + "\n")
