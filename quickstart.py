#!/usr/bin/env python3
"""
HEARAI-EV QUICK START GUIDE
===========================

Run this script for a complete end-to-end demonstration
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from main import HearAISystem, run_demo
from yamnet_training import main as train_main
from data_processing import main as process_main

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                     🚗 HearAI-EV v1.0 🚗                      ║
    ║          Intelligent Acoustic Diagnostics for EVs             ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print main menu"""
    print("""
    SELECT OPERATION MODE:
    ═══════════════════════════════════════════════════════════════
    
    [1] 📋 DATA PROCESSING
        └─ Scan, validate, augment audio data from dataset/
    
    [2] 🤖 MODEL TRAINING  
        └─ Fine-tune YAMNet on processed data
    
    [3] 🔍 INFERENCE DEMO
        └─ Run predictions on test set with visualizations
    
    [4] ⏱️  CONTINUOUS MONITORING
        └─ Simulate 1-minute audio monitoring
    
    [5] 🎬 COMPLETE END-TO-END DEMO
        └─ Run all phases in sequence
    
    [0] ❌ EXIT
    
    ═══════════════════════════════════════════════════════════════
    """)

def run_phase1():
    """Run data processing phase"""
    print("\n" + "="*70)
    print("PHASE 1: DATA PROCESSING")
    print("="*70)
    try:
        process_main()
        print("\n✅ Phase 1 completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in Phase 1: {str(e)}")

def run_phase2():
    """Run training phase"""
    print("\n" + "="*70)
    print("PHASE 2: MODEL TRAINING")
    print("="*70)
    try:
        train_main()
        print("\n✅ Phase 2 completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in Phase 2: {str(e)}")

def run_phase3():
    """Run inference demo"""
    print("\n" + "="*70)
    print("PHASE 3: INFERENCE & VISUALIZATION")
    print("="*70)
    try:
        system = HearAISystem()
        
        # Process test samples
        test_dir = Path('data/processed/test')
        if test_dir.exists():
            results = system.process_batch(test_dir, limit=10)
            system.dashboard.generate_dashboard()
            system.dashboard.generate_html_dashboard()
            print(f"\n✅ Phase 3 completed! Processed {len(results)} samples")
        else:
            print(f"❌ Test directory not found: {test_dir}")
    
    except Exception as e:
        print(f"\n❌ Error in Phase 3: {str(e)}")

def run_phase4():
    """Run continuous monitoring"""
    print("\n" + "="*70)
    print("PHASE 4: CONTINUOUS MONITORING SIMULATION")
    print("="*70)
    try:
        system = HearAISystem()
        system.continuous_monitoring_demo(duration_samples=10)
        print("\n✅ Phase 4 completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in Phase 4: {str(e)}")

def run_complete_demo():
    """Run complete end-to-end system"""
    print("\n" + "="*70)
    print("🎬 COMPLETE END-TO-END DEMONSTRATION")
    print("="*70)
    
    run_demo()

def main():
    """Main menu loop"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice [0-5]: ").strip()
            
            if choice == '0':
                print("\n👋 Thank you for using HearAI-EV!")
                break
            
            elif choice == '1':
                run_phase1()
            
            elif choice == '2':
                run_phase2()
            
            elif choice == '3':
                run_phase3()
            
            elif choice == '4':
                run_phase4()
            
            elif choice == '5':
                run_complete_demo()
            
            else:
                print("\n❌ Invalid choice. Please try again.")
            
            # Ask to continue
            input("\n\nPress ENTER to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            input("\nPress ENTER to continue...")

if __name__ == "__main__":
    main()
