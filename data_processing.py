"""
EV ACOUSTIC DIAGNOSTICS - PHASE 1: DATA PROCESSING (FINAL)
===========================================================

"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # UPDATE THIS PATH to your actual folder path
    'original_dir': 'dataset',
    'processed_dir': 'data/processed',
    'reports_dir': 'reports',
    'logs_dir': 'logs',
    
    # Map folder names to class names
    'folder_to_class': {
        'Bearing \\M4': 'bearing',  # Note the backslash
        'Healthy': 'healthy',
        'Propeller': 'propeller'
    },
    
    'target_sr': 16000,  # YAMNet requirement
    'test_size': 0.2,
    'val_size': 0.2,
    
    'augmentations_per_file': 20,
    'random_seed': 42,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging():
    """Create log file for this run"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory
    Path(CONFIG['logs_dir']).mkdir(parents=True, exist_ok=True)
    log_file = Path(CONFIG['logs_dir']) / f'phase1_{timestamp}.log'
    
    class Logger:
        def __init__(self, filename):
            self.terminal = __import__('sys').stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            pass
    
    import sys
    sys.stdout = Logger(log_file)
    print(f"Logging to: {log_file}")
    return log_file

def get_class_name(folder_name):
    """
    Map folder name to class name
    """
    # Direct mapping
    if folder_name in CONFIG['folder_to_class']:
        return CONFIG['folder_to_class'][folder_name]
    
    # Try without backslash variants
    for key, value in CONFIG['folder_to_class'].items():
        if key.replace('\\', '').replace(' ', '').lower() in folder_name.replace('\\', '').replace(' ', '').lower():
            return value
    
    # Fallback to folder name
    return folder_name.lower().replace('\\', '').replace(' ', '_')

def find_all_audio_files(base_dir):
    """
    Find ALL audio files in the dataset
    
    Returns: List of dicts with file info
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    all_files = []
    
    base_path = Path(base_dir)
    
    # Find all top-level class folders
    for class_folder in base_path.iterdir():
        if not class_folder.is_dir():
            continue
        
        class_name = get_class_name(class_folder.name)
        
        print(f"\nScanning: {class_folder.name} → Class: {class_name}")
        
        # Check if this folder has subfolders (M1, M2, M3) or files directly
        subfolders = [f for f in class_folder.iterdir() if f.is_dir()]
        
        if len(subfolders) > 0:
            # Has subfolders (like Healthy/M1, Healthy/M2)
            for subfolder in subfolders:
                print(f"  Subfolder: {subfolder.name}")
                
                for ext in audio_extensions:
                    for file_path in subfolder.glob(f'*{ext}'):
                        # Create unique identifier: healthy_M2_1400
                        unique_id = f"{class_name}_{subfolder.name}_{file_path.stem}"
                        
                        all_files.append({
                            'filepath': str(file_path),
                            'class': class_name,
                            'subfolder': subfolder.name,
                            'filename': file_path.name,
                            'unique_id': unique_id,
                            'original_stem': file_path.stem  # For grouping originals
                        })
        else:
            # No subfolders, files directly in class folder
            print(f"  Direct files (no subfolders)")
            
            for ext in audio_extensions:
                for file_path in class_folder.glob(f'*{ext}'):
                    # Create unique identifier: bearing_1700
                    unique_id = f"{class_name}_{file_path.stem}"
                    
                    all_files.append({
                        'filepath': str(file_path),
                        'class': class_name,
                        'subfolder': 'root',
                        'filename': file_path.name,
                        'unique_id': unique_id,
                        'original_stem': file_path.stem
                    })
    
    return all_files

def load_and_validate_audio(file_path, target_sr=16000):
    """
    Load audio file and perform quality checks
    
    Returns: (audio_data, sample_rate, is_valid, issues)
    """
    issues = []
    
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Check 1: Duration
        duration = len(y) / sr
        if duration < 0.5:
            issues.append(f"Too short ({duration:.2f}s)")
        
        # Check 2: Silence
        if np.max(np.abs(y)) < 0.001:
            issues.append("Mostly silent")
        
        # Check 3: Clipping
        clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y)
        if clipping_ratio > 0.01:
            issues.append(f"Clipping detected ({clipping_ratio*100:.1f}%)")
        
        # Check 4: NaN or Inf
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            issues.append("Contains NaN or Inf")
        
        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y)) * 0.95
        
        is_valid = len(issues) == 0
        
        return y, sr, is_valid, issues
    
    except Exception as e:
        return None, None, False, [f"Load error: {str(e)}"]

# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

def add_white_noise(audio, snr_db=20):
    """Add white noise at specified SNR"""
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def add_pink_noise(audio, snr_db=20):
    """Add pink noise (1/f noise)"""
    # Generate white noise
    white = np.random.randn(len(audio))
    
    # Apply 1/f filter in frequency domain
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white))
    
    # Avoid division by zero
    freqs[0] = 1e-10
    
    # Apply 1/f characteristic
    pink_fft = fft_white / np.sqrt(freqs)
    pink = np.fft.irfft(pink_fft, n=len(audio))
    
    # Normalize and scale by SNR
    pink = pink / np.std(pink) * np.std(audio)
    signal_power = np.mean(audio ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    pink = pink * np.sqrt(noise_power / np.mean(pink ** 2))
    
    return audio + pink

def augment_single_file(audio, sr, output_path, base_filename):
    """
    Create 20 augmented versions of audio
    
    base_filename example: healthy_M2_1400
    
    Returns: List of created file paths
    """
    augmented_files = []
    
    # 1. Original (just save normalized version)
    filename = f"{base_filename}_00_original.wav"
    filepath = output_path / filename
    sf.write(filepath, audio, sr)
    augmented_files.append(str(filepath))
    
    # 2-5. Time stretching (4 versions)
    for i, rate in enumerate([0.9, 0.95, 1.05, 1.1], start=1):
        y_stretched = librosa.effects.time_stretch(audio, rate=rate)
        filename = f"{base_filename}_{i:02d}_stretch_{rate}.wav"
        filepath = output_path / filename
        sf.write(filepath, y_stretched, sr)
        augmented_files.append(str(filepath))
    
    # 6-9. Pitch shifting (4 versions)
    for i, n_steps in enumerate([-2, -1, 1, 2], start=5):
        y_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        filename = f"{base_filename}_{i:02d}_pitch_{n_steps:+d}.wav"
        filepath = output_path / filename
        sf.write(filepath, y_shifted, sr)
        augmented_files.append(str(filepath))
    
    # 10-13. White noise (4 versions)
    for i, snr in enumerate([15, 20, 25, 30], start=9):
        y_noisy = add_white_noise(audio, snr_db=snr)
        y_noisy = np.clip(y_noisy, -1.0, 1.0)
        filename = f"{base_filename}_{i:02d}_wnoise_{snr}db.wav"
        filepath = output_path / filename
        sf.write(filepath, y_noisy, sr)
        augmented_files.append(str(filepath))
    
    # 14-15. Pink noise (2 versions)
    for i, snr in enumerate([20, 25], start=13):
        y_noisy = add_pink_noise(audio, snr_db=snr)
        y_noisy = np.clip(y_noisy, -1.0, 1.0)
        filename = f"{base_filename}_{i:02d}_pnoise_{snr}db.wav"
        filepath = output_path / filename
        sf.write(filepath, y_noisy, sr)
        augmented_files.append(str(filepath))
    
    # 16-17. Volume changes (2 versions)
    for i, gain in enumerate([0.7, 1.3], start=15):
        y_gained = audio * gain
        y_gained = np.clip(y_gained, -1.0, 1.0)
        filename = f"{base_filename}_{i:02d}_volume_{gain}.wav"
        filepath = output_path / filename
        sf.write(filepath, y_gained, sr)
        augmented_files.append(str(filepath))
    
    # 18. Combo: Stretch + White Noise
    y_combo = librosa.effects.time_stretch(audio, rate=0.95)
    y_combo = add_white_noise(y_combo, snr_db=20)
    y_combo = np.clip(y_combo, -1.0, 1.0)
    filename = f"{base_filename}_17_combo_stretch_noise.wav"
    filepath = output_path / filename
    sf.write(filepath, y_combo, sr)
    augmented_files.append(str(filepath))
    
    # 19. Combo: Pitch + Volume
    y_combo = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
    y_combo = y_combo * 0.8
    y_combo = np.clip(y_combo, -1.0, 1.0)
    filename = f"{base_filename}_18_combo_pitch_volume.wav"
    filepath = output_path / filename
    sf.write(filepath, y_combo, sr)
    augmented_files.append(str(filepath))
    
    # 20. Combo: Stretch + Pitch + Pink Noise
    y_combo = librosa.effects.time_stretch(audio, rate=1.05)
    y_combo = librosa.effects.pitch_shift(y_combo, sr=sr, n_steps=-1)
    y_combo = add_pink_noise(y_combo, snr_db=22)
    y_combo = np.clip(y_combo, -1.0, 1.0)
    filename = f"{base_filename}_19_combo_all.wav"
    filepath = output_path / filename
    sf.write(filepath, y_combo, sr)
    augmented_files.append(str(filepath))
    
    return augmented_files

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def step1_scan_and_validate():
    """
    Step 1: Scan dataset and validate all files
    """
    print("\n" + "="*70)
    print("STEP 1: SCANNING AND VALIDATING FILES")
    print("="*70)
    
    # Find all files
    all_files = find_all_audio_files(CONFIG['original_dir'])
    
    if len(all_files) == 0:
        print("\n❌ ERROR: No audio files found!")
        print(f"   Check that '{CONFIG['original_dir']}' is the correct path")
        return None
    
    print(f"\n📊 Found {len(all_files)} total audio files")
    
    # Count by class
    df = pd.DataFrame(all_files)
    print("\nFiles per class:")
    print(df.groupby('class').size())
    
    print("\nFiles per class/subfolder:")
    print(df.groupby(['class', 'subfolder']).size())
    
    # Validate each file
    print("\n🔍 Validating audio quality...")
    validation_results = []
    
    for file_info in tqdm(all_files, desc="Validating"):
        y, sr, is_valid, issues = load_and_validate_audio(
            file_info['filepath'],
            target_sr=CONFIG['target_sr']
        )
        
        validation_results.append({
            **file_info,
            'duration': len(y) / sr if y is not None else 0,
            'sample_rate': sr,
            'is_valid': is_valid,
            'issues': '; '.join(issues) if issues else 'None'
        })
    
    val_df = pd.DataFrame(validation_results)
    
    # Save validation report
    report_path = Path(CONFIG['reports_dir']) / 'validation_report.csv'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(report_path, index=False)
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"   Total files: {len(val_df)}")
    print(f"   Valid files: {val_df['is_valid'].sum()}")
    print(f"   Invalid files: {(~val_df['is_valid']).sum()}")
    
    if (~val_df['is_valid']).sum() > 0:
        print(f"\n⚠️  Issues found:")
        issues_df = val_df[~val_df['is_valid']][['class', 'subfolder', 'filename', 'issues']]
        print(issues_df.to_string(index=False))
    
    print(f"\n✅ Validation report saved to: {report_path}")
    
    return val_df

def step2_split_files(val_df):
    """
    Step 2: Split files by UNIQUE_ID (LEAK-FREE)
    """
    print("\n" + "="*70)
    print("STEP 2: SPLITTING FILES (LEAK-FREE)")
    print("="*70)
    
    # Filter valid files only
    valid_df = val_df[val_df['is_valid']].copy()
    
    split_info = {
        'train': [],
        'val': [],
        'test': []
    }
    
    classes = valid_df['class'].unique()
    
    for class_name in classes:
        class_df = valid_df[valid_df['class'] == class_name]
        
        # Get list of file info dicts
        class_files = class_df.to_dict('records')
        
        print(f"\n📁 Class: {class_name}")
        print(f"   Total valid files: {len(class_files)}")
        
        # Split by unique_id (this ensures no leakage)
        train_files, temp_files = train_test_split(
            class_files,
            test_size=(CONFIG['test_size'] + CONFIG['val_size']),
            random_state=CONFIG['random_seed']
        )
        
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(CONFIG['test_size'] / (CONFIG['test_size'] + CONFIG['val_size'])),
            random_state=CONFIG['random_seed']
        )
        
        split_info['train'].extend(train_files)
        split_info['val'].extend(val_files)
        split_info['test'].extend(test_files)
        
        print(f"   Train: {len(train_files)} files ({len(train_files)/len(class_files)*100:.1f}%)")
        print(f"   Val:   {len(val_files)} files ({len(val_files)/len(class_files)*100:.1f}%)")
        print(f"   Test:  {len(test_files)} files ({len(test_files)/len(class_files)*100:.1f}%)")
        
        # Verify no overlap by unique_id
        train_ids = {f['unique_id'] for f in train_files}
        val_ids = {f['unique_id'] for f in val_files}
        test_ids = {f['unique_id'] for f in test_files}
        
        assert len(train_ids & val_ids) == 0, f"Train-Val overlap in {class_name}!"
        assert len(train_ids & test_ids) == 0, f"Train-Test overlap in {class_name}!"
        assert len(val_ids & test_ids) == 0, f"Val-Test overlap in {class_name}!"
        
        print(f"   ✅ No overlap verified")
    
    # Save split information
    split_info_path = Path(CONFIG['reports_dir']) / 'split_info.json'
    
    # Convert to serializable format
    split_info_serializable = {}
    for split, files in split_info.items():
        split_info_serializable[split] = files
    
    with open(split_info_path, 'w') as f:
        json.dump(split_info_serializable, f, indent=2)
    
    print(f"\n✅ Split information saved to: {split_info_path}")
    
    return split_info

def step3_augment_splits(split_info):
    """
    Step 3: Augment each split separately
    """
    print("\n" + "="*70)
    print("STEP 3: AUGMENTING SPLITS SEPARATELY")
    print("="*70)
    
    augmentation_log = []
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*70}")
        
        files = split_info[split_name]
        
        if len(files) == 0:
            continue
        
        # Group by class
        class_files = {}
        for file_info in files:
            class_name = file_info['class']
            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append(file_info)
        
        for class_name, files_list in class_files.items():
            print(f"\n📁 {split_name}/{class_name}")
            print(f"   Original files: {len(files_list)}")
            print(f"   Target augmented files: {len(files_list) * CONFIG['augmentations_per_file']}")
            
            output_dir = Path(CONFIG['processed_dir']) / split_name / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            total_augmented = 0
            
            for file_info in tqdm(files_list, desc=f"   Augmenting"):
                # Load and validate
                y, sr, is_valid, issues = load_and_validate_audio(
                    file_info['filepath'],
                    target_sr=CONFIG['target_sr']
                )
                
                if not is_valid:
                    print(f"   ⚠️  Skipping {file_info['filename']}: {issues}")
                    continue
                
                # Augment using unique_id as base filename
                base_filename = file_info['unique_id']
                augmented_files = augment_single_file(
                    y, sr, output_dir, base_filename
                )
                
                total_augmented += len(augmented_files)
                
                # Log
                for aug_file in augmented_files:
                    aug_filename = Path(aug_file).name
                    
                    # Extract augmentation type from filename
                    parts = aug_filename.split('_')
                    if len(parts) >= 4:
                        aug_type = '_'.join(parts[-2:]).replace('.wav', '')
                    else:
                        aug_type = 'original'
                    
                    augmentation_log.append({
                        'split': split_name,
                        'class': class_name,
                        'original_file': file_info['filename'],
                        'original_unique_id': file_info['unique_id'],
                        'subfolder': file_info['subfolder'],
                        'augmented_file': aug_filename,
                        'augmentation_type': aug_type
                    })
            
            print(f"   ✅ Created {total_augmented} augmented files")
    
    # Save augmentation log
    log_df = pd.DataFrame(augmentation_log)
    log_path = Path(CONFIG['reports_dir']) / 'augmentation_log.csv'
    log_df.to_csv(log_path, index=False)
    
    print(f"\n✅ Augmentation log saved to: {log_path}")
    
    return log_df

def step4_generate_reports(val_df, augmentation_log):
    """
    Step 4: Generate comprehensive reports and visualizations
    """
    print("\n" + "="*70)
    print("STEP 4: GENERATING REPORTS & VISUALIZATIONS")
    print("="*70)
    
    # Count files in processed directory
    counts = {}
    for split in ['train', 'val', 'test']:
        counts[split] = {}
        for class_name in ['bearing', 'propeller', 'healthy']:
            class_dir = Path(CONFIG['processed_dir']) / split / class_name
            if class_dir.exists():
                counts[split][class_name] = len(list(class_dir.glob('*.wav')))
            else:
                counts[split][class_name] = 0
    
    # Create summary DataFrame
    summary_data = []
    for split in ['train', 'val', 'test']:
        for class_name in ['bearing', 'propeller', 'healthy']:
            summary_data.append({
                'Split': split,
                'Class': class_name,
                'Count': counts[split][class_name]
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = Path(CONFIG['reports_dir']) / 'dataset_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Dataset summary saved to: {summary_path}")
    
    # Print summary table
    print(f"\n📊 DATASET SUMMARY")
    pivot_table = summary_df.pivot(index='Class', columns='Split', values='Count')
    print(pivot_table)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Files per split
    ax = axes[0, 0]
    split_totals = summary_df.groupby('Split')['Count'].sum()
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    split_totals.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Total Files per Split', fontsize=14, fontweight='bold')
    ax.set_xlabel('Split')
    ax.set_ylabel('Number of Files')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Plot 2: Files per class per split
    ax = axes[0, 1]
    summary_pivot = summary_df.pivot(index='Class', columns='Split', values='Count')
    summary_pivot.plot(kind='bar', ax=ax)
    ax.set_title('Files per Class per Split', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Files')
    ax.legend(title='Split')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Plot 3: Original vs Augmented
    ax = axes[1, 0]
    original_counts = val_df[val_df['is_valid']].groupby('class').size()
    augmented_counts = summary_df.groupby('Class')['Count'].sum()
    
    comparison = pd.DataFrame({
        'Original': original_counts,
        'Augmented': augmented_counts
    })
    comparison.plot(kind='bar', ax=ax, color=['#95a5a6', '#2ecc71'])
    ax.set_title('Original vs Augmented Files', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Files')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Plot 4: Augmentation type distribution
    ax = axes[1, 1]
    aug_types = augmentation_log['augmentation_type'].value_counts().head(10)
    aug_types.plot(kind='barh', ax=ax, color='#9b59b6')
    ax.set_title('Top 10 Augmentation Types', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_ylabel('Augmentation Type')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    viz_path = Path(CONFIG['reports_dir']) / 'phase1_summary.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {viz_path}")
    
    plt.close()
    
    # Final statistics
    print(f"\n📊 FINAL STATISTICS")
    print(f"{'='*70}")
    print(f"Total original valid files: {val_df['is_valid'].sum()}")
    print(f"Total augmented files: {summary_df['Count'].sum()}")
    print(f"Augmentation factor: {summary_df['Count'].sum() / val_df['is_valid'].sum():.1f}x")
    print(f"\nSplit distribution:")
    for split in ['train', 'val', 'test']:
        total = summary_df[summary_df['Split'] == split]['Count'].sum()
        percentage = total / summary_df['Count'].sum() * 100
        print(f"  {split.capitalize():5s}: {total:4d} files ({percentage:5.1f}%)")
    
    # Class distribution within each split
    print(f"\nClass distribution per split:")
    for split in ['train', 'val', 'test']:
        print(f"\n  {split.capitalize()}:")
        split_data = summary_df[summary_df['Split'] == split]
        for _, row in split_data.iterrows():
            print(f"    {row['Class']:10s}: {row['Count']:4d} files")
    
    return summary_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("EV ACOUSTIC DIAGNOSTICS - PHASE 1: DATA PROCESSING")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ----------------------------------------------------------------------
    # Setup logging
    # ----------------------------------------------------------------------
    log_file = setup_logging()

    # ----------------------------------------------------------------------
    # Ensure required directories exist
    # ----------------------------------------------------------------------
    for dir_path in [
        CONFIG['processed_dir'],
        CONFIG['reports_dir'],
        CONFIG['logs_dir']
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    try:
        # ------------------------------------------------------------------
        # STEP 1: Scan and validate original dataset
        # ------------------------------------------------------------------
        validation_df = step1_scan_and_validate()
        if validation_df is None or validation_df.empty:
            raise RuntimeError("No valid audio files found after validation.")

        # ------------------------------------------------------------------
        # STEP 2: Leak-free train / val / test split
        # ------------------------------------------------------------------
        split_info = step2_split_files(validation_df)

        # ------------------------------------------------------------------
        # STEP 3: Augment each split independently
        # ------------------------------------------------------------------
        augmentation_log = step3_augment_splits(split_info)

        # ------------------------------------------------------------------
        # STEP 4: Generate reports and visualizations
        # ------------------------------------------------------------------
        summary_df = step4_generate_reports(validation_df, augmentation_log)

        # ------------------------------------------------------------------
        # Completion message
        # ------------------------------------------------------------------
        print("\n" + "="*70)
        print("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n📁 Generated outputs:")
        print(f"  • Processed data: {CONFIG['processed_dir']}/")
        print(f"  • Reports:        {CONFIG['reports_dir']}/")
        print(f"  • Logs:           {CONFIG['logs_dir']}/")
        print(f"  • Log file:       {log_file}")

        print("\n🎯 NEXT STEPS:")
        print("  1. Review validation_report.csv")
        print("  2. Inspect augmentation_log.csv")
        print("  3. Listen to a few augmented samples")
        print("  4. Proceed to PHASE 2: MODEL TRAINING")

        return True

    except Exception as e:
        print("\n" + "="*70)
        print("❌ PHASE 1 FAILED")
        print("="*70)
        print(f"Error: {str(e)}")

        import traceback
        traceback.print_exc()

        print("\n🔍 DEBUGGING TIPS:")
        print("  • Check dataset path in CONFIG['original_dir']")
        print("  • Verify folder names match folder_to_class mapping")
        print("  • Ensure audio files are readable by librosa")

        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
