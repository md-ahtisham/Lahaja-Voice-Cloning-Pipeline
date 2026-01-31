"""
Advanced Usage & Extension Guide for Lahaja Bengali Voice Cloning
Includes: Multi-language support, batch processing, quality metrics
"""

# ============================================================
# EXAMPLE 1: Generate Multiple Languages Sequentially
# ============================================================

def generate_multiple_languages():
    """
    Generate L2 datasets for multiple Indian languages
    """
    from lahaja_bengali_voice_clone import Config, load_and_filter_lahaja, load_tts_model, generate_l2_dataset, save_l2_dataset
    
    languages = {
        'Bengali': ('bn', 'बंगाली'),
        'Marathi': ('mr', 'मराठी'),
        'Tamil': ('ta', 'தமிழ்'),
        'Telugu': ('te', 'తెలుగు'),
        'Kannada': ('kn', 'ಕನ್ನಡ'),
    }
    
    # Load dataset once
    df_bengali = load_and_filter_lahaja()
    tts_model = load_tts_model()
    
    for lang_name, (lang_code, script_name) in languages.items():
        print(f"\n{'='*60}")
        print(f"🎯 Generating {lang_name} (Code: {lang_code})")
        print(f"{'='*60}")
        
        # Update config
        Config.TARGET_LANGUAGE = lang_name
        Config.TARGET_LANG_CODE = lang_code
        Config.OUTPUT_DIR = Path(f"./lahaja_{lang_name.lower()}_cloned")
        
        try:
            # Generate dataset
            audio_samples, metadata_df = generate_l2_dataset(
                tts_model,
                df_bengali,
                n_samples=50
            )
            
            # Save dataset
            save_l2_dataset(audio_samples, metadata_df)
            print(f"✅ {lang_name} dataset saved successfully!")
            
        except Exception as e:
            print(f"❌ Error generating {lang_name}: {e}")
            continue

# ============================================================
# EXAMPLE 2: Batch Processing with Custom Texts
# ============================================================

def generate_with_custom_texts(custom_texts_file: str):
    """
    Generate voice clones using custom text file
    
    Expected format:
    - One sentence per line
    - UTF-8 encoded
    - Bengali script
    """
    from lahaja_bengali_voice_clone import Config, load_tts_model
    
    # Load custom texts
    with open(custom_texts_file, 'r', encoding='utf-8') as f:
        custom_texts = [line.strip() for line in f if line.strip()]
    
    print(f"📖 Loaded {len(custom_texts)} custom texts from {custom_texts_file}")
    
    # Replace in Config
    Config.BENGALI_SENTENCES = custom_texts
    Config.N_SAMPLES = len(custom_texts)
    
    # Now run pipeline
    from lahaja_bengali_voice_clone import main
    main()

# Usage:
# 1. Create file: my_bengali_texts.txt with Bengali sentences
# 2. Call: generate_with_custom_texts("my_bengali_texts.txt")

# ============================================================
# EXAMPLE 3: Quality Metrics & Analysis
# ============================================================

def analyze_generated_dataset(metadata_file: str):
    """
    Analyze generated L2 dataset for quality metrics
    """
    import json
    import pandas as pd
    from pathlib import Path
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    df = pd.DataFrame(metadata)
    
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)
    
    # Duration statistics
    print(f"\n⏱️  DURATION STATISTICS:")
    print(f"   Total: {df['duration'].sum():.2f}s ({df['duration'].sum()/60:.2f}min)")
    print(f"   Average: {df['duration'].mean():.2f}s")
    print(f"   Min: {df['duration'].min():.2f}s")
    print(f"   Max: {df['duration'].max():.2f}s")
    print(f"   Std Dev: {df['duration'].std():.2f}s")
    
    # Speaker diversity
    print(f"\n🎤 SPEAKER DIVERSITY:")
    print(f"   Total samples: {len(df)}")
    print(f"   Unique speakers: {df['sp_id'].nunique()}")
    print(f"   Samples per speaker: {len(df) / df['sp_id'].nunique():.2f}")
    
    # Geographic diversity
    print(f"\n🗺️  GEOGRAPHIC DIVERSITY:")
    print(f"   States: {df['native_state'].nunique()}")
    print(f"   Districts: {df['native_district'].nunique()}")
    print(f"   Top 5 districts:")
    for dist, count in df['native_district'].value_counts().head(5).items():
        print(f"      {dist}: {count}")
    
    # Demographic balance
    print(f"\n👥 DEMOGRAPHIC DISTRIBUTION:")
    print(f"   Gender:")
    for gender, count in df['gender'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"      {gender}: {count} ({pct:.1f}%)")
    
    print(f"   Age groups:")
    for age, count in df['age_group'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"      {age}: {count} ({pct:.1f}%)")
    
    # Text statistics
    print(f"\n📝 TEXT STATISTICS:")
    df['text_length'] = df['text'].str.len()
    print(f"   Average characters: {df['text_length'].mean():.1f}")
    print(f"   Average words (approx): {(df['text_length'] / 4).mean():.1f}")
    
    print("\n" + "="*60)
    
    return df

# Usage:
# df_analysis = analyze_generated_dataset("lahaja_bengali_cloned/metadata.json")

# ============================================================
# EXAMPLE 4: Audio Quality Verification
# ============================================================

def verify_audio_quality(audio_dir: str):
    """
    Verify generated audio files for quality issues
    """
    import librosa
    import numpy as np
    from pathlib import Path
    
    audio_dir = Path(audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav"))
    
    print(f"\n🔍 Verifying {len(audio_files)} audio files...")
    print("="*60)
    
    issues = []
    stats = {
        'sample_rate': [],
        'duration': [],
        'rms_energy': [],
        'peak_amplitude': [],
    }
    
    for i, audio_file in enumerate(audio_files):
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=None)
            
            # Check sampling rate
            if sr != 16000:
                issues.append(f"⚠️  {audio_file.name}: Wrong SR ({sr} Hz, expected 16000)")
            
            # Compute metrics
            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.sqrt(np.mean(y**2))
            peak = np.max(np.abs(y))
            
            stats['sample_rate'].append(sr)
            stats['duration'].append(duration)
            stats['rms_energy'].append(rms)
            stats['peak_amplitude'].append(peak)
            
            # Check for common issues
            if duration < 0.5:
                issues.append(f"⚠️  {audio_file.name}: Very short ({duration:.2f}s)")
            if duration > 30:
                issues.append(f"⚠️  {audio_file.name}: Very long ({duration:.2f}s)")
            if rms < 0.001:
                issues.append(f"⚠️  {audio_file.name}: Possibly silent (RMS={rms:.6f})")
            if peak > 0.99:
                issues.append(f"⚠️  {audio_file.name}: Clipping detected (peak={peak:.3f})")
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i+1}/{len(audio_files)}...")
        
        except Exception as e:
            issues.append(f"❌ {audio_file.name}: {str(e)}")
    
    # Print report
    print("\n📋 QUALITY REPORT:")
    print(f"   Total files: {len(audio_files)}")
    print(f"   Issues found: {len(issues)}")
    
    if issues:
        print("\n⚠️  ISSUES:")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues)-10} more issues")
    else:
        print("   ✅ All files verified successfully!")
    
    # Print statistics
    print("\n📊 AUDIO STATISTICS:")
    print(f"   Duration: {np.mean(stats['duration']):.2f}s ± {np.std(stats['duration']):.2f}s")
    print(f"   RMS Energy: {np.mean(stats['rms_energy']):.4f} ± {np.std(stats['rms_energy']):.4f}")
    print(f"   Peak Amplitude: {np.mean(stats['peak_amplitude']):.3f} ± {np.std(stats['peak_amplitude']):.3f}")

# Usage:
# verify_audio_quality("lahaja_bengali_cloned/audio")

# ============================================================
# EXAMPLE 5: Create Train/Val/Test Splits
# ============================================================

def create_dataset_splits(metadata_file: str, train_ratio=0.7, val_ratio=0.15):
    """
    Split generated dataset into train/validation/test sets
    """
    import json
    import pandas as pd
    from pathlib import Path
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    df = pd.DataFrame(metadata)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate splits
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Create splits
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Save splits
    base_dir = Path(metadata_file).parent
    
    train_df.to_json(base_dir / "train_manifest.jsonl", orient='records', lines=True)
    val_df.to_json(base_dir / "val_manifest.jsonl", orient='records', lines=True)
    test_df.to_json(base_dir / "test_manifest.jsonl", orient='records', lines=True)
    
    print(f"\n✅ Dataset splits created:")
    print(f"   Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    print(f"   Val:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    print(f"   Test:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df

# Usage:
# train, val, test = create_dataset_splits("lahaja_bengali_cloned/metadata.json")

# ============================================================
# EXAMPLE 6: Convert to HuggingFace Dataset Format
# ============================================================

def export_to_huggingface_format(metadata_file: str, audio_dir: str):
    """
    Convert to HuggingFace Datasets format for easy integration
    """
    import json
    import pandas as pd
    from datasets import Dataset, Audio
    from pathlib import Path
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    df = pd.DataFrame(metadata)
    
    # Add full paths
    audio_dir = Path(audio_dir)
    df['audio'] = df['audio_file'].apply(lambda x: str(audio_dir / x))
    
    # Create HF dataset
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Save to disk (can be pushed to HuggingFace Hub)
    hf_dataset.save_to_disk("lahaja_bengali_hf")
    
    print("✅ Exported to HuggingFace format: lahaja_bengali_hf/")
    print("   Ready for fine-tuning ASR/TTS models!")
    
    return hf_dataset

# Usage:
# hf_ds = export_to_huggingface_format(
#     "lahaja_bengali_cloned/metadata.json",
#     "lahaja_bengali_cloned/audio"
# )

# ============================================================
# EXAMPLE 7: Upload to HuggingFace Hub
# ============================================================

def upload_dataset_to_hub(local_path: str, repo_id: str):
    """
    Upload generated dataset to HuggingFace Hub
    
    Args:
        local_path: Path to lahaja_bengali_cloned directory
        repo_id: HuggingFace repo (format: "username/dataset-name")
    """
    from datasets import load_from_disk
    from huggingface_hub import HfApi
    
    print(f"📤 Loading dataset from {local_path}...")
    dataset = load_from_disk(local_path)
    
    print(f"📤 Uploading to {repo_id}...")
    dataset.push_to_hub(repo_id, private=False)
    
    print(f"✅ Dataset uploaded!")
    print(f"   URL: https://huggingface.co/datasets/{repo_id}")

# Usage:
# upload_dataset_to_hub("lahaja_bengali_hf", "your-username/lahaja-bengali")

# ============================================================
# EXAMPLE 8: Fine-tune Bengali ASR Model
# ============================================================

def finetune_bengali_asr():
    """
    Template for fine-tuning Bengali ASR model on generated data
    """
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    from datasets import load_from_disk
    import torch
    
    # Load generated dataset
    dataset = load_from_disk("lahaja_bengali_hf")
    
    # Split
    split_dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Load pre-trained Bengali model
    model_id = "ai4bharat/indicwav2vec2_v1_bengali"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    
    print("✅ Model and processor loaded")
    print("📝 Ready for fine-tuning with generated Bengali data!")
    
    # Training setup (pseudocode)
    # from transformers import TrainingArguments, Trainer
    # args = TrainingArguments(
    #     output_dir="./bengali-asr-finetuned",
    #     per_device_train_batch_size=16,
    #     learning_rate=1e-5,
    #     num_train_epochs=3,
    # )
    # trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    # trainer.train()

# ============================================================
# EXAMPLE 9: Compute Audio Similarity Metrics
# ============================================================

def compute_speaker_similarity(audio1_path: str, audio2_path: str):
    """
    Compute similarity between two audio files
    Useful for verifying voice cloning quality
    """
    import librosa
    import numpy as np
    from scipy.spatial.distance import cosine
    
    # Load both audios
    y1, sr1 = librosa.load(audio1_path, sr=16000)
    y2, sr2 = librosa.load(audio2_path, sr=16000)
    
    # Extract speaker characteristics
    # Method 1: MFCCs (Mel-Frequency Cepstral Coefficients)
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
    
    mfcc1_mean = np.mean(mfcc1, axis=1)
    mfcc2_mean = np.mean(mfcc2, axis=1)
    
    # Cosine similarity (0 = completely different, 1 = identical)
    similarity = 1 - cosine(mfcc1_mean, mfcc2_mean)
    
    print(f"\n🎵 AUDIO SIMILARITY ANALYSIS")
    print(f"   Audio1: {audio1_path}")
    print(f"   Audio2: {audio2_path}")
    print(f"   Similarity Score: {similarity:.3f} (0-1 scale)")
    print(f"   Interpretation: {'🟢 Very similar' if similarity > 0.8 else '🟡 Moderately similar' if similarity > 0.5 else '🔴 Quite different'}")
    
    return similarity

# Usage:
# sim = compute_speaker_similarity("original_speaker.wav", "cloned_speaker.wav")

# ============================================================
# EXAMPLE 10: Generate Evaluation Report
# ============================================================

def generate_evaluation_report(output_dir: str):
    """
    Generate comprehensive evaluation report
    """
    from pathlib import Path
    import json
    import datetime
    
    output_dir = Path(output_dir)
    
    report = {
        "generated_date": datetime.datetime.now().isoformat(),
        "dataset_path": str(output_dir),
        "components": {
            "audio_files": len(list(output_dir.glob("audio/*.wav"))),
            "metadata": (output_dir / "metadata.json").exists(),
            "manifest": (output_dir / "manifest.jsonl").exists(),
            "summary": (output_dir / "dataset_summary.json").exists(),
        }
    }
    
    # Load and analyze
    with open(output_dir / "dataset_summary.json", 'r') as f:
        summary = json.load(f)
    
    report.update(summary)
    
    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Evaluation report saved: {report_path}")
    
    return report

# ============================================================
# MAIN: Run Examples
# ============================================================

if __name__ == "__main__":
    # Run analysis on generated dataset
    print("🚀 Running Advanced Examples...\n")
    
    # Example: Analyze dataset
    try:
        df = analyze_generated_dataset("lahaja_bengali_cloned/metadata.json")
        print("✅ Analysis complete!")
    except FileNotFoundError:
        print("⚠️  Dataset not found. Run main pipeline first:")
        print("   python lahaja_bengali_voice_clone.py")
    
    # Example: Verify audio quality
    try:
        verify_audio_quality("lahaja_bengali_cloned/audio")
        print("✅ Quality verification complete!")
    except FileNotFoundError:
        print("⚠️  Audio directory not found.")
    
    # Example: Create splits
    try:
        train, val, test = create_dataset_splits("lahaja_bengali_cloned/metadata.json")
        print("✅ Dataset splits created!")
    except FileNotFoundError:
        print("⚠️  Metadata not found.")
