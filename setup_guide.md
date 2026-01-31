# Lahaja Bengali Voice Cloning - Setup Guide

## 🎯 Overview
This pipeline clones the voices of Hindi speakers from the Lahaja dataset to synthesize Bengali text, creating a new L2 dataset with 50 Bengali audio samples at 16kHz sampling rate.

### What it does:
- Loads Lahaja dataset (132 Hindi speakers, 83 districts)
- Filters Bengali native speakers
- Generates random 1-2 line Bengali sentences
- Uses IndicF5 (AI4Bharat) for voice cloning and TTS
- Outputs Bengali audio files with metadata

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- ~8-16 GB RAM
- ~20 GB disk space for models

### Step 1: Clone/Download the Script
```bash
# Save the main script
# File: lahaja_bengali_voice_clone.py
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv lahaja_env
source lahaja_env/bin/activate  # On Windows: lahaja_env\Scripts\activate

# Or using conda
conda create -n lahaja_voice python=3.10 -y
conda activate lahaja_voice
```

### Step 3: Install Dependencies

**Option A: Using requirements.txt (Recommended)**
```bash
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q datasets transformers librosa soundfile scipy numpy pandas
pip install -q huggingface-hub  # For authentication
```

**Option B: Manual Installation (if above fails)**
```bash
# Core ML libraries
pip install torch==2.0.0 torchvision torchaudio

# HuggingFace ecosystem
pip install datasets transformers==4.36.0 huggingface-hub

# Audio processing
pip install librosa soundfile scipy

# Data processing
pip install numpy pandas

# Optional: for GPU optimization
pip install accelerate
```

### Step 4: HuggingFace Authentication
```bash
# Login to HuggingFace Hub (required for model access)
huggingface-cli login
# Paste your HF token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

---

## 🚀 Quick Start

### Basic Usage
```bash
# Navigate to script directory
cd path/to/script

# Run the pipeline
python lahaja_bengali_voice_clone.py
```

### Output Structure
```
./lahaja_bengali_cloned/
├── audio/                          # 50 Bengali audio files (16kHz WAV)
│   ├── sp_id_001_000.wav
│   ├── sp_id_002_001.wav
│   └── ...
├── metadata.json                   # Complete metadata (JSON)
├── metadata.csv                    # Metadata spreadsheet (CSV)
├── manifest.jsonl                  # One-sample-per-line manifest
└── dataset_summary.json            # Dataset statistics
```

---

## 📋 Configuration Options

Edit the `Config` class in the script to customize:

```python
class Config:
    # Change target language (once supported)
    TARGET_LANGUAGE = "Bengali"      # Change to "Marathi", "Tamil", etc.
    TARGET_LANG_CODE = "bn"          # Language code
    
    # Number of samples to generate
    N_SAMPLES = 50                   # Change to 100, 200, etc.
    
    # Sampling rate (16000 Hz = 16kHz)
    SAMPLING_RATE = 16000
    
    # Output directory
    OUTPUT_DIR = Path("./lahaja_bengali_cloned")
    
    # Device (auto-detects GPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 🔊 Understanding the Output

### 1. Audio Files (WAV format)
- **Format**: PCM 16-bit mono
- **Sampling Rate**: 16,000 Hz (16 kHz)
- **Duration**: 0.5-5 seconds per file
- **Naming**: `{speaker_id}_{sample_index}.wav`

### 2. Metadata JSON
```json
[
  {
    "sample_id": "sp_id_001_000",
    "audio_file": "sp_id_001_000.wav",
    "text": "আমার নাম রহিম। আমি একজন শিক্ষক।",
    "lang": "bn",
    "sp_id": "sp_id_001",
    "native_language": "Bengali",
    "gender": "M",
    "age_group": "20-30",
    "native_state": "West Bengal",
    "native_district": "Kolkata",
    "duration": 2.45
  }
]
```

### 3. Dataset Summary
```json
{
  "dataset_name": "Lahaja-Bengali-Cloned-L2",
  "total_samples": 50,
  "total_speakers": 45,
  "target_language": "Bengali",
  "sampling_rate": 16000,
  "total_duration_seconds": 125.4,
  "avg_duration_seconds": 2.51,
  "districts_covered": 25,
  "states_covered": 8
}
```

---

## 💻 System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB
- Time: ~30 minutes for 50 samples

### Recommended Setup
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 or better
- Storage: 40 GB
- Time: ~10 minutes for 50 samples (with GPU)

---

## 🐛 Troubleshooting

### Issue 1: "CUDA out of memory"
**Solution**: 
```python
Config.DEVICE = "cpu"  # Force CPU usage
# Or reduce batch processing
```

### Issue 2: "Model not found on HuggingFace"
**Solution**:
```bash
# Check internet connection
# Login to HuggingFace
huggingface-cli login

# Or use alternative model
Config.TTS_MODEL_ID = "facebook/mms-tts-ben"
```

### Issue 3: "ModuleNotFoundError: No module named 'transformers'"
**Solution**:
```bash
pip install --upgrade transformers
```

### Issue 4: "Audio files not generating"
**Solution**:
```bash
# Check if model is downloading
# Monitor network in first run (~5-10 minutes)

# Verify Bengali text is not corrupted
# Test with sample code:
text = "আমার নাম রহিম"
print(text)  # Should display in Bengali
```

### Issue 5: "Permission denied" on output directory
**Solution**:
```bash
# Change output directory
Config.OUTPUT_DIR = Path("/home/user/my_output")

# Or run with sudo
sudo python lahaja_bengali_voice_clone.py
```

---

## 📊 Dataset Structure Details

### L1 (Input - Lahaja)
- **Language**: Hindi only
- **Speakers**: 132 unique (native language varies)
- **Districts**: 83
- **Duration**: 12.5 hours total
- **Format**: 16 kHz PCM WAV

### L2 (Output - Bengali Cloned)
- **Language**: Bengali
- **Speakers**: 50 (selected from Lahaja speakers)
- **Districts**: Up to 83 (depends on random selection)
- **Samples**: 50 audio files
- **Text**: Bengali sentences (not Hindi)
- **Format**: 16 kHz PCM WAV

---

## 🔄 Extending to Other Languages

To generate voice clones for other Indian languages (Marathi, Tamil, Telugu, etc.):

```python
class Config:
    # Change these values
    TARGET_LANGUAGE = "Marathi"
    TARGET_LANG_CODE = "mr"
    
    # Update Bengali sentences corpus to target language
    MARATHI_SENTENCES = [
        "माझे नाव रहिम आहे।",
        "आज हवामान खूप सुंदर आहे।",
        # ... add more Marathi sentences
    ]
    
    # Map language code in pipeline
    # Note: IndicF5 supports: bn, gu, hi, kn, ml, mr, od, pa, ta, te, as
```

---

## 📚 Dataset Usage Examples

### 1. Load and Explore
```python
import json
import pandas as pd

# Load metadata
with open('lahaja_bengali_cloned/metadata.json', 'r') as f:
    metadata = json.load(f)

df = pd.DataFrame(metadata)
print(f"Total samples: {len(df)}")
print(f"Speakers: {df['sp_id'].nunique()}")
print(f"Average duration: {df['duration'].mean():.2f}s")
```

### 2. Load Audio Files
```python
import librosa
import soundfile as sf

# Load single audio
audio, sr = librosa.load('lahaja_bengali_cloned/audio/sp_id_001_000.wav', sr=16000)
print(f"Shape: {audio.shape}, SR: {sr}")

# Load all audio files
import glob
audio_files = glob.glob('lahaja_bengali_cloned/audio/*.wav')
audios = [librosa.load(f, sr=16000)[0] for f in audio_files]
```

### 3. Train ASR Model
```python
from datasets import load_dataset, Audio
import soundfile as sf

# Convert to HuggingFace dataset format
audio_dataset = load_dataset('json', data_files='lahaja_bengali_cloned/manifest.jsonl')
audio_dataset = audio_dataset.cast_column('audio', Audio(sampling_rate=16000))

# Now you can fine-tune Bengali ASR models
```

---

## 📖 Reference Links

- **Lahaja Dataset**: https://huggingface.co/datasets/ai4bharat/Lahaja
- **IndicF5 Model**: https://huggingface.co/ai4bharat/IndicF5
- **AI4Bharat**: https://ai4bharat.iitm.ac.in/
- **Transformers Docs**: https://huggingface.co/docs/transformers/
- **Librosa Docs**: https://librosa.org/doc/latest/

---

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@article{lahaja2024,
  title={LAHAJA: A Robust Multi-accent Benchmark for Evaluating Indian Language ASR},
  author={Javed, T. and others},
  journal={arXiv},
  year={2024}
}

@article{indicf5,
  title={IN-F5: IndicF5 - Near-Human Polyglot Text-to-Speech},
  author={AI4Bharat},
  year={2025}
}
```

---

## ⚙️ Advanced Configuration

### GPU Memory Optimization
```python
import torch
torch.cuda.empty_cache()

# Use mixed precision
Config.DEVICE = "cuda"
torch.set_float32_matmul_precision('medium')
```

### Parallel Processing (for multiple runs)
```bash
# Generate for multiple languages sequentially
for lang in Bengali Marathi Tamil Telugu Kannada; do
    python lahaja_bengali_voice_clone.py --lang $lang
done
```

### Using Different TTS Models
```python
# Alternative models from AI4Bharat
Config.TTS_MODEL_ID = "ai4bharat/indic-parler-tts"  # More control
Config.TTS_MODEL_ID = "facebook/mms-tts-ben"       # Lightweight
```

---

## 📞 Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review logs in `lahaja_bengali_cloned/` directory
3. Open issue on GitHub with error traceback
4. Contact AI4Bharat: https://ai4bharat.iitm.ac.in/

---

## ✅ Verification Checklist

After running the pipeline:

- [ ] `lahaja_bengali_cloned/` directory created
- [ ] 50 audio files in `audio/` subdirectory
- [ ] `metadata.json` file generated (readable)
- [ ] `metadata.csv` file created
- [ ] `dataset_summary.json` shows correct counts
- [ ] All audio files are 16 kHz PCM WAV
- [ ] Text transcriptions are in Bengali (not Hindi)
- [ ] Speaker IDs match Lahaja dataset format
- [ ] All metadata fields populated

---

**Generated**: 2026-01-31
**Version**: 1.0
**Status**: Ready for local VSCode execution ✅
