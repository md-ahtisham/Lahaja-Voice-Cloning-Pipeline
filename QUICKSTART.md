# 🚀 QUICK START - Lahaja Bengali Voice Cloning

## ⚡ 5-Minute Setup

### Step 1: Install Python Requirements (2 min)
```bash
# Open VSCode terminal (Ctrl+`)

# Create virtual environment
python -m venv lahaja_env

# Activate environment
# On Windows:
lahaja_env\Scripts\activate
# On macOS/Linux:
source lahaja_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# (If GPU available, also run:)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Get HuggingFace Token (1 min)
```bash
# 1. Go to: https://huggingface.co/settings/tokens
# 2. Click "New token" → Copy token
# 3. Run in terminal:
huggingface-cli login
# Paste your token when prompted
```

### Step 3: Run Pipeline (2 min + ~10-30 min processing)
```bash
# In VSCode terminal:
python lahaja_bengali_voice_clone.py

# Output will be in: ./lahaja_bengali_cloned/
```

---

## 📁 Expected Output Structure

```
lahaja_bengali_cloned/
├── audio/                          # 50 .wav files (16 kHz)
│   ├── sp_id_001_000.wav
│   ├── sp_id_002_001.wav
│   └── ... (48 more)
│
├── metadata.json                   # Detailed metadata
├── metadata.csv                    # Spreadsheet format
├── manifest.jsonl                  # One-per-line format
└── dataset_summary.json            # Statistics
```

---

## ✅ Verification Checklist

After running, check:

- [ ] `lahaja_bengali_cloned/` folder created
- [ ] 50 audio files in `audio/` subfolder
- [ ] All files are `.wav` format
- [ ] `metadata.json` is readable
- [ ] `dataset_summary.json` shows 50 samples
- [ ] Text is in Bengali (not Hindi)

**Command to verify:**
```bash
# Check directory structure
ls -la lahaja_bengali_cloned/

# Check audio files
ls -la lahaja_bengali_cloned/audio/ | head -5

# Check metadata
cat lahaja_bengali_cloned/dataset_summary.json | python -m json.tool | head -20
```

---

## 🔧 Troubleshooting Quick Fixes

### ❌ "ModuleNotFoundError: torch"
```bash
pip install --upgrade torch transformers datasets
```

### ❌ "CUDA out of memory"
```python
# Edit line in lahaja_bengali_voice_clone.py:
Config.DEVICE = "cpu"  # Change from "cuda"
```

### ❌ "Model not found on HuggingFace"
```bash
# Verify internet connection & token:
huggingface-cli login
huggingface-cli whoami  # Should show your username
```

### ❌ "Permission denied" saving files
```bash
# Create output directory first:
mkdir -p ./lahaja_bengali_cloned/audio
# Or run with explicit path:
Config.OUTPUT_DIR = Path("/home/user/Downloads/lahaja_bengali_cloned")
```

### ❌ "Audio files are empty/silent"
```bash
# Check if model is downloading (first run takes 5-10 min)
# Monitor network activity while running
# Increase timeout if needed
```

---

## 📊 Dataset Information

### L1 (Input - Lahaja)
- **Speakers**: 132 unique Hindi speakers
- **Districts**: 83 across India
- **Language**: Hindi (Devanagari script)
- **Duration**: 12.5 hours total

### L2 (Output - Generated)
- **Speakers**: Up to 50 (from Lahaja)
- **Language**: Bengali (Bangla script)
- **Samples**: 50 audio files
- **Sampling Rate**: 16,000 Hz (16 kHz)
- **Format**: PCM WAV (16-bit mono)
- **Text**: Bengali sentences (auto-generated)

---

## 🎯 Use Cases

### 1. ASR Model Training
```python
# Load and use for training Bengali speech recognition
from datasets import load_dataset, Audio

# After generation, convert to HF format
dataset = load_from_disk("lahaja_bengali_hf")
# Fine-tune on: facebook/wav2vec2-large-xls-r-300m
```

### 2. TTS Evaluation
```python
# Test speech synthesis quality
# Compare original Hindi voices to Bengali synthesis
```

### 3. Speaker Verification
```python
# Validate voice cloning accuracy
# Compare speaker embeddings before/after
```

### 4. Linguistic Research
```python
# Analyze Bengali speech patterns across speakers
# Study accent variation in cloned voices
```

---

## 🌐 Advanced Usage

### Generate Multiple Languages
```bash
python advanced_examples.py
# Generates: Marathi, Tamil, Telugu, Kannada, Bengali
```

### Custom Sentences
```python
# Edit Config.BENGALI_SENTENCES in the main script
# Or use: generate_with_custom_texts("my_texts.txt")
```

### Quality Analysis
```python
from advanced_examples import analyze_generated_dataset
df = analyze_generated_dataset("lahaja_bengali_cloned/metadata.json")
```

### Convert to HuggingFace Format
```python
from advanced_examples import export_to_huggingface_format
hf_ds = export_to_huggingface_format(
    "lahaja_bengali_cloned/metadata.json",
    "lahaja_bengali_cloned/audio"
)
```

---

## 📱 System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|------------|
| **Python** | 3.8 | 3.10+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 20 GB | 40 GB |
| **GPU** | Not needed | NVIDIA RTX 3060+ |
| **Time** | 30 min | 10 min (w/ GPU) |

---

## 📞 Support & Resources

**Documentation:**
- Full setup guide: `setup_guide.md`
- Advanced examples: `advanced_examples.py`
- Requirements: `requirements.txt`

**External Links:**
- Lahaja Dataset: https://huggingface.co/datasets/ai4bharat/Lahaja
- IndicF5 Model: https://huggingface.co/ai4bharat/IndicF5
- AI4Bharat: https://ai4bharat.iitm.ac.in/

**Files Included:**
- `lahaja_bengali_voice_clone.py` - Main pipeline
- `setup_guide.md` - Detailed setup guide
- `requirements.txt` - Python dependencies
- `advanced_examples.py` - Advanced usage examples
- `QUICKSTART.md` - This file

---

## 🎓 Understanding the Code

### Main Function Flow

```
main()
  ├─ load_and_filter_lahaja()
  │   └─ Filters for Bengali native speakers from Lahaja dataset
  │
  ├─ load_tts_model()
  │   └─ Downloads IndicF5 from HuggingFace Hub (~5-10 min first run)
  │
  ├─ generate_l2_dataset()
  │   ├─ Selects 50 random speakers
  │   ├─ Gets 50 random Bengali sentences
  │   └─ For each: clone voice → synthesize → save audio
  │
  └─ save_l2_dataset()
      ├─ Saves audio files (50 × .wav)
      ├─ Saves metadata (JSON, CSV, JSONL)
      └─ Generates summary statistics
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `AudioSample` | Data class holding audio + metadata |
| `Config` | Configuration settings (paths, models, language) |
| `load_and_filter_lahaja()` | Load dataset, filter by language |
| `load_tts_model()` | Load IndicF5 TTS model from HF Hub |
| `generate_l2_dataset()` | Main voice cloning loop |
| `save_l2_dataset()` | Export results |

---

## 💡 Tips & Tricks

### 💻 Running in Background
```bash
# Run and save logs
nohup python lahaja_bengali_voice_clone.py > output.log 2>&1 &

# Monitor progress
tail -f output.log
```

### 🎙️ Listen to Generated Audio
```bash
# On macOS
open lahaja_bengali_cloned/audio/*.wav

# On Linux
vlc lahaja_bengali_cloned/audio/*.wav

# On Windows
start lahaja_bengali_cloned/audio/
```

### 📊 Quick Statistics
```bash
# Count files
ls lahaja_bengali_cloned/audio/ | wc -l

# Total audio duration
ffprobe -v error -show_format -of default=noprint_wrappers=1:nokey=1:noescapes=1 lahaja_bengali_cloned/audio/*.wav | grep duration
```

### 🔄 Regenerate (Different Samples)
```python
# In Config class, change:
set_seed(42)  # Change to: set_seed(123), set_seed(456), etc.
# Each seed generates different speaker/text combinations
```

---

## 📝 Next Steps

1. **Run the pipeline** → Generates 50 Bengali audio samples
2. **Analyze results** → Run `advanced_examples.py`
3. **Train ASR model** → Use generated data for fine-tuning
4. **Share dataset** → Push to HuggingFace Hub (optional)
5. **Expand languages** → Modify for other Indian languages

---

## ✨ Citation

If you use this pipeline in research, cite:

```
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

**Last Updated:** 2026-01-31  
**Status:** ✅ Ready to Run  
**Version:** 1.0
