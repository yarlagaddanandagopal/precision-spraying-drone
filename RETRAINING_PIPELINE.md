# 🤖 Automated Retraining Pipeline

## Overview

The automated retraining pipeline allows your precision spraying drone system to:
- ✅ Automatically collect disease images from production
- ✅ Detect new/unknown disease types
- ✅ Retrain the model with new data
- ✅ Deploy updated models without stopping operation

## System Architecture

```
Live Detection (main_v2.py)
    ↓
Data Collection (data_collector.py)
    ↓
Pipeline Orchestrator (pipeline_orchestrator.py)
    ↓
Model Training (model_trainer.py)
    ↓
Validation & Deployment
```

## Files

| File | Purpose |
|------|---------|
| `main_v2.py` | Main application with integrated data collection |
| `data_collector.py` | Handles dataset collection and management |
| `model_trainer.py` | YOLO model training and validation |
| `pipeline_orchestrator.py` | Orchestrates the complete pipeline |
| `trigger_retraining.py` | Manual script to start retraining |
| `pipeline_config.json` | Configuration file for pipeline settings |

## Usage

### 1. Start Live Detection with Data Collection

```bash
python main_v2.py
```

**Controls:**
- `q` - Quit application
- `s` - Save current frame for manual review
- `p` - Show pipeline status (dataset size, spray count, etc.)
- `r` - Start retraining when collection threshold is reached

### 2. Monitor Collection Progress

Press `p` during detection to see:
- Number of images collected per disease class
- Total images collected
- Whether retraining threshold is met

### 3. Trigger Automated Retraining

**Automatic (when threshold reached):**
- Once 50 image collection threshold is met, press `r` to retrain

**Manual:**
```bash
python trigger_retraining.py
```

### 4. Pipeline Phases

The retraining pipeline runs **5 automatic phases**:

1. **Collection Phase**
   - Verifies collected data
   - Shows statistics

2. **Preprocessing Phase**
   - Validates image quality
   - Checks class balance

3. **Training Phase**
   - Retrains model with collected data
   - Creates dataset.yaml automatically
   - Backs up old model before training
   - 50 epochs by default (configurable)

4. **Validation Phase**
   - Tests model on validation set
   - Reports performance metrics

5. **Deployment Phase**
   - Model ready to use
   - Logs completion

## Configuration

Edit `pipeline_config.json` to customize:

```json
{
    "min_images_per_class": 10,      // Minimum images per disease class
    "min_total_images": 50,           // Total images needed
    "training_epochs": 50,            // Training iterations
    "batch_size": 16,                 // Batch size (reduce for low VRAM)
    "imgsz": 640,                     // Image size
    "auto_retrain_enabled": true,     // Enable auto retraining
    "collection_threshold": 50        // Images needed to trigger retraining
}
```

## Dataset Structure

Collected images are automatically organized:

```
disease_dataset/
├── blast/                  (collected during detection)
├── bacterial_blight/       (collected during detection)
├── brown_spot/             (collected during detection)
├── tungro/                 (collected during detection)
├── unknown_disease/        (new diseases detected)
└── uncertain_detections/   (frames saved with 's' key)
```

## New Disease Detection

### Detection Methods

1. **Low Confidence Detection**
   - When detection confidence < 60% on unknown disease
   - System alerts user
   - Image is automatically saved

2. **Manual Capture**
   - Press `s` during detection
   - Frame saved to `uncertain_detections/`
   - Later manually labeled and added to dataset

3. **Unknown Disease Class**
   - Class ID = 4 (reserved for new diseases)
   - Automatically created when detected

### Workflow for New Diseases

1. **Capture Phase** (~1-2 days of operation)
   - System detects and saves suspicious patterns
   - Collections uncertainty detections manually

2. **Labeling Phase** (manual)
   - Review collected images
   - Rename files with disease name
   - Move to appropriate folder

3. **Retraining Phase**
   - Run pipeline to retrain with new data
   - Model learns new disease type

4. **Deployment Phase**
   - Updated model is automatically deployed
   - Next restart uses new model

## Example Workflow

```
Day 1: Start Detection
├─ Point camera at rice fields
├─ Detect 4 known diseases
└─ Automatically collect images

Day 2-3: Data Collection
├─ Continue detection
├─ System detecting an unfamiliar pattern
├─ Saves suspicious frames
└─ Reaches 50 images threshold

Day 3 Evening: Retraining
├─ Press 'r' in main_v2.py
├─ OR run: python trigger_retraining.py
└─ Pipeline retrains with new data (5-10 minutes)

Next Run: Updated Model
├─ Restart main_v2.py
├─ New model loaded
└─ Can now detect the new disease
```

## Monitoring and Logs

### Pipeline Logs

Retraining execution logged in `pipeline_logs.json`:

```json
[
    {
        "timestamp": "2026-02-23T10:30:45.123456",
        "status": "completed"
    },
    {
        "timestamp": "2026-02-23T14:15:20.654321",
        "status": "failed_at_training"
    }
]
```

### Console Output During Main Detection

```
📊 DATASET STATISTICS
==================================================
   blast: 15 images
   bacterial_blight: 12 images
   brown_spot: 18 images
   tungro: 14 images
   unknown_disease: 5 images

   📈 Total Images: 64
==================================================
```

## Hardware Requirements

- **GPU** (Recommended): NVIDIA GPU with CUDA support
- **CPU** (Fallback): Works but slower (~30-60 min per epoch)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 2-5GB for dataset + training runs

## Troubleshooting

### Issue: Training is slow

**Solution**: 
- Reduce `batch_size` in `pipeline_config.json`
- Reduce `training_epochs`
- Use GPU with CUDA support

### Issue: Out of memory during training

**Solution**:
- Reduce `batch_size` (try 8 or 4)
- Reduce `imgsz` to 480 or 416
- Ensure no other heavy programs running

### Issue: Model performance decreased after retraining

**Solution**:
- Check `pipeline_logs.json` for errors
- Restore backup: `cp best_rice_disease_model.pt.backup best_rice_disease_model.pt`
- Collect more diverse data
- Verify dataset quality

### Issue: "No detections" on new diseases

**Solution**:
1. Collect more images of the new disease
2. Ensure images have clear disease symptoms
3. Try lowering confidence threshold
4. Retrain with collected data

## Performance Tips

1. **Collect diverse angles**: Get images from multiple angles
2. **Various lighting**: Collect in different lighting conditions
3. **Multiple plants**: Include different plant stages
4. **Clean labels**: Verify collected images are labeled correctly
5. **Regular retraining**: Retrain monthly with new operational data

## Advanced Usage

### Manual Model Update

```python
from model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.retrain_model(
    epochs=100,           # More epochs for better accuracy
    batch_size=8,         # Smaller batch for limited GPU
    imgsz=512             # Smaller images for speed
)
```

### Check Dataset Before Retraining

```python
from data_collector import DataCollector

collector = DataCollector()
total = collector.print_stats()
print(f"Ready to train: {total >= 50}")
```

## Next Steps

1. ✅ Start using `main_v2.py` for live detection
2. ✅ Let it run and collect data for 1-2 days
3. ✅ Monitor progress with `p` key
4. ✅ Trigger retraining with `r` key when ready
5. ✅ System automatically handles the rest!

---

**Questions?** Check the code comments in each module!
