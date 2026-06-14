"""
Model Training Pipeline
Automatically retrains YOLO model with collected data
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml

class ModelTrainer:
    """Handles model retraining with new disease data"""
    
    def __init__(self, dataset_path="disease_dataset", model_name="best_rice_disease_model.pt"):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.backup_model = f"{model_name}.backup"
        self.training_dir = "training_runs"
        os.makedirs(self.training_dir, exist_ok=True)
    
    def create_yolo_dataset_yaml(self, num_classes=5):
        """
        Create dataset.yaml for YOLO training
        
        Args:
            num_classes: Number of disease classes (4 known + unknown)
        """
        yaml_content = {
            'path': os.path.abspath(self.dataset_path),
            'train': [
                os.path.join(self.dataset_path, 'blast'),
                os.path.join(self.dataset_path, 'bacterial_blight'),
                os.path.join(self.dataset_path, 'brown_spot'),
                os.path.join(self.dataset_path, 'tungro'),
                os.path.join(self.dataset_path, 'unknown_disease')
            ],
            'val': [
                os.path.join(self.dataset_path, 'blast'),
                os.path.join(self.dataset_path, 'bacterial_blight'),
                os.path.join(self.dataset_path, 'brown_spot'),
                os.path.join(self.dataset_path, 'tungro'),
                os.path.join(self.dataset_path, 'unknown_disease')
            ],
            'nc': num_classes,
            'names': {
                0: 'blast',
                1: 'bacterial_blight',
                2: 'brown_spot',
                3: 'tungro',
                4: 'unknown_disease'
            }
        }
        
        with open('dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print("✅ Created dataset.yaml for training")
        return 'dataset.yaml'
    
    def check_training_readiness(self, min_images_per_class=10):
        """
        Check if dataset is ready for training
        
        Args:
            min_images_per_class: Minimum images needed per class
        """
        print("\n" + "=" * 60)
        print("🔍 TRAINING READINESS CHECK")
        print("=" * 60)
        
        ready = True
        
        for disease_dir in os.listdir(self.dataset_path):
            disease_path = os.path.join(self.dataset_path, disease_dir)
            if os.path.isdir(disease_path):
                image_count = len([f for f in os.listdir(disease_path) if f.endswith(('.jpg', '.png'))])
                status = "✅" if image_count >= min_images_per_class else "❌"
                print(f"{status} {disease_dir}: {image_count} images (need {min_images_per_class})")
                
                if image_count < min_images_per_class:
                    ready = False
        
        print("=" * 60)
        if ready:
            print("✅ Dataset is READY for training!")
        else:
            print(f"❌ Need more images (minimum {min_images_per_class} per class)")
        print("=" * 60 + "\n")
        
        return ready
    
    def backup_current_model(self):
        """Backup current model before retraining"""
        if os.path.exists(self.model_name):
            shutil.copy(self.model_name, self.backup_model)
            print(f"✅ Model backed up to {self.backup_model}")
            return True
        return False
    
    def retrain_model(self, epochs=50, imgsz=640, batch_size=16):
        """
        Retrain YOLO model with new data
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size for training
        """
        print("\n" + "=" * 60)
        print("🚀 STARTING MODEL RETRAINING")
        print("=" * 60)
        
        try:
            # Backup current model
            self.backup_current_model()
            
            # Load base model
            print("📦 Loading base model...")
            model = YOLO(self.model_name)
            
            # Create dataset YAML
            dataset_yaml = self.create_yolo_dataset_yaml()
            
            # Train model
            print(f"🎯 Training model for {epochs} epochs...")
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                patience=10,
                device=0,  # GPU device, set to 'cpu' if no GPU
                project=self.training_dir,
                name='rice_disease_model',
                exist_ok=True,
                verbose=True
            )
            
            print("\n✅ TRAINING COMPLETED!")
            print("=" * 60)
            
            return results
        
        except Exception as e:
            print(f"\n❌ ERROR during training: {e}")
            print("🔄 Restoring backup model...")
            if os.path.exists(self.backup_model):
                shutil.copy(self.backup_model, self.model_name)
                print("✅ Backup model restored")
            return None
    
    def validate_model(self):
        """Validate retrained model"""
        print("\n" + "=" * 60)
        print("✅ VALIDATING MODEL")
        print("=" * 60)
        
        try:
            model = YOLO(self.model_name)
            dataset_yaml = self.create_yolo_dataset_yaml()
            
            results = model.val(data=dataset_yaml)
            
            print("✅ Validation complete!")
            print("=" * 60 + "\n")
            return results
        
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return None
    
    def update_dataset_from_production(self, data_collector):
        """
        Update dataset with data collected from production
        
        Args:
            data_collector: DataCollector instance
        """
        total_images = data_collector.print_stats()
        
        if total_images > 50:  # Only retrain if enough data
            print("📊 Dataset size sufficient. Ready to retrain!")
            return True
        else:
            print(f"⏳ Collecting more data... ({total_images}/50 images needed)")
            return False
