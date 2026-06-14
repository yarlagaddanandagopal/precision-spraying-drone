"""
Automated Retraining Pipeline Orchestrator
Manages data collection, training, and model deployment
"""
import os
import json
from datetime import datetime
from data_collector import DataCollector
from model_trainer import ModelTrainer

class PipelineOrchestrator:
    """Orchestrates the complete retraining pipeline"""
    
    def __init__(self, config_file="pipeline_config.json"):
        self.config_file = config_file
        self.collector = DataCollector()
        self.trainer = ModelTrainer()
        self.load_config()
    
    def load_config(self):
        """Load pipeline configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "min_images_per_class": 10,
                "min_total_images": 50,
                "training_epochs": 50,
                "batch_size": 16,
                "imgsz": 640,
                "auto_retrain_enabled": True,
                "collection_threshold": 50
            }
            self.save_config()
        
        print("✅ Configuration loaded")
    
    def save_config(self):
        """Save pipeline configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def check_trigger_retraining(self):
        """Check if conditions are met to trigger retraining"""
        stats = self.collector.get_dataset_stats()
        total_images = sum(stats.values())
        
        if total_images >= self.config["collection_threshold"]:
            return True
        return False
    
    def run_collection_phase(self, duration_minutes=None):
        """
        Run data collection phase
        
        Args:
            duration_minutes: How long to collect data (None = until manual stop)
        """
        print("\n" + "=" * 60)
        print("📸 DATA COLLECTION PHASE")
        print("=" * 60)
        print(f"Target: {self.config['collection_threshold']} images")
        print(f"Collecting until threshold is reached...")
        print("=" * 60 + "\n")
        
        self.collector.print_stats()
    
    def run_preprocessing_phase(self):
        """Preprocess and validate collected data"""
        print("\n" + "=" * 60)
        print("⚙️  DATA PREPROCESSING PHASE")
        print("=" * 60)
        
        stats = self.collector.get_dataset_stats()
        total_images = sum(stats.values())
        
        print(f"✅ Total images collected: {total_images}")
        print(f"✅ Classes detected: {len(stats)}")
        
        # Check balance
        if total_images > 0:
            avg = total_images / len(stats) if len(stats) > 0 else 0
            print(f"✅ Average per class: {avg:.1f}")
        
        print("=" * 60 + "\n")
    
    def run_training_phase(self):
        """Run model training phase"""
        print("\n" + "=" * 60)
        print("🚀 TRAINING PHASE")
        print("=" * 60)
        
        # Check readiness
        if not self.trainer.check_training_readiness(
            self.config["min_images_per_class"]
        ):
            print("❌ Dataset not ready for training")
            return False
        
        # Run training
        results = self.trainer.retrain_model(
            epochs=self.config["training_epochs"],
            batch_size=self.config["batch_size"],
            imgsz=self.config["imgsz"]
        )
        
        return results is not None
    
    def run_validation_phase(self):
        """Run model validation phase"""
        print("\n" + "=" * 60)
        print("✅ VALIDATION PHASE")
        print("=" * 60)
        
        results = self.trainer.validate_model()
        return results is not None
    
    def run_deployment_phase(self):
        """Deploy retrained model"""
        print("\n" + "=" * 60)
        print("🎯 DEPLOYMENT PHASE")
        print("=" * 60)
        print("✅ Model is ready for deployment!")
        print("✅ Restart main.py to use the new model")
        print("=" * 60 + "\n")
    
    def run_full_pipeline(self):
        """Run complete retraining pipeline"""
        print("\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 58 + "║")
        print("║" + "  🤖 AUTOMATED RETRAINING PIPELINE".center(58) + "║")
        print("║" + " " * 58 + "║")
        print("╚" + "=" * 58 + "╝")
        
        pipeline_steps = [
            ("Collection", self.run_collection_phase),
            ("Preprocessing", self.run_preprocessing_phase),
            ("Training", self.run_training_phase),
            ("Validation", self.run_validation_phase),
            ("Deployment", self.run_deployment_phase)
        ]
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "status": "started"
        }
        
        try:
            for step_name, step_func in pipeline_steps:
                print(f"\n▶️  Running: {step_name}")
                result = step_func()
                
                if step_name in ["Training", "Validation"] and result is False:
                    log_entry["status"] = f"failed_at_{step_name}"
                    print(f"❌ Pipeline failed at {step_name} phase")
                    self.save_pipeline_log(log_entry)
                    return False
            
            log_entry["status"] = "completed"
            self.save_pipeline_log(log_entry)
            
            print("\n")
            print("╔" + "=" * 58 + "╗")
            print("║" + " " * 58 + "║")
            print("║" + "  ✅ PIPELINE COMPLETED SUCCESSFULLY!".center(58) + "║")
            print("║" + " " * 58 + "║")
            print("╚" + "=" * 58 + "╝")
            print()
            
            return True
        
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            log_entry["status"] = f"error: {str(e)}"
            self.save_pipeline_log(log_entry)
            return False
    
    def save_pipeline_log(self, entry):
        """Save pipeline execution log"""
        log_file = "pipeline_logs.json"
        logs = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=4)
    
    def print_status(self):
        """Print current pipeline status"""
        print("\n" + "=" * 60)
        print("📊 PIPELINE STATUS")
        print("=" * 60)
        
        self.collector.print_stats()
        
        should_retrain = self.check_trigger_retraining()
        status = "🟢 READY TO RETRAIN" if should_retrain else "🔴 COLLECTING DATA"
        print(f"Status: {status}\n")

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    orchestrator.print_status()
