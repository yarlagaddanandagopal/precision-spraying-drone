"""
Data Collection Module
Collects and manages disease images for model retraining
"""
import cv2
import os
from datetime import datetime
from pathlib import Path

class DataCollector:
    """Collects disease images for dataset expansion"""
    
    def __init__(self, base_path="disease_dataset"):
        self.base_path = base_path
        self.disease_classes = {
            0: "blast",
            1: "bacterial_blight",
            2: "brown_spot",
            3: "tungro",
            4: "unknown_disease"  # New diseases
        }
        self.setup_directories()
    
    def setup_directories(self):
        """Create directory structure for dataset"""
        for disease_class in self.disease_classes.values():
            disease_dir = os.path.join(self.base_path, disease_class)
            os.makedirs(disease_dir, exist_ok=True)
            print(f"✅ Created directory: {disease_dir}")
    
    def save_disease_image(self, frame, disease_id, confidence, location):
        """
        Save detected disease image to dataset
        
        Args:
            frame: Image frame from camera
            disease_id: Class ID of detected disease
            confidence: Detection confidence
            location: Tuple of (x, y) center coordinates
        """
        disease_name = self.disease_classes.get(disease_id, "unknown_disease")
        disease_dir = os.path.join(self.base_path, disease_name)
        
        # Create filename with timestamp and confidence
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{disease_name}_{timestamp}_{confidence:.2f}.jpg"
        filepath = os.path.join(disease_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        
        return filepath
    
    def save_frame_for_labeling(self, frame, suspected_class="unknown"):
        """Save frame that needs manual labeling"""
        uncertain_dir = os.path.join(self.base_path, "uncertain_detections")
        os.makedirs(uncertain_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"uncertain_{suspected_class}_{timestamp}.jpg"
        filepath = os.path.join(uncertain_dir, filename)
        
        cv2.imwrite(filepath, frame)
        return filepath
    
    def get_dataset_stats(self):
        """Get statistics about collected data"""
        stats = {}
        for disease_name in self.disease_classes.values():
            disease_dir = os.path.join(self.base_path, disease_name)
            if os.path.exists(disease_dir):
                image_count = len([f for f in os.listdir(disease_dir) if f.endswith(('.jpg', '.png'))])
                stats[disease_name] = image_count
        return stats
    
    def print_stats(self):
        """Print dataset statistics"""
        stats = self.get_dataset_stats()
        print("\n" + "=" * 50)
        print("📊 DATASET STATISTICS")
        print("=" * 50)
        total_images = 0
        for disease, count in stats.items():
            print(f"   {disease.capitalize()}: {count} images")
            total_images += count
        print(f"\n   📈 Total Images: {total_images}")
        print("=" * 50 + "\n")
        return total_images
