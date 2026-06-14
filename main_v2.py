import cv2
from ultralytics import YOLO
from spray_control import activate_spray
from data_collector import DataCollector
from pipeline_orchestrator import PipelineOrchestrator

model = YOLO("best_rice_disease_model.pt")
data_collector = DataCollector()
orchestrator = PipelineOrchestrator()

# Disease labels
DISEASE_NAMES = {
    0: "🦠 Blast",
    1: "🦠 Bacterial Blight",
    2: "🦠 Brown Spot",
    3: "🦠 Tungro"
}

# VERY HIGH CONFIDENCE THRESHOLDS TO AVOID FALSE POSITIVES (STRICT MODE)
CONFIDENCE_THRESHOLD_KNOWN = 0.88  # Known diseases must be 88% confident (STRICTER!)
CONFIDENCE_THRESHOLD_UNKNOWN = 0.80  # Unknown diseases must be 80% confident (STRICTER!)

# MINIMUM DETECTION SIZE - disease should be reasonably large (not tiny specs)
MIN_DETECTION_WIDTH = 30   # Minimum width in pixels
MIN_DETECTION_HEIGHT = 30  # Minimum height in pixels

def run_precision_spraying():
    """
    Main function to run the precision spraying drone system.
    Detects rice diseases in real-time, activates spray, and collects data for retraining.
    """
    current_model = model
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False

    print("✅ Camera opened successfully")
    print("🎯 Starting live disease detection...")
    print("📹 Press 'q' to quit, 's' to save frame for review, 'p' for pipeline status")
    print("📊 Data is being collected for automated retraining")
    print(f"\n⚠️  ULTRA-STRICT CONFIDENCE THRESHOLDS (to prevent false positives):")
    print(f"   Known diseases: >{CONFIDENCE_THRESHOLD_KNOWN*100:.0f}% confidence required")
    print(f"   Unknown diseases: >{CONFIDENCE_THRESHOLD_UNKNOWN*100:.0f}% confidence required")
    print(f"   Minimum detection size: {MIN_DETECTION_WIDTH}x{MIN_DETECTION_HEIGHT} pixels\n")
    
    disease_detected = False
    frame_count = 0
    spray_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Run YOLO detection with VERY STRICT confidence (0.7 minimum to even consider)
        results = model(frame, conf=0.7)
        annotated_frame = results[0].plot()

        # Check for detected diseases
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            disease_name = DISEASE_NAMES.get(cls_id, "Unknown Disease")
            
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            
            # Check minimum size (ignore tiny specs/noise)
            if width < MIN_DETECTION_WIDTH or height < MIN_DETECTION_HEIGHT:
                continue  # Skip this detection - too small
            
            # Determine threshold based on disease type
            is_unknown = cls_id == 4 or conf < CONFIDENCE_THRESHOLD_KNOWN
            threshold = CONFIDENCE_THRESHOLD_UNKNOWN if is_unknown else CONFIDENCE_THRESHOLD_KNOWN
            
            # VERY STRICT: only spray if EXTREMELY confident
            if conf > threshold:
                disease_detected = True
                spray_count += 1
                
                # Calculate center point (x1, y1, x2, y2 already extracted above)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                print(f"\n🚨 DISEASE DETECTED! [Frame {frame_count}]")
                print(f"   Disease Type: {disease_name}")
                print(f"   Confidence: {conf:.2%}")
                print(f"   Location: ({center_x}, {center_y})")
                print(f"   Region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                # Activate spray for the affected area
                activate_spray()
                print(f"   💧 Spraying {disease_name} at coordinates ({center_x}, {center_y})")
                
                # Save image to training dataset
                filepath = data_collector.save_disease_image(
                    frame, cls_id, conf, (center_x, center_y)
                )
                print(f"   💾 Saved to training dataset: {filepath}")
                
                # Check if we should trigger retraining
                if orchestrator.check_trigger_retraining():
                    print(f"\n🚀 COLLECTION THRESHOLD REACHED!")
                    print(f"   Ready to retrain model with new data")
                    print(f"   Type 'r' to start retraining, or 'q' to continue detection")

        # Display live feed with frame counter and spray counter
        display_text = f"Frame: {frame_count} | Sprayed: {spray_count}"
        cv2.putText(annotated_frame, display_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("🌾 Precision Spraying Drone - Live Detection", annotated_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n🛑 Stopping detection...")
            break
        
        elif key == ord('s'):
            # Save current frame for manual review
            filepath = data_collector.save_frame_for_labeling(frame, "manual_review")
            print(f"\n📸 Frame saved for review: {filepath}")
        
        elif key == ord('p'):
            # Print pipeline status
            print("\n" + "=" * 60)
            print("📊 CURRENT PIPELINE STATUS")
            orchestrator.print_status()
            print(f"   Total Sprays: {spray_count}")
            print(f"   Total Frames: {frame_count}")
            print("=" * 60)
        
        elif key == ord('r'):
            # Start retraining
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 60)
            print("🚀 STARTING AUTOMATED RETRAINING")
            print("=" * 60)
            
            success = orchestrator.run_full_pipeline()
            
            if success:
                print("\n✅ Model retraining completed successfully!")
                print("🔄 Restarting detection with new model...")
                
                # Reload model with new weights
                current_model = YOLO("best_rice_disease_model.pt")
                print("✅ New model loaded!")
                
                # Reset counters
                spray_count = 0
                frame_count = 0
            else:
                print("\n❌ Retraining failed. Continuing with current model...")
            
            # Reopen camera
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("❌ Cannot reopen camera")
                break

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM SHUTDOWN")
    print("=" * 60)
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Disease Detections: {spray_count}")
    data_collector.print_stats()
    print("=" * 60)
    
    return disease_detected

if __name__ == "__main__":
    print("=" * 60)
    print("🚁 PRECISION SPRAYING DRONE SYSTEM v2.0")
    print("🌾 Rice Disease Detection & Automated Retraining")
    print("=" * 60)
    print("\n📋 Detecting Diseases:")
    for cls_id, disease in DISEASE_NAMES.items():
        print(f"   {disease}")
    print("\n🎮 Controls:")
    print("   'q' - Quit")
    print("   's' - Save frame for manual review")
    print("   'p' - Show pipeline status")
    print("   'r' - Start retraining (when ready)")
    print("\n" + "=" * 60 + "\n")
    
    # Show initial status
    orchestrator.print_status()
    
    run_precision_spraying()
