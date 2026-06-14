import cv2
from ultralytics import YOLO
from spray_control import activate_spray

model = YOLO("best_rice_disease_model.pt")

# Disease labels
DISEASE_NAMES = {
    0: "🦠 Blast",
    1: "🦠 Bacterial Blight",
    2: "🦠 Brown Spot",
    3: "🦠 Tungro"
}

def run_precision_spraying():
    """
    Main function to run the precision spraying drone system.
    Detects rice diseases in real-time and activates spray for affected areas.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False

    print("✅ Camera opened successfully")
    print("🎯 Starting live disease detection...")
    print("📹 Press 'q' to quit")
    
    disease_detected = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        # Check for detected diseases
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            disease_name = DISEASE_NAMES.get(cls_id, "Unknown Disease")
            
            if conf > 0.5:
                disease_detected = True
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
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

        # Display live feed
        cv2.imshow("🌾 Precision Spraying Drone - Live Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Stopping detection...")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ System shutdown complete")
    return disease_detected

if __name__ == "__main__":
    print("=" * 60)
    print("🚁 PRECISION SPRAYING DRONE SYSTEM")
    print("🌾 Rice Disease Detection & Spray Control")
    print("=" * 60)
    print("\n📋 Detecting Diseases:")
    for cls_id, disease in DISEASE_NAMES.items():
        print(f"   {disease}")
    print("\n" + "=" * 60 + "\n")
    run_precision_spraying()
