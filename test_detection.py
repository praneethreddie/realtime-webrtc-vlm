import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

def test_object_detection():
    print("🔍 Testing Object Detection Directly")
    print("=====================================")
    
    # Load the model
    model_path = Path('backend/models/yolov5nu.pt')
    if not model_path.exists():
        print("📥 Model not found, downloading yolov5n...")
        model = YOLO('yolov5n.pt')
    else:
        print(f"📂 Loading model: {model_path}")
        model = YOLO(str(model_path))
    
    print(f"✅ Model loaded successfully")
    
    # Initialize camera
    print("📷 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    print("\n🎯 Starting detection test...")
    print("Press 'q' to quit, 's' to save current frame")
    print("Try pointing camera at: person, chair, cup, phone, laptop, etc.")
    print("\n" + "="*50)
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Run detection with multiple confidence levels
            results_high = model(frame, conf=0.5, verbose=False)
            results_medium = model(frame, conf=0.3, verbose=False)
            results_low = model(frame, conf=0.1, verbose=False)
            
            detection_time = time.time() - start_time
            
            # Count detections at different confidence levels
            detections_high = len(results_high[0].boxes) if len(results_high[0].boxes) > 0 else 0
            detections_medium = len(results_medium[0].boxes) if len(results_medium[0].boxes) > 0 else 0
            detections_low = len(results_low[0].boxes) if len(results_low[0].boxes) > 0 else 0
            
            if detections_high > 0 or detections_medium > 0 or detections_low > 0:
                detection_count += 1
            
            # Display results on frame
            annotated_frame = results_medium[0].plot()
            
            # Add info text
            info_text = [
                f"Frame: {frame_count} | Time: {detection_time:.3f}s",
                f"Detections - High(0.5): {detections_high}, Med(0.3): {detections_medium}, Low(0.1): {detections_low}",
                f"Total frames with detections: {detection_count}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Print detection info every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: High={detections_high}, Med={detections_medium}, Low={detections_low} | {detection_time:.3f}s")
                
                # Print detected objects if any
                if detections_medium > 0:
                    print("  Detected objects:")
                    for box in results_medium[0].boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        print(f"    - {class_name}: {confidence:.2f}")
            
            # Show the frame
            cv2.imshow('Object Detection Test', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"💾 Saved frame as {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print("📊 TEST SUMMARY:")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with detections: {detection_count}")
        print(f"Detection rate: {(detection_count/frame_count*100):.1f}%" if frame_count > 0 else "N/A")
        
        if detection_count == 0:
            print("\n🤔 NO DETECTIONS FOUND - Possible reasons:")
            print("   1. Camera pointing at blank wall/ceiling")
            print("   2. Poor lighting conditions")
            print("   3. No recognizable objects in view")
            print("   4. Objects too small or far away")
            print("\n💡 Try pointing camera at yourself or common objects!")
        else:
            print("\n✅ DETECTION IS WORKING! The model can detect objects.")
            print("   The WebRTC app should work fine with proper camera positioning.")

if __name__ == "__main__":
    test_object_detection()