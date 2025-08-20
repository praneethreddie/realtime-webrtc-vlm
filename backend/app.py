import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import tensorflow as tf
from pathlib import Path
import time
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.model_type = None  # 'ultralytics' or 'tensorflow'
        self.yolo_model = None
        self.labels = self.load_labels()
        self.load_models()
    
    def load_labels(self):
        """Load COCO labels"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def load_models(self):
        """Load all available models"""
        models_dir = Path('models')
        if not models_dir.exists():
            print("Models directory not found. Please run download_models.py first.")
            return
        
        # Load Ultralytics YOLO models (.pt files)
        pt_files = list(models_dir.glob('*.pt'))
        if pt_files:
            for pt_file in pt_files:
                try:
                    model_name = pt_file.stem
                    self.models[model_name] = {
                        'type': 'ultralytics',
                        'path': str(pt_file)
                    }
                    print(f"Found Ultralytics YOLO model: {model_name}")
                except Exception as e:
                    print(f"Failed to register YOLO model {pt_file}: {e}")
        
        # If no .pt files, try to download yolov5n.pt (but check if it already exists first)
        if not pt_files:
            yolo_path = models_dir / 'yolov5n.pt'
            if not yolo_path.exists():
                try:
                    print("No .pt files found. Downloading yolov5n.pt...")
                    yolo_model = YOLO('yolov5n.pt')  # This will auto-download
                    # Move the downloaded file to models directory
                    import shutil
                    if Path('yolov5n.pt').exists():
                        shutil.move('yolov5n.pt', str(yolo_path))
                    model_name = 'yolov5n'
                    self.models[model_name] = {
                        'type': 'ultralytics',
                        'path': str(yolo_path)
                    }
                    print(f"Downloaded and loaded: {model_name}")
                except Exception as e:
                    print(f"Failed to download yolov5n.pt: {e}")
            else:
                # File exists, just register it
                model_name = 'yolov5n'
                self.models[model_name] = {
                    'type': 'ultralytics',
                    'path': str(yolo_path)
                }
                print(f"Found existing YOLO model: {model_name}")
        
        # Load TensorFlow models (suppress warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for tf_dir in models_dir.iterdir():
                if tf_dir.is_dir():
                    try:
                        model = tf.saved_model.load(str(tf_dir))
                        model_name = tf_dir.name
                        self.models[model_name] = {
                            'model': model,
                            'type': 'tensorflow',
                            'infer': model.signatures['serving_default'] if 'serving_default' in model.signatures else model
                        }
                        print(f"Loaded TensorFlow model: {model_name}")
                    except Exception as e:
                        print(f"Failed to load TensorFlow model {tf_dir}: {e}")
        
        # Set default model
        if self.models:
            first_model = list(self.models.keys())[0]
            self.set_model(first_model)
    
    def set_model(self, model_name):
        """Set the current model"""
        if model_name in self.models:
            self.current_model = model_name
            self.model_type = self.models[model_name]['type']
            
            # Load Ultralytics YOLO model
            if self.model_type == 'ultralytics':
                try:
                    model_path = self.models[model_name]['path']
                    self.yolo_model = YOLO(model_path)
                    print(f"Loaded Ultralytics YOLO: {model_name}")
                except Exception as e:
                    print(f"Failed to load YOLO model: {e}")
                    return False
            
            print(f"Switched to model: {model_name} (type: {self.model_type})")
            return True
        return False
    
    def get_available_models(self):
        """Get list of available models"""
        return {
            'models': list(self.models.keys()),
            'current': self.current_model,
            'details': {name: {'type': info['type']} for name, info in self.models.items()}
        }
    
    def detect_objects(self, image, confidence_threshold=0.5):
        """Detect objects in image using current model"""
        if not self.current_model:
            return []
        
        try:
            if self.model_type == 'ultralytics':
                return self.detect_ultralytics_yolo(image, confidence_threshold)
            elif self.model_type == 'tensorflow':
                return self.detect_tensorflow(image, confidence_threshold)
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_ultralytics_yolo(self, image, confidence_threshold=0.5):
        """Ultralytics YOLO detection"""
        if not self.yolo_model:
            return []
        
        # Run inference
        results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]  # [x, y, width, height]
                    })
        
        return detections
    
    def preprocess_image_tensorflow(self, image, target_size=(320, 320)):
        """Preprocess image for TensorFlow models"""
        # Resize image
        resized = cv2.resize(image, target_size)
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Convert to uint8 tensor
        input_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)
        # Add batch dimension
        input_tensor = tf.expand_dims(input_tensor, 0)
        return input_tensor
    
    def detect_tensorflow(self, image, confidence_threshold):
        """TensorFlow model detection"""
        model_info = self.models[self.current_model]
        model = model_info['model']
        infer = model_info['infer']
        
        # Preprocess
        input_tensor = self.preprocess_image_tensorflow(image)
        
        # Run inference
        detections = infer(input_tensor)
        
        # Post-process
        return self.postprocess_tensorflow(detections, image.shape, confidence_threshold)
    
    def postprocess_tensorflow(self, detections, image_shape, confidence_threshold):
        """Post-process TensorFlow detection results"""
        results = []
        h, w = image_shape[:2]
        
        # Extract detection results
        if 'detection_boxes' in detections:
            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(int)
            scores = detections['detection_scores'][0].numpy()
        else:
            return results
        
        for i in range(len(boxes)):
            if scores[i] > confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                y1, x1, y2, x2 = boxes[i]
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                
                # Get class name
                class_id = classes[i] - 1  # TensorFlow models often use 1-based indexing
                class_name = self.labels[class_id] if 0 <= class_id < len(self.labels) else f"class_{class_id}"
                
                results.append({
                    'class': class_name,
                    'confidence': float(scores[i]),
                    'bbox': [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]
                })
        
        return results

# Initialize model manager
model_manager = ModelManager()

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify(model_manager.get_available_models())

@app.route('/api/models/<model_name>', methods=['POST'])
def set_model(model_name):
    """Set current model"""
    success = model_manager.set_model(model_name)
    return jsonify({'success': success, 'current_model': model_manager.current_model})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to object detection server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Check if data contains frame (not image)
        if not isinstance(data, dict) or 'frame' not in data:
            print(f"Invalid data format. Expected dict with 'frame' key, got: {type(data)}")
            emit('detection_result', {'error': 'Invalid data format - missing frame key'})
            return
        
        # Get confidence threshold from data if provided (lowered default)
        confidence_threshold = data.get('confidence', 0.3)  # Changed from 0.5 to 0.3
        
        # Decode base64 image
        image_data_str = data['frame']  # Changed from 'image' to 'frame'
        if ',' in image_data_str:
            image_data = base64.b64decode(image_data_str.split(',')[1])
        else:
            image_data = base64.b64decode(image_data_str)
            
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            emit('detection_result', {'error': 'Failed to decode image'})
            return
        
        start_time = time.time()
        
        # Perform object detection with dynamic confidence threshold
        detections = model_manager.detect_objects(image, confidence_threshold=confidence_threshold)
        
        detection_time = time.time() - start_time
        
        print(f"Processing frame: {image.shape}, Model: {model_manager.current_model} ({model_manager.model_type})")
        print(f"Detected {len(detections)} objects in {detection_time:.3f}s (confidence: {confidence_threshold})")
        
        # Send results back to client
        emit('detection_result', {
            'detections': detections,
            'processing_time': detection_time,
            'model': model_manager.current_model,
            'confidence_used': confidence_threshold
        })
        
    except KeyError as e:
        print(f"Missing key in data: {e}")
        emit('detection_result', {'error': f'Missing required data: {str(e)}'})
    except Exception as e:
        print(f"Error processing frame: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        emit('detection_result', {'error': str(e)})

if __name__ == '__main__':
    print("Starting object detection server...")
    print(f"Available models: {list(model_manager.models.keys())}")
    print(f"Current model: {model_manager.current_model}")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)  # Changed debug=True to debug=False