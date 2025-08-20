import os
import urllib.request
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import zipfile
import requests

def download_models():
    """Download ONNX and TensorFlow models from various sources"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # ONNX Models
    onnx_models = {
        'mobilenet_ssd.onnx': 'https://huggingface.co/qualcomm/MobileNet-v2/resolve/main/MobileNet-v2.onnx',
        'yolov5s.onnx': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx'
    }
    
    # TensorFlow Models
    tf_models = {
        'ssd_mobilenet_v2_coco': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
        'efficientdet_d0_coco': 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
        'centernet_mobilenetv2_fpn_od': 'https://tfhub.dev/tensorflow/centernet/mobilenetv2_fpn_od/1'
    }
    
    # Download ONNX models
    for filename, url in onnx_models.items():
        filepath = models_dir / filename
        if not filepath.exists():
            print(f'Downloading {filename}...')
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f'✓ Downloaded {filename}')
            except Exception as e:
                print(f'✗ Failed to download {filename}: {e}')
        else:
            print(f'✓ {filename} already exists')
    
    # Download TensorFlow models
    for model_name, url in tf_models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f'Downloading TensorFlow model {model_name}...')
            try:
                # Download and save TensorFlow Hub model
                model = hub.load(url)
                tf.saved_model.save(model, str(model_path))
                print(f'✓ Downloaded {model_name}')
            except Exception as e:
                print(f'✗ Failed to download {model_name}: {e}')
        else:
            print(f'✓ {model_name} already exists')
    
    # Download COCO labels
    labels_path = models_dir / 'coco_labels.txt'
    if not labels_path.exists():
        print('Downloading COCO labels...')
        try:
            labels_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
            response = requests.get(labels_url)
            # Convert pbtxt to simple text format
            coco_labels = [
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
            with open(labels_path, 'w') as f:
                for label in coco_labels:
                    f.write(f'{label}\n')
            print('✓ Downloaded COCO labels')
        except Exception as e:
            print(f'✗ Failed to download COCO labels: {e}')
    else:
        print('✓ COCO labels already exist')

if __name__ == '__main__':
    download_models()