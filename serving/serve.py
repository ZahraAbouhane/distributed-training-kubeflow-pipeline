from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
from minio import Minio

app = Flask(__name__)

print("Downloading model from MinIO...")
client = Minio("minio.minio.svc.cluster.local:9000",
              access_key="minioadmin",
              secret_key="minioadmin",
              secure=False)
client.fget_object("models", "resnet18_cifar10.pth", "model.pth")
print("Model downloaded!")

# Load model
model = models.resnet18(num_classes=10)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
print("Model loaded!")

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/v1/models/cifar10:predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    img = Image.open(request.files['image']).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
    
    return jsonify({
        'predictions': [{
            'class': classes[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {classes[i]: float(probs[i]) for i in range(10)}
        }]
    })

@app.route('/v1/models/cifar10', methods=['GET'])
def health():
    return jsonify({'name': 'cifar10', 'ready': True})

if __name__ == '__main__':
    print("Starting Flask server on port 8080...")
    app.run(host='0.0.0.0', port=8080)