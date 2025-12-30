import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

from torchvision import transforms

inference_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

onnx_model_path = "/Users/fdl/Downloads/hair_classifier_v1.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# Get input and output names
inputs = session.get_inputs()
outputs = session.get_outputs()
input_name = inputs[0].name
output_name = outputs[0].name



def lambda_handler(event, context):

    url = event['url']
    # Run inference
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
    img_tensor = inference_transforms(img).unsqueeze(0)
    result = session.run([output_name], {input_name: img_tensor.numpy()})
    float_pred = result[0][0].tolist()

    # Return the result
    return float_pred