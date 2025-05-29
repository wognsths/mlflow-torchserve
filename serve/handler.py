import sys
import os
import io
import json
import base64
import torch
import torchvision.transforms as T
from PIL import Image
from model import MNISTModel
from ts.torch_handler.base_handler import BaseHandler

sys.path.append(os.path.dirname(__file__))

class MNISTHandler(BaseHandler):
    def initialize(self, ctx):
        self.device = "cpu"
        self.manifest = ctx.manifest
        self.system_properties = ctx.system_properties
        self.model_dir = self.system_properties.get("model_dir")  # 💡 fix point

        model_path = os.path.join(self.model_dir, "mnist_model_state.pt")
        self.model = MNISTModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = T.Compose([
            T.Grayscale(), 
            T.Resize((28, 28)), 
            T.ToTensor()
        ])

    def preprocess(self, data):
        payload = data[0].get("body").get("body") or data[0].get("body")

        if not isinstance(payload, str):
            raise ValueError("Expected base64-encoded string in 'body'")

        img_bytes = base64.b64decode(payload)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        return self.transform(img).unsqueeze(0)

    def inference(self, x):
        with torch.no_grad():
            out = self.model(x)
            print(f"out: {out}")
            result = torch.argmax(out, dim=1).item()
            print(f"result: {result}")
            return torch.argmax(out, dim=1).item()

    def postprocess(self, pred):
        return [int(pred)]
