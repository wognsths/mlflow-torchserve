from ts.torch_handler.base_handler import BaseHandler
import torch, io, torchvision.transforms as T
from model.model import MNISTModel
from PIL import Image
import json, base64

class MNISTHandler(BaseHandler):
    def initialize(self, ctx):
        self.device = "cpu"
        self.model  = MNISTModel()
        self.model.load_state_dict(torch.load(ctx.get_model_path(), map_location=self.device))
        self.model.eval()
        self.transform = T.Compose([T.Grayscale(), T.Resize((28,28)), T.ToTensor()])

    def preprocess(self, data):
        # Base64-encoded png/jpeg → tensor
        img_bytes = base64.b64decode(data[0]["body"])
        img = img = Image.open(io.BytesIO(data[0]["body"])).convert("L")
        return self.transform(img).unsqueeze(0)

    def inference(self, x):
        with torch.no_grad():
            out = self.model(x)
            return torch.argmax(out, dim=1).item()

    def postprocess(self, pred):
        return {"digit": int(pred)}
