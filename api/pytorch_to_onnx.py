import torch
import torchvision
import onnx
import os

path = os.getcwd()
model = torch.jit.load("{}/model/GLUE.pt".format(path), map_location="cuda")

path_export = "{}/model/GLUE.onnx".format(path)
fake_image = torch.randn(1, 3, 256, 256).to("cuda")
torch.onnx.export(model, fake_image, path_export)
model = onnx.load(path_export)
print(onnx.checker.check_model(model))
