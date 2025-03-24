import torch
from torchvision.transforms import transforms as T


class PresetTrain:
    def __init__(self):   
        transforms = [T.PILToTensor(),
                      T.ConvertImageDtype(torch.float)]
        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class PresetEval:
    def __init__(self):
        transforms = [T.PILToTensor(),
                      T.ConvertImageDtype(torch.float)]
        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
