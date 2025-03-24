import os
from PIL import Image
from torch.utils.data import Dataset


class_names = ['airplane', 'apple', 'ball', 'bear', 'bed', 'bench', 'bird', 'burger', 'butterfly', 'car',
               'cat', 'clock', 'cup', 'dog', 'elephant', 'fox', 'frog', 'horse', 'house', 'koala',
               'ladybug', 'monkey', 'motorcycle', 'mushroom', 'panda', 'pen', 'phone', 'piano', 'pizza', 'rabbit',
               'shark', 'ship', 'shoe', 'snail', 'snake', 'spaghetti', 'swan', 'table', 'tie', 'tiger',
               'train', 'turtle']


class ImageNetDatasetPair(Dataset):
    def __init__(self, hq_dir, lq_dir, transform):
        self.hq_paths = sorted([os.path.join(hq_dir, path) for path in os.listdir(hq_dir)])
        self.lq_paths = sorted([os.path.join(lq_dir, path) for path in os.listdir(lq_dir)])
        self.transform = transform


    def __getitem__(self, index):
        hq_path = self.hq_paths[index]
        hq_img = Image.open(hq_path)
        hq_img = self.transform(hq_img)

        lq_path = self.lq_paths[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.transform(lq_img)

        _, image_names = os.path.split(hq_path)
        image_name = image_names.split('_')[0]
        label = class_names.index(image_name)

        return hq_img, lq_img, label


    def __len__(self):
        return len(self.hq_paths)
