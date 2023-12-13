import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import keep_image_size_open
import sys
sys.path.append("../")

transform = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path ,mode = 'train'):
        self.path = path
        self.mode = mode
        if mode == 'train':
            self.names_hazy = [f for f in os.listdir(os.path.join(path, 'train_hazy')) if f.endswith('.bmp')]
            self.names_image = [f for f in os.listdir(os.path.join(path, 'train_GT')) if f.endswith('.bmp')]
        else:
            self.names_hazy = [f for f in os.listdir(os.path.join(path, 'Hazy')) if f.endswith('.bmp')]
            self.names_image = [f for f in os.listdir(os.path.join(path, 'GT')) if f.endswith('.bmp')]

    def __len__(self):
        return min(len(self.names_hazy), len(self.names_image))

    def __getitem__(self, index):
        index_str = str(index + 1) # Assuming index starts from 1
        hazy_name = f"{index_str}_Hazy.bmp"
        image_name = f"{index_str}_Image_.bmp"

        if self.mode == 'train':
            hazy_path = os.path.join(self.path, 'train_hazy', hazy_name)
            image_path = os.path.join(self.path, 'train_GT', image_name)
        else:
            hazy_path = os.path.join(self.path, 'Hazy', hazy_name)
            image_path = os.path.join(self.path, 'GT', image_name)


        hazy_image = keep_image_size_open(hazy_path)
        clear_image = keep_image_size_open(image_path)

        hazy_tensor = transform(hazy_image)
        clear_tensor = transform(clear_image)

        return hazy_tensor, clear_tensor

if __name__ == '__main__':
    data = MyDataset('./dataset/',mode='test')

    print(len(data))
    print(data[0][0].shape)
    print(data[0][1].shape)
