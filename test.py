import os
from torch.utils.data import DataLoader
from data import MyDataset
from model import unet
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import time
import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import ipdb;

# ipdb.set_trace()
weight_path = './weight/unet.pth'
data_path = './dataset/'  # Replace with the path to your test dataset
result_path = './test_results/'

if __name__ == '__main__':
    # Assuming you have a MyDataset class for testing similar to the training data loader
    test_data_loader = DataLoader(MyDataset(data_path, mode='test'), batch_size=1, shuffle=False)

    model = unet.UNet().to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with torch.no_grad():
        for i, (hazy, clear) in enumerate(test_data_loader):
            hazy = hazy.to(device)

            # Forward pass
            print("image shape:", hazy.shape)
            inference_times=[]
            for i in range(20):
                start = time.time()
                output = model(hazy)
                end = time.time()
                inference_time=end-start
                inference_times.append(inference_time)
            median_time=np.median(inference_times)
            plt.bar(range(1,len(inference_times)+1),inference_times)
            plt.xticks(range(1,len(inference_times)+1))
            plt.xlabel('Iteration')
            plt.ylabel('Inference time (seconds)')
            plt.title('Inference time in unet_small model and 1920*1920 images')
            plt.axhline(y=median_time, color='r', linestyle='-',label='Median: {:.4f}s'.format(median_time))
            plt.legend()
            plt.show()
            # Save the result image
            _hazy = hazy[0].cpu()
            _output = output[0].cpu()
            
            img = torch.stack([_hazy, _output], dim=0)
            torchvision.utils.save_image(img, f'{result_path}/{i}_result.png')

            # Optionally, you can save the clear image for comparison
            _clear = clear[0].cpu()
            img_clear = torch.stack([_clear], dim=0)
            torchvision.utils.save_image(img_clear, f'{result_path}/{i}_clear.png')

    print("Testing complete. Results saved in:", result_path)
