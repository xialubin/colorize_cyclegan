import os
from skimage import io

path = './dataset'
for folder in os.listdir(path):
    if folder.find("A") != -1:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            if len(img.shape) != 2:
                os.remove(img_path)
    if folder.find("B") != -1:
        folder_path = os.path.join(path, folder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            if len(img.shape) != 3:
                os.remove(img_path)
            elif img.shape[2] != 3:
                os.remove(img_path)