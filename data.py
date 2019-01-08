import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, color
from skimage.transform import resize


class Dataset(Dataset):
    def __init__(self, dataset_name, augment, phase='train', img_size=256, flip=True):
        super(Dataset, self).__init__()
        for folder in os.listdir(dataset_name):
            if folder == phase + '_A':
                self.datapath_A = []
                for filename in os.listdir(os.path.join(dataset_name, folder)):
                    self.datapath_A.append(os.path.join(dataset_name, folder, filename))
            elif folder == phase + '_B':
                self.datapath_B = []
                for filename in os.listdir(os.path.join(dataset_name, folder)):
                    self.datapath_B.append(os.path.join(dataset_name, folder, filename))
            # else:
            #     raise ValueError("dataset only contains train_A, train_B, test_A, test_B")
        self.augment = augment
        self.phase = phase
        self.img_size = img_size
        self.flip = flip

    def __len__(self):
        return np.minimum(len(self.datapath_A), len(self.datapath_B))

    def __getitem__(self, item):
        image_A = io.imread(self.datapath_A[item])
        image_B = io.imread(self.datapath_B[item])

        if self.augment is not None:
            image_A = self.crop_or_resize(image_A)
            image_B = self.crop_or_resize(image_B)
        if self.phase == 'train' and self.flip:
            if np.random.rand(1) > 0.5:
                image_A = image_A[:, ::-1]
                image_B = image_B[:, ::-1, :]
        # rgb to lab  image_A is gray
        image_B = color.rgb2lab(image_B)
        # to [-1, 1]
        re_A = ((image_A / 128.0) - 1.0)[np.newaxis, :, :].astype(np.float32)
        re_L = ((image_B[:, :, 0:1] / 50) - 1.0).transpose(2, 0, 1).astype(np.float32)
        re_ab = (image_B[:, :, 1:] / 110).transpose(2, 0, 1).astype(np.float32)
        re_B = np.concatenate((re_L, re_ab), axis=0)
        return {'A': re_L, 'B': re_ab}

    def crop_or_resize(self, image):
        augment_type = self.augment
        if augment_type == 'crop':
            h = image.shape[0]
            w = image.shape[1]
            if h > self.img_size and w > self.img_size:
                start_h = np.random.randint(h - self.img_size)
                start_w = np.random.randint(w - self.img_size)
                if len(image.shape) == 2:
                    return image[start_h:(start_h+self.img_size), start_w:(start_w+self.img_size)]
                elif len(image.shape) == 3:
                    return image[start_h:(start_h + self.img_size), start_w:(start_w + self.img_size), :]
                else:
                    raise ValueError('the image channel must be 2 or 3')
            else:
                print('the image is too small, turn to resize augment')
                augment_type = 'resize'
        if augment_type == 'resize':
            h = image.shape[0]
            w = image.shape[1]
            if h == w:
                return (resize(image, (self.img_size, self.img_size)) * 255).astype(np.uint8)
            elif h > w:
                start = np.random.randint(h - w)
                if len(image.shape) == 2:
                    img = image[start:(start+w), :]
                elif len(image.shape) == 3:
                    img = image[start:(start+w), :, :]
                else:
                    raise ValueError('the image channel must be 2 or 3')
            else:
                start = np.random.randint(w - h)
                if len(image.shape) == 2:
                    img = image[:, start:(start + h)]
                elif len(image.shape) == 3:
                    img = image[:, start:(start + h), :]
                else:
                    raise ValueError('the image channel must be 2 or 3')
            return (resize(img, (self.img_size, self.img_size)) * 255).astype(np.uint8)


def data_loader(dataset_name, augment, batchsize, phase='train', img_size=256, flip=True):
    dataset = Dataset(dataset_name, augment, phase, img_size, flip)
    dataloader = DataLoader(dataset,
                            batch_size=batchsize,
                            shuffle=True)
    # return dataloader
    for i, data in enumerate(dataloader):
        if i*batchsize >= len(dataset):
            break
        yield data
