import os
import cv2
import ast
import io
import torch
import ujson as json
import numpy as np
import random
from script.resize import imresize_np
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AdobeDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.load_data()
        self.allframe = True

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        def read(name):
            data_list = []
            with open("data/adobe240fps_folder_{}.txt".format(name)) as f:
                data = f.readlines()
                for l in data:
                    l = l.strip('\n')
                    path = '/data/adobe240/frame/{}/{}'.format(name, l)
                    interval = 1
                    if name != 'train':
                        interval = 9
                    for i in range(0, len(os.listdir(path)) - 9, interval):
                        data_tuple = []
                        for j in range(9):
                            data_tuple.append('{}/{}.png'.format(path, i+j))
                        data_list.append(data_tuple)
            return data_list
        self.meta_data = read(self.dataset_name)
        self.nr_sample = len(self.meta_data)        

    def aug(self, imgs, h, w):
        ih, iw, _ = imgs[0].shape        
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+h, y:y+w, :]
        return imgs

    def read(self, x):
        return cv2.imread(x)
    
    def getimg(self, index, training=False):
        data = self.meta_data[index]
        if not training:
            imgs = []
            if self.allframe:
                for i in range(9):
                    imgs.append(self.read(data[i]))
            else:
                imgs.append(self.read(data[0]))
                imgs.append(self.read(data[4]))
                imgs.append(self.read(data[8]))
            step = 0.5
        else:
            ind = [1, 2, 3, 4, 5, 6, 7]   
            random.shuffle(ind)
            ind[1] = ind[0]
            ind[0] = 0
            ind[2] = 8
            img0 = self.read(data[ind[0]])
            gt = self.read(data[ind[1]])
            img1 = self.read(data[ind[2]])
            step = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0])
            imgs = [img0, gt, img1]            
        return imgs, step

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        # For details on unsharp masking, see:
        # https://en.wikipedia.org/wiki/Unsharp_masking
        # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened
            
    def __getitem__(self, index):
        if self.dataset_name == 'train':
            imgs, timestep = self.getimg(index, True)
            if random.uniform(0, 1) < 0.5:
                imgs = self.aug(imgs, 128, 128)
            else:
                imgs = self.aug(imgs, 256, 256)
                for i in range(len(imgs)):
                    imgs[i] = cv2.resize(imgs[i], (128, 128), interpolation=cv2.INTER_CUBIC)
            for i in range(3):
                imgs[i] = self.unsharp_mask(imgs[i])
            img0, gt, img1 = imgs
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            imgs = [img0.copy(), gt.copy(), img1.copy()]
        else:
            imgs, timestep = self.getimg(index, training=False)
        imgs = torch.from_numpy(np.concatenate(imgs.copy(), 2)).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        
        # for img in imgs:
        #    lowres.append(cv2.resize(imresize_np(img, 0.25), (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
        # We synthesize data like RealESRGAN
        
        lowres = imgs.clone()
        lowres = lowres.reshape(3, 3, 128, 128).permute(0, 2, 3, 1).numpy()
        origin_lowres = [lowres[0].copy(), lowres[1].copy(), lowres[2].copy()]
        lowres = [lowres[0], lowres[1], lowres[2]]
        if random.uniform(0, 1) < 0.3:
            lowres[i] = cv2.resize(lowres[i], (112, 112))
            lowres[i] = cv2.resize(lowres[i], (128, 128))
        if random.uniform(0, 1) < 0.3:
            quality_factor = random.randint(60, 90)
            params = [cv2.IMWRITE_JPEG_QUALITY, quality_factor]
            for i in range(3):
                lowres[i] = cv2.imencode(".jpg", lowres[i], params)[1]
                lowres[i] = cv2.imdecode(np.frombuffer(lowres[i], np.uint8), cv2.IMREAD_COLOR)
        def addnoise(image):
            mean = 0
            sigma = random.uniform(0, 0.05)
            gauss = np.random.normal(mean,sigma,image.shape)
            if random.uniform(0, 1) < 0.5:
                gauss = np.mean(gauss, 2, keepdims=True)
            image = np.clip(image/255.+gauss, 0, 1) * 255
            return image.astype('uint8')
        if random.uniform(0, 1) < 0.3:
            for i in range(3):
                lowres[i] = cv2.GaussianBlur(lowres[i], (5, 5), 0)
        if random.uniform(0, 1) < 0.3:
            for i in range(3):
                lowres[i] = cv2.GaussianBlur(lowres[i], (7, 7), 0)
        if random.uniform(0, 1) < 0.3:
            for i in range(3):
                lowres[i] = addnoise(lowres[i])
        if random.uniform(0, 1) < 0.3:
            quality_factor = random.randint(60, 90)
            params = [cv2.IMWRITE_JPEG_QUALITY, quality_factor]
            for i in range(3):
                lowres[i] = cv2.imencode(".jpg", lowres[i], params)[1]
                lowres[i] = cv2.imdecode(np.frombuffer(lowres[i], np.uint8), cv2.IMREAD_COLOR)
        if random.uniform(0, 1) < 0.3:
            p = random.uniform(0.5, 1)
            for i in range(3):
                lowres[i] = ((lowres[i]/255. * p + origin_lowres[i]/255. * (1 - p)) * 255).astype('uint8')
        lowres = torch.from_numpy(np.concatenate(lowres, 2)).permute(2, 0, 1)
        return imgs, lowres, timestep
    
if __name__ == '__main__':
    ds = DataLoader(AdobeDataset('train'))
