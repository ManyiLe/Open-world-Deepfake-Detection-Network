import cv2
import numpy as np
import os  
import glob 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter

class CMPDataset(Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.data = self.__load_data__()
        self.rz_dict = {'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
            'nearest': Image.NEAREST}
        self.jpeg_dict = {'cv2': self.cv2_jpg, 'pil': self.pil_jpg}
    
    def __getitem__(self, index):
        data = self.preprocess(self.data[index]) 
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __load_data__(self):
        dataset = []
        tf = None
        for cls in self.opt.classes:
            root = os.path.join(self.opt.dataroot, cls)
            for root, dirs, files in os.walk(root): 
                if '0' in root.split('/')[-1]:
                    tf = 0
                elif '1' in root.split('/')[-1]:
                    tf = 1
                for file in files:  
                    file_path = os.path.join(root, file)  
                    dataset.append([file_path, tf, random() < self.opt.jpg_prob])  
        return dataset
      
    #design for testing model's ability especially in compression image
    def lower_quality(self,img, compress_val):
        img = np.array(img)
        img = self.cv2_jpg(img,compress_val)
        return Image.fromarray(img)

    def data_augment(self,img,cmp):
        img = np.array(img)

        if random() < self.opt.blur_prob:
            sig = self.sample_continuous(self.opt.blur_sig)
            self.gaussian_blur(img, sig)

        if cmp:
            method = self.sample_discrete(self.opt.jpg_method)
            qual = self.sample_discrete(self.opt.jpg_qual)
            if self.opt.isTrain:
                img = self.jpeg_from_key(img, 60, method) #C40
            else:
                img = self.jpeg_from_key(img, qual, method) #Quality-agagnostic test

        return Image.fromarray(img)

    def sample_continuous(self,s):
        if len(s) == 1:
            return s[0]
        if len(s) == 2:
            rg = s[1] - s[0]
            return random() * rg + s[0]
        raise ValueError("Length of iterable s should be 1 or 2.")

    def sample_discrete(self,s):
        if len(s) == 1:
            return s[0]
        return choice(s)


    def gaussian_blur(self,img, sigma):
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


    def cv2_jpg(self,img, compress_val):
        img_cv2 = img[:,:,::-1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
        result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg[:,:,::-1]


    def pil_jpg(self,img, compress_val):
        out = BytesIO()
        img = Image.fromarray(img)
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        out.close()
        return img

    def jpeg_from_key(self,img, compress_val, key):
        method = self.jpeg_dict[key]
        return method(img, compress_val)

    def custom_resize(self,img):
        interp = self.sample_discrete(self.opt.rz_interp)
        return TF.resize(img, (self.opt.loadSize, self.opt.loadSize), interpolation=self.rz_dict[interp])

    def preprocess(self, data):
        image_path, tf_label, cmp_label = data
        image = Image.open(image_path)

        if image.mode == 'L':  
            image = image.convert('RGB')

        if self.opt.isTrain:
            crop_func = transforms.RandomCrop(self.opt.cropSize)
        elif self.opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(self.opt.cropSize)

        if self.opt.isTrain and not self.opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)

        if not self.opt.isTrain and self.opt.no_resize:
            # print('='*20,'No Resize')
            rz_func = transforms.Lambda(lambda img: img)
        else:
            # print('='*20,' Resize')
            rz_func = transforms.Lambda(lambda img: self.custom_resize(img))
        
        aug_func = transforms.Lambda(lambda img: self.data_augment(img, cmp_label))

        if self.opt.cmp:
            cmp_func = transforms.Lambda(lambda img: self.lower_quality(img,self.opt.cmp_arg))
        else:
            cmp_func = transforms.Lambda(lambda img: img)

        no_aug_transform =  transforms.Compose([
                    rz_func,
                    crop_func,
                    flip_func,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        transform = transforms.Compose([
                    rz_func,
                    aug_func,
                    crop_func,
                    flip_func,
                    cmp_func,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        return [transform(image), no_aug_transform(image), tf_label, cmp_label]
