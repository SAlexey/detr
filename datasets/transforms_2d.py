import torch 
import torchvision.transforms.functional as TF
from random import random
from scipy import ndimage


class ToTensor(object):

    def __call__(self, image, mask):
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask


class RandomFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if self.p >= random():

            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if self.p >= random():

            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask


class RandomRotate(object):

    def __init__(self, p=0.5, angle=5):
        self.p = p
        self.angle = angle


    def __call__(self, image, mask):
        
        if self.p >= random():
            angle = torch.randint(-self.angle, +self.angle, (1,))

            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            
        return image, mask


class Rot90(object):

    def __init__(self, p=1, n=1):
        self.p=p 
        self.n=n

    def __call__(self, image, mask):

        if self.p >= random():

            image = torch.as_tensor(ndimage.rotate(image, 90*self.n, (1, 2)).copy(), dtype=image.dtype)
            mask = torch.as_tensor(ndimage.rotate(mask, 90*self.n, (0,1)).copy(), dtype=mask.dtype)

        return image, mask


class CenterCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, mask):

        image = TF.center_crop(image, self.output_size)
        mask = TF.center_crop(mask, self.output_size)

        return image, mask


class ZNormalize(object):

    def __call__(self, image, mask):
        image = image - image.mean()
        image = image / image.std()

        return image, mask
        

class Normalize(object):

    def __init__(self, mean=None, std=None):
        self.mean = mean 
        self.std = std

    def __call__(self, image, mask):
        
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, mask


class Compose(object):

    def __init__(self, t):
        self.t = t

    def __call__(self, image, mask):
        for t in self.t:
            image, mask = t(image, mask)

        return image, mask