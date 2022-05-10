import os
import sys
import torch
import numpy as np
import glob
import json
import random
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFile
from skimage import io, transform, color
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import functional as F


KEYPOINT_COLOR = (0, 255, 0) # Green
def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=2, show=False): # image input is denormalized range(0~1 ndarrays)
    if not isinstance(image, (np.ndarray, np.generic)):
        sys.exit('vis_keypoints inputs should be a numpy arrays!')
    image = image.copy()
    for (y, x) in keypoints:
        # print(f'(x,y) = {(x, y)}')
        cv2.circle(image, (np.float32(x), np.float32(y)), diameter, color, -1)
    # image = image*255
    if show:
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(image)
        plt.show()
    return image


class Rescale(object):
    """Rescale the image in a sample to a given size.
    cannot use transform.resize/pillow resize
    need to use cv2.resize to be consistent with the keypoint resize
    
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):        
        image, landmarks = sample['image'], sample['landmarks']
        assert isinstance(image, np.ndarray)
        assert isinstance(landmarks, np.ndarray)

        orig_h, orig_w = image.shape[:2]
        inp_h, inp_w = self.output_size[0], self.output_size[1]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, self.output_size, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        
        Rw = inp_w/orig_w
        Rh = inp_h/orig_h
        new_keypoints=[]
        for i, kp in enumerate(landmarks): # keypoints were stored as: (row/height/y, column/width/x)   
            x, y = kp[1], kp[0]
            resized_kpts = (np.around(Rh * y), np.around(Rw * x))
            new_keypoints.append(resized_kpts)
        landmarks = np.array(new_keypoints)


        return {'image': image, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors range(0~1)."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        assert isinstance(image, np.ndarray)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if image.ndim == 2:
            image = image[:, :, None]

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).contiguous().div(255)
        
        return {'image': image,
                'landmarks': torch.from_numpy(landmarks)}


class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        # _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        image, landmarks = sample['image'], sample['landmarks']
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return {'image': image, 'landmarks': landmarks} 

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"



class NormalizeKeypoints(object):
    """Normalize keypoints -1 ~ 1

    """

    def __init__(self, input_shape):
        
        self.input_shape = input_shape

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        landmarks = (2 * landmarks - 0)/(self.input_shape[0] - 0) - 1 # norm to -1 ~ 1
        # norm_resized_kpts = (2*(resized_kpts[0]-0)/(inp_h-0) - 1, 2*(resized_kpts[1]-0)/(inp_w-0) - 1) # norm to -1 ~ 1

        return {'image': image, 'landmarks': landmarks}


class FaceLandmarksDataset(Dataset):
    def __init__(self, root, input_shape):
        self.transform = transforms.Compose([
            Rescale(input_shape[-2:]),
            # transforms.ColorJitter(brightness=(0.5,1.5)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            NormalizeKeypoints(input_shape=input_shape[-2:])
        ])
        
        self.shape = input_shape[-2:]
        self.files_A = sorted(glob.glob(os.path.join(root,"Input") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root,"Target") + "/*.*"))

    def __getitem__(self, index):

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # image
        image = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB') # .convert('RGB') is for turning 1 channel to 3 channels
        image = np.array(image)

        # nose landmark
        landmarks_json = json.load(open(self.files_B[index % len(self.files_B)]))
        centernoselabels = landmarks_json['labels'][3]['mask']
        centernoselandmarks = [landmarks_json['landmarks']['points'][pts] for pts in centernoselabels]
        bottomnoselabels = landmarks_json['labels'][4]['mask']
        bottomnoselandmarks = [landmarks_json['landmarks']['points'][pts] for pts in bottomnoselabels]
        landmarks = centernoselandmarks + bottomnoselandmarks
        landmarks = np.array([landmarks]).astype('float').reshape(-1, 2)

                
        sample = {'image': image, 'landmarks': landmarks}
        transformed = self.transform(sample)  
         
        return {'image': transformed['image'], 'landmark': transformed['landmarks']}

    def __len__(self):
        return len(self.files_A)

