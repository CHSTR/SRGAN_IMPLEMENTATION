import random

import cv2
import torch
from torch.utils.data import Dataset


class GenerateDataset(Dataset):
    def __init__(self, img_paths, h_hr=800, w_hr=800):
        self.img_paths = img_paths
        self.h_hr = h_hr
        self.w_hr = w_hr

    def __len__(self):
        return len(self.img_paths)

    def rescale(self, image):
        h, w = image.shape[:2]
        scale_factor = min(h, w) / self.h_hr
        if scale_factor < 1:
            if h < w:
                h, w = self.h_hr, w * scale_factor
            else:
                h, w = h * scale_factor, self.h_hr
            h, w = int(h + 0.5), int(w + 0.5)
            image = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
        return image

    def random_crop(self, image):
        h, w = image.shape[:2]
        x1 = abs(random.randint(0, w - self.w_hr))
        y1 = abs(random.randint(0, h - self.h_hr))
        x2 = x1 + self.w_hr
        y2 = y1 + self.h_hr

        return image[y1:y2, x1:x2]

    @staticmethod
    def normalize(high_res, low_res):
        high_res = (high_res / 255.0) * 2 - 1
        low_res = (low_res / 255.0) * 2 - 1

        return high_res, low_res

    def __getitem__(self, idx):
        high_res = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        high_res = self.rescale(high_res)
        high_res = self.random_crop(high_res)
        low_res = cv2.resize(high_res, (self.h_hr // 4, self.w_hr // 4), cv2.INTER_LINEAR)
        high_res, low_res = self.normalize(high_res, low_res)
        high_res = torch.from_numpy(high_res).permute(2, 0, 1)
        low_res = torch.from_numpy(low_res).permute(2, 0, 1)
        return low_res.float(), high_res.float()
