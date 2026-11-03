from random import randint

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input


def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


class SRGANDataset(Dataset):
    def __init__(self, annotation_lines_LR, annotation_lines_NLR, annotation_lines_HR, data_path, lr_shape, nlr_shape, hr_shape):
        super(SRGANDataset, self).__init__()

        self.annotation_lines_LR = annotation_lines_LR
        self.annotation_lines_NLR = annotation_lines_NLR
        self.annotation_lines_HR = annotation_lines_HR
        self.data_path = data_path
        self.length = len(annotation_lines_LR)

        self.lr_shape = lr_shape
        self.nlr_shape = nlr_shape
        self.hr_shape = hr_shape

        if not (len(annotation_lines_LR) == len(annotation_lines_NLR) == len(annotation_lines_HR)):
            raise ValueError('LR, NLR and HR annotation lists must have equal lengths.')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        img_path_h = self.data_path + '/HR_npy/' + self.annotation_lines_HR[index]
        # F = self.annotation_lines_HR[index].replace("HR", "LR")
        # F = F.replace("SR", "LR")
        img_path_nl = self.data_path + '/NLR_npy/' + self.annotation_lines_NLR[index]
        img_path_l = self.data_path + '/LR_npy/' + self.annotation_lines_LR[index]
        # path = self.annotation_lines[index].split()[0]
        # image_origin = Image.open(img_path.replace("\n", ""))
        # if self.rand() < .5:
        #     img_h = self.get_random_data(image_origin, self.hr_shape)
        # else:
        #     img_h = self.random_crop(image_origin, self.hr_shape[1], self.hr_shape[0])
        # img_l = img_h.resize((self.lr_shape[1], self.lr_shape[0]), Image.BICUBIC)
        img_l = self.load_normalized(img_path_l.strip(), self.lr_shape)
        img_nl = self.load_normalized(img_path_nl.strip(), self.nlr_shape)
        img_h = self.load_normalized(img_path_h.strip(), self.hr_shape)
        # img_l = cvtColor((Image.open(img_path_l.replace("\n", ""))))
        # img_h = cvtColor((Image.open(img_path_h.replace("\n", ""))))

        # img_h = np.transpose(preprocess_input(np.array(img_h, dtype=np.float32)), [2, 0, 1])
        # img_l = np.transpose(preprocess_input(np.array(img_l, dtype=np.float32)), [2, 0, 1])

        return np.array(img_l), np.array(img_nl), np.array(img_h)

    def load_normalized(self, path, expected_shape):
        image = np.load(path, allow_pickle=False)
        if image.shape != tuple(expected_shape):
            raise ValueError('{}: expected shape {}, got {}'.format(path, expected_shape, image.shape))
        if not np.isfinite(image).all():
            raise ValueError('{}: input contains NaN or Inf'.format(path))
        return self.custom_normalize(image)

    def custom_normalize(self, img):
        # 单通道图像的自定义归一化函数
        img = np.asarray(img, dtype=np.float64)
        if not np.isfinite(img).all():
            raise ValueError('Input contains NaN or Inf before normalization.')
        min_value = img.min()
        max_value = img.max()
        if max_value - min_value <= 1e-12:
            img = np.zeros_like(img)
        else:
            img = (2 * (img - min_value) / (max_value - min_value)) - 1
        img = torch.from_numpy(img.astype(np.float32))
        img = img.unsqueeze(0)   # 添加批次维度
        return img.float()  # 确保数据类型为float

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            return image_data

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-15, 15)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return Image.fromarray(np.uint8(image_data))

    def random_crop(self, image, width, height):
        # --------------------------------------------#
        #   如果图像过小无法截取，先对图像进行放大
        # --------------------------------------------#
        if image.size[0] < self.hr_shape[1] or image.size[1] < self.hr_shape[0]:
            resized_width, resized_height = get_new_img_size(width, height, img_min_side=np.max(self.hr_shape))
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        # --------------------------------------------#
        #   随机截取一部分
        # --------------------------------------------#
        width1 = randint(0, image.size[0] - width)
        height1 = randint(0, image.size[1] - height)

        width2 = width1 + width
        height2 = height1 + height

        image = image.crop((width1, height1, width2, height2))
        return image


def SRGAN_dataset_collate(batch):
    images_l = []
    images_nl = []
    images_h = []
    for img_l, img_nl, img_h in batch:
        images_l.append(img_l)
        images_nl.append(img_nl)
        images_h.append(img_h)

    images_l = torch.from_numpy(np.array(images_l, np.float32))
    # images_l = torch.unsqueeze(images_l, 1)
    images_nl = torch.from_numpy(np.array(images_nl, np.float32))
    images_h = torch.from_numpy(np.array(images_h, np.float32))
    # images_h = torch.unsqueeze(images_h, 1)
    return images_l, images_nl, images_h
