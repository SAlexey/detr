# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
from scipy import ndimage
import torch
from torchio.data.subject import Subject
import torchvision.transforms as T
import torchvision.transforms.functional as F
import albumentations as A
import torchio as tio

from util.box_ops import box_xyxy_to_cxcywh, convert
from util.misc import interpolate


class ObjectSlice(tio.transforms.Transform):

    def __init__(self, *args, obj_id=1, img_key="image", map_key="label_map", **kwargs):
        super().__init__(*args, **kwargs)
        self.args_names = []
        self.obj_id = obj_id
        self.img_key = img_key
        self.map_key = map_key

    def apply_transform(self, subject: tio.Subject):

        label_map = subject[self.map_key].data.squeeze()

        obj = ndimage.find_objects(label_map, max_label=self.obj_id + 1)[self.obj_id-1]
        assert obj is not None

        sl = (obj[0].stop + obj[0].start) // 2 
        subject[self.img_key].data = subject[self.img_key].data[:, sl].unsqueeze(0)
        subject[self.map_key].data = subject[self.map_key].data[:, sl].unsqueeze(0)

        return subject


class LabelMapToBbox(tio.transforms.Transform):

    """
    Extracts object bounding boxes from a segmentation map using ndimage.find_objects

    writes output back into the subject object
    
    Parameters: 
        map_key: (string) key for the input segmentation (segmentation_map = subject[map_key])
        tgt_key: (string) key for the output labels
        box_key: (string) key for the output boxes
        box_fmt: (string) format specifier for the output boxes 
            choices are: xyxy (default), xxyy, xywh, ccwh 

    Notes:
        Additionally adds keys:
            num_oobjects: (list) a list containing an int - the number of bounding boxes found in the map 

    """

    def __init__(self, *args, max_label=0, map_key="label_map", tgt_key="labels", box_key="boxes", box_fmt="xyxy", **kwargs):
        super().__init__(*args, **kwargs)
        self.args_names = []
        self.box_key = box_key
        self.tgt_key = tgt_key
        self.map_key = map_key
        self.max_label = max_label
        self.box_fmt = box_fmt

        assert isinstance(self.label_mapping, dict)


    def apply_transform(self, subject: tio.Subject):
        label_map = subject[self.map_key][tio.DATA].squeeze() # remove single dim axes
        objects = ndimage.find_objects(label_map, max_label=self.max_label) 
        
        bboxes = []
        labels = []

        for label, slices in enumerate(objects): 
            if slices is not None:
                left, right = zip(*((xs.start, xs.stop) for xs in slices))
                bboxes.append(list(left + right))
                labels.append(self.label_mapping.get(label, label))

        subject["num_objects"] = len(bboxes)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        bboxes = convert(bboxes, "xyxy", self.box_fmt)

        subject[self.tgt_key] = labels
        subject[self.box_key] = bboxes

        return subject


class NormalizeBbox(tio.transforms.Transform):

    def __init__(self, *args, scale_fct=None, boxes_key="boxes", boxes_fmt="xyxy", **kwargs):
        super().__init__(*args, **kwargs)
        self.args_names = []
        self.scale_fct = scale_fct 
        self.boxes_key = boxes_key
        self.boxes_fmt = boxes_fmt


    def apply_transform(self, subject: tio.Subject):

        boxes = subject[self.boxes_key]

        assert boxes.size(-1) // 2 == len(self.scale_fct)

        if self.boxes_fmt != "xyxy": # convert to xyxy format for scaling
            boxes = convert(boxes, self.boxes_fmt, "xyxy")

        boxes = boxes / torch.as_tensor(self.scale_fct + self.scale_fct, dtype=torch.float32)

        if self.boxes_fmt != "xyxy": # convert back to the original format
            boxes = convert(boxes, "xyxy", self.boxes_fmt)

        subject[self.boxes_key] = boxes

        return subject


class LRFlip(tio.transforms.Transform):

    def __init__(self, *args, side="left", **kwargs):
        super().__init__(*args, **kwargs)
        self.args_names = []
        self.flip = tio.transforms.RandomFlip(axes=("LR",), flip_probability=1)

    def apply_transform(self, subject: tio.Subject):
        if subject["side"][0] == "left":
            subject = self.flip(subject)
        return subject


class SafeCrop(object):

    def __init__(self, width, height, p=0.5):
        self.t = A.Compose(
            [A.RandomSizedBBoxSafeCrop(width, height, erosion_rate=0.0, interpolation=1, always_apply=True, p=p)],
            bbox_params=A.BboxParams(format="coco", label_fields=["labels"])
        )

    def __call__(self, inputs, targets):
        transformed = self.t(image=inputs.numpy(), bboxes=[targets["boxes"]], labels=targets["labels"])
        targets["boxes"] = transformed["bboxes"]
        targets["labels"] = transformed["labels"]
        return transformed["image"], targets

# class NormalizeBbox(object):

#     def __init__(self, rows, cols):
#         self.rows = rows 
#         self.cols = cols

#     def __call__(self, inputs, target):
#         boxes = target["boxes"][0]
#         div = torch.tensor([self.cols, self.rows, self.cols, self.rows], dtype=torch.float32)
#         target["boxes"] = torch.as_tensor(boxes) / div
#         return inputs, target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
