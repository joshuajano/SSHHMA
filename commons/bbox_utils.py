import numpy as np
import torch
def bbox_to_center_scale(bbox, dset_scale_factor=1.0, ref_bbox_size=200):
    if bbox is None:
        return None, None, None
    bbox = bbox.reshape(-1)
    bbox_size = dset_scale_factor * max(
        bbox[2] - bbox[0], bbox[3] - bbox[1])
    scale = bbox_size / ref_bbox_size
    center = np.stack(
        [(bbox[0] + bbox[2]) * 0.5,
         (bbox[1] + bbox[3]) * 0.5]).astype(np.float32)
    return center, scale, bbox_size
def bbox_to_center_scale_wcoco(bbox, dset_scale_factor=1.0, ref_bbox_size=200):
    if bbox is None:
        return None, None, None
    bbox = bbox.reshape(-1)
    bbox_size = dset_scale_factor * max(
        bbox[2], bbox[3])
    scale = bbox_size / ref_bbox_size
    center = np.stack(
        [bbox[0] + bbox[2]/2,
         bbox[1] + bbox[3]/2]).astype(np.float32)
    return center, scale, bbox_size
def bbox_area(bbox):
    if torch.is_tensor(bbox):
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = torch.split(bbox.reshape(-1, 4), 1, dim=1)
        return torch.abs((xmax - xmin) * (ymax - ymin)).squeeze(dim=-1)
    else:
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = np.split(bbox.reshape(-1, 4), 4, axis=1)
        return np.abs((xmax - xmin) * (ymax - ymin))
def keyps_to_bbox(keypoints, conf, img_size=None, clip_to_img=False,
                  min_valid_keypoints=6, scale=1.0):
    valid_keypoints = keypoints[conf > 0]
    if len(valid_keypoints) < min_valid_keypoints:
        return None

    xmin, ymin = np.amin(valid_keypoints, axis=0)
    xmax, ymax = np.amax(valid_keypoints, axis=0)
    # Clip to the image
    if img_size is not None and clip_to_img:
        H, W, _ = img_size
        xmin = np.clip(xmin, 0, W)
        xmax = np.clip(xmax, 0, W)
        ymin = np.clip(ymin, 0, H)
        ymax = np.clip(ymax, 0, H)

    width = (xmax - xmin) * scale
    height = (ymax - ymin) * scale

    x_center = 0.5 * (xmax + xmin)
    y_center = 0.5 * (ymax + ymin)
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.stack([xmin, ymin, xmax, ymax], axis=0).astype(np.float32)
    if bbox_area(bbox) > 0:
        return bbox
    else:
        return None