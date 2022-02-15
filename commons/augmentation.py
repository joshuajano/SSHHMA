import numpy as np
import random
def get_aug_config( scale_factor = 0.25, 
                    rot_factor = 30,
                    color_factor = 0.2
):
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), 
                    random.uniform(c_low, c_up), 
                    random.uniform(c_low, c_up)])
    if np.random.uniform() <= 0.6:
        rot = 0
    return scale, rot, color_scale