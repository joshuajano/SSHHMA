import cv2
import numpy as np
def denormalize_keypoints(keypoints, crop_size=224):
    # unnormalize to crop coords
    keypoints = 0.5 * crop_size * (keypoints + 1.0)
    return keypoints
def draw_keypoints_w_black_bg(x, y, crop_size=224, batch_size=32, seqlen=16):
    for i in range(2):
        for j in range(seqlen):
            gt = y[i, j]
            pred = x[i, j] 
            black_bg = np.zeros((crop_size, crop_size, 3))
            gt = denormalize_keypoints(gt, crop_size)
            pred= denormalize_keypoints(pred, crop_size)

            for k in range(gt.shape[0]):
                cv2.circle(black_bg, (int(pred[k][0]), int(pred[k][1])), 4, (255, 0, 0), -1)
                cv2.circle(black_bg, (int(gt[k][0]), int(gt[k][1])), 4, (0, 255, 0), -1)
            # savename_path = os.path.join(save_dir, f'{i}_{j}.jpg')
            # cv2.imwrite(savename_path, rend_img.astype(np.uint8))
            cv2.imwrite('test.jpg', black_bg.astype(np.uint8))
    pass