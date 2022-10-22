import matplotlib.pyplot as plt
import numpy as np
FIG_PATH = 'figs'
def draw_mask(img, mask):
    """Draws mask on an image.
    """
    mask = mask.astype(np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1:
                img[i, j] = (0, 0, 255)
            elif mask[i, j] == 2:
                img[i, j] = (0, 255, 0)
            elif mask[i, j] == 3:
                img[i, j] = (255, 0, 0)
    return img

def draw_mask_for_list(imgs, masks, masks_pred):
    """Draws mask on a list of images.
    """
    img = imgs.numpy()[0, 0]
    mask = masks.numpy()[0, 0]
    mask_pred = masks_pred.numpy()[0]
    plt.figure()
    plt.imshow((mask_pred * 50).astype(np.uint8), cmap='gray')
    plt.savefig('mask_pred.png')
    return None