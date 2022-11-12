# utils.py
import matplotlib.pyplot as plt
import numpy as np


def draw_mask(img, mask):
    """Draws mask on an image.
    """
    mask = mask.astype(np.uint8)
    img = np.copy(img)
    h, w = img.shape[:2]
    div_factor = 255 / np.max(img)
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1:
                img[i, j] = np.array([247, 86, 124], dtype=np.float32) / div_factor
            elif mask[i, j] == 2:
                img[i, j] = np.array([153, 225, 217], dtype=np.float32) / div_factor
            elif mask[i, j] == 3:
                img[i, j] = np.array([238, 227, 171], dtype=np.float32) / div_factor
    return img

def draw_mask_comparsion(img, mask, mask_pred):
    """Draws mask on an image with ground truth and prediction.
    """
    h, w= img.shape[:2]
    img = np.repeat(img.reshape(h, w, 1), 3, axis=2)
    annotated = draw_mask(img, mask)
    annotated_pred = draw_mask(img, mask_pred)
    return (img, annotated, annotated_pred)

def save_result(img, annotated, annotated_pred, filename):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(1, 3, 2)
    plt.imshow(annotated)
    plt.title('Ground Truth')
    plt.subplot(1, 3, 3)
    plt.imshow(annotated_pred)
    plt.title('Prediction')
    plt.savefig(filename)

if __name__ == '__main__':
    images = np.load('downloads/task2_2D_4classtestimages.npy')
    masks = np.load('downloads/task2_2D_4classtestlabels.npy')
    img = images[0]
    mask = masks[0]
    mask_pred = masks[0]
    h, w= img.shape
    img = np.repeat(img.reshape(h, w, 1), 3, axis=2)
    annotated = draw_mask(img, mask)
    annotated_pred = draw_mask(img, mask_pred)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(1, 3, 2)
    plt.imshow(annotated)
    plt.title('Ground Truth')
    plt.subplot(1, 3, 3)
    plt.imshow(annotated_pred)
    plt.title('Prediction')
    plt.savefig('annotated.png')