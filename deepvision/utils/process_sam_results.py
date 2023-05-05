import numpy as np


def process_mask_generator(sam_results, opacity):
    if not sam_results:
        print("No annotations found.")
        return np.empty((0, 0, 0))
    results = []
    for ann in sam_results:
        m = ann["segmentation"]
        color_mask = np.random.random((1, 3))
        img = np.repeat(color_mask, m.shape[0] * m.shape[1], axis=0)
        img = np.reshape(img, (m.shape[0], m.shape[1], 3))
        results.append(np.dstack((img, m * opacity)))

    return np.array(results)


def process_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image
