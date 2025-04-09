import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.feature import peak_local_max
from scipy import ndimage
import os
from skimage.segmentation import watershed

filename = '/Users/aaronrassiq/Desktop/Watershed-Transformation/tstImages/3-2-2024-300-300.jpg'

import os
if os.path.exists(filename):
    print(f"File found: {filename}")
else:
    print(f"Error: File not found at {filename}")

def detect_individual_trees(image_path, visualize=True):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    original = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 20, 20])
    upper_green = np.array([100, 255, 200])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    hsv_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
    mask_enhanced = cv2.inRange(hsv_enhanced, lower_green, upper_green)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_enhanced = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel, iterations=3)

    mask = cv2.bitwise_or(mask, mask_enhanced)

    dist_transform = ndimage.distance_transform_edt(mask)

    tree_centers = peak_local_max(dist_transform, min_distance=10,
                                 threshold_abs=5, exclude_border=False)

    markers = np.zeros_like(mask)
    for i, (x, y) in enumerate(tree_centers, start=1):
        markers[x, y] = i


    labels = watershed(-dist_transform, markers, mask=mask)
    #print(f"Shape of labels: {labels.shape}")

    '''nativeLabelArr = labels.tolist()

    labelPropArr = []
    treePresent = 0
    for row in nativeLabelArr:
        for pix in row:
            if pix > 0:
                treePresent += 1

    labelPropArr.append(f"%{round((treePresent/(300 * 300)) * 100, 2)}")

    print(labelPropArr)'''


    result = original.copy()
    tree_count = 0
    for label in range(1, np.max(labels) + 1):
        tree_pixels = np.where(labels == label)
        print(f"Tree Pixels {tree_pixels}")

        if len(tree_pixels[0]) > 0:
            min_x, max_x = np.min(tree_pixels[1]), np.max(tree_pixels[1])
            min_y, max_y = np.min(tree_pixels[0]), np.max(tree_pixels[0])

            if (max_x - min_x) > 5 and (max_y - min_y) > 5:
                #print(f"\nTree {tree_count + 1}:\t minX - maxX: ({min_x}, {max_x}) to minY - maxY{min_y}, {max_y})\n")
                cv2.rectangle(result, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                tree_count += 1

    print(f"Detected {tree_count} trees")

    if visualize:
        plt.figure(figsize=(20, 15))

        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('Green Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('Distance Transform')
        plt.imshow(dist_transform, cmap='jet')
        plt.axis('off')

        centers_vis = original.copy()
        for x, y in tree_centers:
            cv2.circle(centers_vis, (y, x), 5, (0, 0, 255), -1)

        plt.subplot(2, 3, 4)
        plt.title('Tree Centers')
        plt.imshow(cv2.cvtColor(centers_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        segmented = np.zeros_like(original)
        for label in range(1, np.max(labels) + 1):
            color = np.random.randint(0, 255, size=3).tolist()
            segmented[labels == label] = color

        plt.subplot(2, 3, 5)
        plt.title('Segmented Trees')
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title('Detected Trees')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return result

def tune_parameters(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    original = image.copy()

    from ipywidgets import interact, interactive, fixed, widgets

    def update_params(h_min, h_max, s_min, v_min, v_max, morph_size, morph_iter, min_distance, threshold):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([h_min, s_min, v_min])
        upper_green = np.array([h_max, 255, v_max])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((morph_size, morph_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

        dist_transform = ndimage.distance_transform_edt(mask)

        tree_centers = peak_local_max(dist_transform, min_distance=min_distance,
                                     threshold_abs=threshold, exclude_border=False)

        result = original.copy()
        for x, y in tree_centers:
            cv2.circle(result, (y, x), 5, (0, 0, 255), -1)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Green Mask')
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f'Detected Centers: {len(tree_centers)}')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    interact(update_params,
             h_min=widgets.IntSlider(min=20, max=100, step=1, value=30, description='H min:'),
             h_max=widgets.IntSlider(min=70, max=130, step=1, value=100, description='H max:'),
             s_min=widgets.IntSlider(min=0, max=100, step=1, value=20, description='S min:'),
             v_min=widgets.IntSlider(min=0, max=100, step=1, value=20, description='V min:'),
             v_max=widgets.IntSlider(min=100, max=255, step=5, value=200, description='V max:'),
             morph_size=widgets.IntSlider(min=1, max=7, step=2, value=3, description='Morph Size:'),
             morph_iter=widgets.IntSlider(min=1, max=5, step=1, value=2, description='Morph Iter:'),
             min_distance=widgets.IntSlider(min=5, max=50, step=1, value=10, description='Min Distance:'),
             threshold=widgets.IntSlider(min=1, max=20, step=1, value=5, description='Threshold:'))

print("\nProcessing the image...")
result = detect_individual_trees(filename)

#output_filename = f"detected_trees_oakland_test_2.jpg"
#cv2.imshow(result)
#cv2.imwrite(output_filename, result)
