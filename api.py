import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.feature import peak_local_max
from scipy import ndimage
import os
from skimage.segmentation import watershed

filename = '/Users/aaronrassiq/Desktop/Watershed-Transformation/tstImages/AHLV3043.jpg'

class treeDetection:
    def __init__(self):
        pass

    def detect_individual_trees(self, image_path, visualize):
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

        print(f"Mask Shape: {mask.shape}")

        totalPixels = mask.shape[0] * mask.shape[1]

        mask_pixels = np.count_nonzero(mask)
        coveragePercent = mask_pixels/totalPixels
        print(f"Coverage Percentage: {coveragePercent * 100:.2f}%")
        
        
        treeArea = 0
        '''    for msk in mask:
            print(f"Mask: {msk}")
            for pixel in msk:
                if pixel > 0:
                    treeArea += 1'''

        #print(f"Mask Shape: {mask.shape}")
        #print(f"Areas of Trees: {treeArea / (720 * 1280)}")


        dist_transform = ndimage.distance_transform_edt(mask)

        tree_centers = peak_local_max(dist_transform, min_distance=10,
                                    threshold_abs=5, exclude_border=False)

        markers = np.zeros_like(mask, dtype=np.int32)
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
            #print(f"Tree Pixels {tree_pixels}")

            if len(tree_pixels[0]) > 0:
                min_x, max_x = np.min(tree_pixels[1]), np.max(tree_pixels[1])
                min_y, max_y = np.min(tree_pixels[0]), np.max(tree_pixels[0])

                if (max_x - min_x) > 0 and (max_y - min_y) > 0:
                    #print(f"\nTree {tree_count + 1}:\t minX - maxX: ({min_x}, {max_x}) to minY - maxY{min_y}, {max_y})\n")
                    cv2.rectangle(result, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                    tree_count += 1


        color_mask = np.zeros_like(original)
        color_mask[mask > 0] = [0, 255, 0]  # green highlight

        # Blend the original image with the color mask where mask is active
        alpha = 0.4  # transparency factor
        highlighted = original.copy()
        highlighted = cv2.addWeighted(color_mask, alpha, highlighted, 1 - alpha, 0)

        # Replace the result with the blended version where trees were detected
        result = highlighted.copy()

        # Convert from BGR (OpenCV default) to RGB for matplotlib
        '''plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Overlayed Tree Mask')
        plt.axis('off')
        plt.show()'''

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

        return coveragePercent, result, original

    def tune_parameters(self,image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        original = image.copy()

        from ipywidgets import interact, interactive, fixed, widgets

        def update_params( h_min, h_max, s_min, v_min, v_max, morph_size, morph_iter, min_distance, threshold):
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
        
def overlay_and_save(original_img_path, mask, output_dir, filename):
    # Load original image
    image = cv2.imread(original_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # If you're using matplotlib or PIL

    # Resize mask if necessary
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Convert binary mask to 3-channel
    if len(mask.shape) == 2:
        mask_rgb = np.stack([mask * 255, np.zeros_like(mask), np.zeros_like(mask)], axis=2)
    else:
        mask_rgb = mask

    # Overlay mask (make it transparent)
    overlayed = cv2.addWeighted(image, 0.7, mask_rgb.astype(np.uint8), 0.3, 0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save image
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))

    print(f"Saved overlay result to {out_path}")

def main():
    tree_detector = treeDetection()
    img2024, img2023, img2022, img2021, img2020, img2019, im2018, img2017, img2016 = [], [], [], [], []

    im = "/Users/aaronrassiq/Desktop/Watershed-Transformation/tstImages/3-2-2024-300-300.jpg"
    output_dir = "/Users/aaronrassiq/Desktop/Watershed-Transformation/resultImageDir"
    if os.path.doesntexist(output_dir):
        os.makedirs(output_dir)

    coveragePercent, result = tree_detector.detect_individual_trees(im, visualize=True)

    for root, dirs, files in os.walk("/Users/aaronrassiq/Downloads/Oakland Images"):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                filename = os.path.join(root, file)

                if "2021" in filename:
                    img2021.append(filename)
                elif "2022" in filename:
                    img2022.append(filename)
                elif "2023" in filename:
                    img2023.append(filename)
                elif "2024" in filename:
                    img2024.append(filename)
              

                    

                    '''coveragePercent, result = tree_detector.detect_individual_trees(filename, visualize=False)
                    print(f"Coverage Percentage: {coveragePercent * 100:.2f}%")
                    output_filename = f"detected_trees_{file}"
                    cv2.imwrite(output_filename, result)'''

    print(f"2021: {len(img2021)} images")
    print(f"2022: {len(img2022)} images")
    print(f"2023: {len(img2023)} images")
    print(f"2024: {len(img2024)} images")
    print(f"2025: {len(img2025)} images")

    bool2025to2021 = False

    if bool2025to2021:
            

        percentage2021, percentage2025 = 0, 0

        total_green_pixels_2021 = 0
        total_pixels_2021 = 0

        for im in img2021:
            coveragePercent, result, ogImage= tree_detector.detect_individual_trees(im, visualize=False)

            image = cv2.imread(im)
            h, w = image.shape[:2]
            total_pixels_2021 += h * w
            total_green_pixels_2021 += coveragePercent * h * w

            overlay_and_save(ogImage, result, output_dir, im)
            # plot the result on the ogImage and save it to dir

        
        total_green_pixels_2025 = 0
        total_pixels_2025 = 0
        
        for im in img2025:
            coveragePercent, result, ogImage = tree_detector.detect_individual_trees(im, visualize=False)

            image = cv2.imread(im)
            h, w = image.shape[:2]
            total_pixels_2025 += h * w
            total_green_pixels_2025 += coveragePercent * h * w
            overlay_and_save(ogImage, result, output_dir, im)
        

        total_percent_2021 = (total_green_pixels_2021 / total_pixels_2021) * 100
        print(f"Total green coverage in 2021: {total_percent_2021:.2f}%")
        total_percent_2025 = (total_green_pixels_2025 / total_pixels_2025) * 100
        print(f"Total green coverage in 2025: {total_percent_2025:.2f}%")
        

    

    #coveragePercent, result = tree_detector.detect_individual_trees(filename, visualize=False)


    
if __name__ == "__main__":
    main()


#output_filename = f"detected_trees_oakland_test_2.jpg"
#cv2.imshow(result)
#cv2.imwrite(output_filename, result)
