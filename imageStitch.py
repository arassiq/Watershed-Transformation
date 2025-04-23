"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
def stitch_images(imageDir):
    pass

def main():
    inputDir = "/Users/aaronrassiq/Downloads/oakland_detailed_imagery"
    currentYearImage = "2023"
    images = []


    for root, dirs, files in os.walk(inputDir):
        for file in files:
            print(file[:13] + "\n")
            if currentYearImage in os.path.join(root, file):
                images.append(os.path.join(root.split('/')[-1], file))


    print(images)
            
if __name__ == "__main__":
    main()"""


import os
import cv2
import numpy as np

def extract_coords_from_path(path):
    try:
        cellStr = path.split('/')[-2]
        x_str = cellStr.split('_')[2]
        y_str = cellStr.split('_')[3]
        print(x_str, y_str)
        return int(x_str), int(y_str)
    except:
        return None, None

def stitch_images(image_paths, inputDir, grid_size=(16, 16)):
    # Initialize empty grid
    grid = [[None for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    
    for rel_path in image_paths:
        full_path = os.path.join(inputDir, rel_path)
        print(full_path)
        #print(f"Reading: {full_path}")
        x, y = extract_coords_from_path(rel_path)
        if x is not None and y is not None:
            img = cv2.imread(full_path)
            img = cv2.imread(full_path)
            if img is None:
                print(f"❌ cv2.imread failed on: {full_path}")
            else:
                print(f"✅ Loaded image at ({x},{y}) — shape: {img.shape}")

                # Resize if height is not 3500
                target_height = 3500
                if img.shape[0] != target_height:
                    target_width = int(img.shape[1] * (target_height / img.shape[0]))
                    img = cv2.resize(img, (target_width, target_height))
                    print(f"↪️ Resized to: {img.shape}")

                grid[x][y] = img
        else:
            print(f"⚠️ Skipped path (invalid x,y): {rel_path}")



    # Get reference shape
    max_height, max_width = 0, 0
    for row in grid:
        for img in row:
            if img is not None:
                h, w = img.shape[:2]
                max_height = max(max_height, h)
                max_width = max(max_width, w)

    if max_height == 0 or max_width == 0:
        raise ValueError("No valid images found to determine shape.")

    # Pad images to the max size
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if grid[x][y] is None:
                print(f"Warning: Missing image at ({x}, {y}), filling with black.")
                grid[x][y] = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            else:
                img = grid[x][y]
                h, w = img.shape[:2]
                if (h, w) != (max_height, max_width):
                    padded = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    padded[:h, :w] = img
                    grid[x][y] = padded

    # Fill missing cells with black image of same shape
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if grid[x][y] is None:
                print(f"Warning: Missing image at ({x}, {y}), filling with black.")
                grid[x][y] = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # Horizontally stack each row (iterate by y, top to bottom)
    stitched_rows = []
    for y in reversed(range(grid_size[1])):  # reversed to make (0,0) bottom-left
        row_imgs = [grid[x][y] for x in range(grid_size[0])]
        stitched_rows.append(np.hstack(row_imgs))

    full_image = np.vstack(stitched_rows)
    return full_image

def main():
    inputDir = "/Users/aaronrassiq/Downloads/oakland_detailed_imagery"
    currentYearImage = "2023"
    image_paths = []


    for root, dirs, files in os.walk(inputDir):
        for file in files:
            if currentYearImage in file and file.endswith(".png") and "sentinel" not in file.lower():
                rel_path = os.path.relpath(os.path.join(root, file), inputDir)
                if "oakland_cell_" in rel_path:
                    image_paths.append(rel_path)
    print(len(image_paths))

    full_image = stitch_images(image_paths, inputDir)
    print(f"Stitched image shape: {full_image.shape}")
    scale_factor = 0.25
    new_width = int(full_image.shape[1] * scale_factor)
    new_height = int(full_image.shape[0] * scale_factor)
    resized_image = cv2.resize(full_image, (new_width, new_height), interpolation=cv2.INTER_AREA)


    output_path = os.path.join(inputDir, "stitched_oakland_map.png")
    cv2.imwrite(output_path, resized_image)
    print(f"Saved stitched image to {output_path}")

if __name__ == "__main__":
    main()