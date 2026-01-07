import os

def find_mismatched_files(images_dir, masks_dir):
    # Get file names (without extensions) from both directories
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))}
    
    # Find files that exist in one directory but not the other
    images_not_in_masks = image_files - mask_files
    masks_not_in_images = mask_files - image_files
    
    # Print results
    if images_not_in_masks:
        print("Files in images directory but not in masks directory:")
        for file in images_not_in_masks:
            print(file)
    else:
        print("No files are missing in the masks directory.")
    
    if masks_not_in_images:
        print("\nFiles in masks directory but not in images directory:")
        for file in masks_not_in_images:
            print(file)
    else:
        print("No files are missing in the images directory.")

if __name__ == "__main__":
    images_dir = "dataset/images"
    masks_dir = "dataset/masks"
    find_mismatched_files(images_dir, masks_dir)