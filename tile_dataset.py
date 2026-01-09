import os
from PIL import Image
from tqdm import tqdm

def create_tiles():
    image_dir = "dataset/images"
    mask_dir = "dataset/masks"
    output_dir = "dataset/tiles"
    
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)
    
    images = os.listdir(image_dir)
    
    for img_name in tqdm(images, desc="Creating tiles"):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".tiff", ".tif"))
        
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        base_name = img_name.replace(".tiff", "")
        
        for row in range(3):
            for col in range(3):
                x = col * 500
                y = row * 500
                
                img_tile = image.crop((x, y, x + 500, y + 500))
                mask_tile = mask.crop((x, y, x + 500, y + 500))
                
                tile_name = f"{base_name}_r{row}_c{col}.png"
                img_tile.save(f"{output_dir}/images/{tile_name}")
                mask_tile.save(f"{output_dir}/masks/{tile_name}")
    
    print(f"Created {len(images) * 9} tiles")


if __name__ == "__main__":
    create_tiles()