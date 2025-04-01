import os
import random
from PIL import Image, ImageFilter, ImageEnhance

def blend_with_random_background(main_image, background_dir, blur_radius=1, brightness_factor=1.2):
    background_images = [f for f in os.listdir(background_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not background_images:
        raise ValueError("No background images found in the specified directory.")
    
    background_image_path = os.path.join(background_dir, random.choice(background_images))
    background_image = Image.open(background_image_path).convert("RGBA")
    background_image = background_image.resize(main_image.size)

    
    # Ensure main image is in RGBA mode
    main_image = main_image.convert("RGBA")
    
    # Composite the images, preserving transparency
    blended_image = Image.new("RGBA", main_image.size)
    blended_image.paste(background_image, (0, 0))
    blended_image.paste(main_image, (0, 0), main_image)  # Use alpha channel for blending

        
    # Apply blur and brightness adjustments **only to the background**
    blended_image = blended_image.filter(ImageFilter.GaussianBlur(blur_radius))
    enhancer = ImageEnhance.Brightness(blended_image)
    blended_image = enhancer.enhance(brightness_factor)
    
    return blended_image



