import time
import base64
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
from io import BytesIO
import random
import os
import csv
from pre_processing.background_blend import blend_with_random_background
import argparse 
import json

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--raw_output_dir", help="Input the path to the directory where the raw synthetic catan images will be stored.", type=str, default="data/full/mined_synthetic_boards")
    parser.add_argument("--blended_output_dir", help="Input the path to the directory where the synthetic catan, blended with other synthetic images will be stored.", type=str, default="data/full/mined_synthetic_boards_blended")
    parser.add_argument("--num_images", help="Enter the amout of images you wanna create", type=int, default=2000)
    args = parser.parse_args()
    return args


# === Global Configuration - Used for the Selenium Mining ===
website_url = "https://jkirschner.github.io/catan-randomizer/"
button_locator = (By.ID, "gen-map-button")
canvas_locator = (By.ID, "map-canvas")

# === Initialize the WebDriver ===
driver = webdriver.Chrome()  # Ensure chromedriver is installed and in your PATH


def write_csvs(raw_output_dir, blended_output_dir):
    raw_csv_path = os.path.join(raw_output_dir, "bboxes.csv")
    blended_csv_path = os.path.join(blended_output_dir, "bboxes.csv")
    with open(raw_csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "x_min", "y_min", "x_max", "y_max"])
    with open(blended_csv_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "x_min", "y_min", "x_max", "y_max"])
    return raw_csv_path, blended_csv_path


def mine_boards(raw_output_dir, blended_output_dir, num_images):
    bbox_data = {"raw": {}, "blended": {}}
    try:
        driver.get(website_url)
        raw_csv_path, blended_csv_path = write_csvs(raw_output_dir, blended_output_dir)
        
        
        for i in range(num_images):
            raw_image_path = os.path.join(raw_output_dir, f"canvas_image_{i}.png")
            # Wait until the button is clickable and click it
            button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(button_locator)
            )
            button.click()
            
            # Allow a short delay for the canvas to update
            time.sleep(0.1)
            canvas = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(canvas_locator)
            )
            canvas_data_url = driver.execute_script(
                "return arguments[0].toDataURL('image/png');", canvas
            )
            
            # The data URL has the format "data:image/png;base64,...."
            header, encoded = canvas_data_url.split(",", 1)
            
            # Decode the base64 image data
            image_data = base64.b64decode(encoded)
            main_image = Image.open(BytesIO(image_data)).convert("RGBA")
            main_image_size = list(main_image.size)
            
            background_size = (512, 512) 
            background = Image.new("RGBA", background_size, (0, 0, 0, 0))  # Transparent background
            x_offset, y_offset  = int(background_size[0] - main_image_size[0]) // 2, int(background_size[1] - main_image_size[1]) // 2
            background.paste(main_image, (x_offset, y_offset), main_image)
            background.save(raw_image_path)


            bbox_data["raw"][f"canvas_image_{i}.png"] = {
                "x_min": x_offset,
                "y_min": y_offset,
                "x_max": x_offset + main_image_size[0],
                "y_max": y_offset + main_image_size[1]
            }

            background = Image.new("RGBA", background_size, (0, 0, 0, 0))  # Transparent background
        
            # Define a random scale factor (between 0.5x and 1x for variation)
            scale_factor = random.uniform(0.5, 0.8)

            # Compute new image size while maintaining aspect ratio
            new_width = int(background_size[0] * scale_factor)
            new_height = int(background_size[1] * scale_factor)
            new_width = min(new_width, background_size[0])
            new_height = min(new_height, background_size[1])
            main_image = main_image.resize((new_width, new_height), Image.LANCZOS)
            main_image_size = list(main_image.size)
            
            
            
            # Calculate the position to paste the image (centered)
            x_offset, y_offset  = int(background_size[0] - main_image_size[0]) // 2, int(background_size[1] - main_image_size[1]) // 2


            # Paste the main image onto the background
            background.paste(main_image, (x_offset, y_offset), main_image)
            blended_image = blend_with_random_background(background, "data/tables")
            blended_image_name = f"canvas_image_{i}.png"
            blended_image_path = os.path.join(blended_output_dir, blended_image_name)
            blended_image.save(blended_image_path)

            bbox_data["blended"][blended_image_name] = {
                "x_min": x_offset,
                "y_min": y_offset + new_height // 10,
                "x_max": x_offset + new_width,
                "y_max": y_offset + new_height - new_height // 10
            }

            print(f"Image {i} processed successfully.")

    finally:    
        with open(os.path.join(blended_output_dir, "bboxes.json"), "w") as f:
            json.dump(bbox_data, f, indent=4)
        driver.quit()
                
    except Exception as e:
            print("An error occurred:", e)

    finally:    
        # Clean up and close the browser window
        driver.quit()


    


if __name__ == "__main__":
    args = parse_args()
    raw_output_dir = args.raw_output_dir
    blended_output_dir = args.blended_output_dir
    num_images = args.num_images
    # Ensure directories exist
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(blended_output_dir, exist_ok=True)

    mine_boards(raw_output_dir, blended_output_dir, num_images)


