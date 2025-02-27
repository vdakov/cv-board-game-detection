import time
import base64
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === Configuration ===
website_url = "https://jkirschner.github.io/catan-randomizer/"
button_locator = (By.ID, "gen-map-button")
canvas_locator = (By.ID, "map-canvas")

# === Initialize the WebDriver ===
driver = webdriver.Chrome()  # Ensure chromedriver is installed and in your PATH

try:
    # Open the website
    driver.get(website_url)
    
    # Loop 1000 times to generate and download images
    for i in range(1000):
        # Wait until the button is clickable and click it
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(button_locator)
        )
        button.click()
        
        # Allow a short delay for the canvas to update
        time.sleep(0.1)
        
        # Wait until the canvas element is present
        canvas = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(canvas_locator)
        )
        
        # Use JavaScript to retrieve the canvas as a data URL (base64 encoded)
        canvas_data_url = driver.execute_script(
            "return arguments[0].toDataURL('image/png');", canvas
        )
        
        # The data URL has the format "data:image/png;base64,...."
        header, encoded = canvas_data_url.split(",", 1)
        
        # Decode the base64 image data
        image_data = base64.b64decode(encoded)
        
        # Save the image with a unique filename for each iteration
        download_path = f"../catan_data/mined_synthetic_images/canvas_image_{i}.png"
        with open(download_path, "wb") as file:
            file.write(image_data)
        
        print(f"Canvas image {i} downloaded successfully and saved as '{download_path}'.")
        
except Exception as e:
    print("An error occurred:", e)
    
finally:
    # Clean up and close the browser window
    driver.quit()
