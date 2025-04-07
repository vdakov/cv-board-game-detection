from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
import tensorflow as tf
import pickle
import os
import sys


def save_to_file(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def to_tf_datasets(ds_dict, output_path):
    hex_input = [x.numpy() for x in ds_dict["hex_tensor"]]
    hex_labels = ds_dict["img_number_label"]

    # Encode the hexes' labels and save them to file
    hex_encoder = LabelEncoder()
    y_hexagons_encoded = hex_encoder.fit_transform(hex_labels)

    hex_encoder_path = (
        f"{output_path}/label_encoder/label_encoder_numbers.pkl"
    )
    save_to_file(hex_encoder_path, hex_encoder)

    # Convert to TensorFlow tensors
    X_hex = tf.convert_to_tensor(hex_input, dtype=tf.float32)
    y_hexagons = tf.convert_to_tensor(y_hexagons_encoded, dtype=tf.int32)

    print("Compiled datasets; preparing to store")

    save_path_hexagons = f"{output_path}/synthetic_dataset_numbers.pkl"
    save_to_file(save_path_hexagons, (X_hex, y_hexagons))


def draw_number_plate(img, number, font_path):
    # Get image size
    width, height = img.size

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Define circle properties
    circle_radius = min(width, height) // 6  # Adjust size
    circle_center = (width // 2, height // 2)

    # Draw circle (beige color)
    circle_color = (255, 240, 175)  # Light beige color
    outline_color = (178, 142, 0)
    draw.ellipse(
        [
            (circle_center[0] - circle_radius, circle_center[1] - circle_radius),
            (circle_center[0] + circle_radius, circle_center[1] + circle_radius),
        ],
        fill=circle_color,
        outline=outline_color,
        width=9,
    )

    try:
        font_size = circle_radius  # Set font size based on circle size
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw the number in the center
    number_text = number
    text_size = draw.textbbox((0, 0), number_text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    text_position = (circle_center[0] - text_width // 2, circle_center[1] - text_height)

    # Draw text (black or bright red depending on the piece)
    if number_text == "8" or number_text == "6":
        fill_color = "red"
    else:
        fill_color = "black"

    draw.text(text_position, number_text, font=font, fill=fill_color)

    return img


def generate_tile_image(
    image_path,
    bg_path,
    final_img_shape,
    number=None,
    font_path=None,
    tile_type="desert",
):

    # Load the background image
    img = Image.open(image_path).convert("RGBA")

    # Desert tiles do not have a number on them
    if tile_type == "desert":
        bg = Image.open(bg_path).convert("RGBA")
        bg = bg.resize(img.size)
        combined = Image.alpha_composite(bg, img).convert("RGB")

        return combined

    img = draw_number_plate(img, number, font_path)

    bg = Image.open(bg_path).convert("RGBA")
    bg = bg.resize(img.size)

    combined = Image.alpha_composite(bg, img).convert("RGB")
    combined = combined.resize(final_img_shape)

    return combined


def img_to_tensor(img, img_size):
    transform = ToTensor()

    img_np = transform(img)

    # interpolate to ensure same size
    resized_img = interpolate(img_np.unsqueeze(0), img_size).squeeze(0).permute(1, 2, 0)

    return resized_img


if __name__ == "__main__":

    font_path = "C:/Windows/Fonts/georgia.ttf"  # add a path to your own font
    tile_bgs_path = "../catan-randomizer/images"
    output_img_path = "data/output/generated_synthetic_tiles_expanded"
    output_ds_path = "data/output/compiled_dataset"
    backgrounds_path1 = "data/input/tile_datasets/tile_backgrounds/type_1"
    backgrounds_path2 = "data/input/tile_datasets/tile_backgrounds/type_2"

    tile_types = ["brick", "desert", "sheep", "ore", "wheat", "lumber"]
    valid_numbers = ["2", "3", "4", "5", "6", "8", "9", "10", "11", "12"]

    # Determine size of input to tile detector
    img_size = (200, 200)

    final_dict = {
        "img_path": [],
        "hex_tensor": [],
        "img_number_label": [],
    }

    for tile in tile_types:

        backgrounds_path = backgrounds_path1

        backgrounds = os.listdir(backgrounds_path)

        bg_index = 0

        for background in backgrounds:
            bg_path = f"{backgrounds_path}/{background}"

            img_path = f"{tile_bgs_path}/{tile}_1.png"

            for number in valid_numbers:
                # Generate image
                img = generate_tile_image(
                    img_path, bg_path, img_size, number, font_path, tile
                )

                # Save image
                img.save(
                    f"{output_img_path}/{tile}_1_no_{number}_bg_{bg_index}.png"
                )

                final_img = img_to_tensor(img, img_size)

                number_label = '0' if tile == "desert" else number

                # Save relevant information to dictionary
                final_dict["img_path"].append(img_path)
                final_dict["hex_tensor"].append(final_img)
                final_dict["img_number_label"].append(number_label)

                bg_index += 1

        print(f"Images generated for: {tile}")

    print("Synthetic samples obtained")

    to_tf_datasets(final_dict, output_ds_path)

    print("Synthetic samples saved to pickle dataset")
