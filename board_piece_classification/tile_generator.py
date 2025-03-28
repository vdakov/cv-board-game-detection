from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
import tensorflow as tf
import pickle
import os
import sys

def save_to_file(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def preprocess_image(img, zoom, final_size):
    w, h = img.size

    # First zoom to the center of the image
    x = w / 2
    y = h / 2

    zoom2 = zoom * 2

    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))

    img = img.resize(final_size)

    # Then keep the number only (so anything that is either black or bright red)
    # and convert the rest to white
    img = img.convert('RGB')

    img.show()
    return img

def to_tf_datasets(ds_dict, output_path):
    hex_input = [x.numpy() for x in ds_dict['hex_tensor']]
    number_input = [x.numpy() for x in ds_dict['number_tensor']]
    hex_labels = ds_dict['img_label_hexagon']
    no_labels = ds_dict['img_label_number']

    # Encode the hexes' labels and save them to file
    hex_encoder = LabelEncoder()
    y_hexagons_encoded = hex_encoder.fit_transform(hex_labels)
    hex_encoder_path = f'{output_path}/label_encoder/label_encoder_hexagons.pkl'
    save_to_file(hex_encoder_path, hex_encoder)

    # Encode the number labels and save them to file
    no_encoder = LabelEncoder()
    y_numbers_encoded = no_encoder.fit_transform(no_labels)
    number_encoder_path = f'{output_path}/label_encoder/label_encoder_numbers.pkl'
    save_to_file(number_encoder_path, no_encoder)

    # Convert to TensorFlow tensors
    X_hex = tf.convert_to_tensor(hex_input, dtype=tf.float32)
    X_no = tf.convert_to_tensor(number_input, dtype=tf.float32)
    y_hexagons = tf.convert_to_tensor(y_hexagons_encoded, dtype=tf.int32)
    y_numbers = tf.convert_to_tensor(y_numbers_encoded, dtype=tf.int32)

    print('Compiled datasets; preparing to store')

    save_path_hexagons = f'{output_path}/synthetic_dataset_hexagons.pkl'
    save_path_numbers = f'{output_path}/synthetic_dataset_numbers.pkl'

    save_to_file(save_path_hexagons, (X_hex, y_hexagons))
    save_to_file(save_path_numbers, (X_no, y_numbers))


def draw_number_plate(img, number, font):
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
        width=9
    )

    try:
        font_size = circle_radius  # Set font size based on circle size
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw the number in the center
    number_text = number
    text_size = draw.textbbox((0, 0), number_text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    text_position = (circle_center[0] - text_width // 2, circle_center[1] - text_height * 1.5)

    # Draw text (black or bright red depending on the piece)
    if number_text == '8' or number_text == '6':
        fill_color = 'red'
    else:
        fill_color = 'black'

    draw.text(text_position, number_text, font=font, fill=fill_color)

    # Draw the dots underneath the number
    # Calculate the number of dots
    # Formula taken from Alex Beals' Catan Board Generator
    dot_count = 6 - abs(int(number) - 7) + 1
    dot_radius = circle_radius // 12
    dot_spacing = dot_radius * 2.25

    for i in range(dot_count):
        dot_x = circle_center[0] - (dot_count - 1) * dot_spacing // 2 + i * dot_spacing
        dot_y = circle_center[1] + circle_radius // 2
        draw.ellipse(
            [(dot_x - dot_radius, dot_y - dot_radius), (dot_x + dot_radius, dot_y + dot_radius)],
            fill=fill_color
        )

    return img

def generate_tile_image(image_path, bg_path, number=None, font=None, tile_type='desert'):
    # Load the background image
    img = Image.open(image_path).convert('RGBA')

    # Desert tiles do not have a number on them
    if tile_type == 'desert':
        bg = Image.open(bg_path).convert('RGBA')
        bg = bg.resize(img.size)
        combined = Image.alpha_composite(bg, img).convert('RGB')

        return combined

    img = draw_number_plate(img, font, number)

    bg = Image.open(bg_path).convert('RGBA')
    bg = bg.resize(img.size)
    combined = Image.alpha_composite(bg, img).convert('RGB')

    return combined

def img_to_tensor(img, img_size):
    transform = ToTensor()

    img_np = transform(img)

    # interpolate to ensure same size
    resized_img = interpolate(img_np.unsqueeze(0), img_size).squeeze(0).permute(1, 2, 0)

    return resized_img

if __name__ == '__main__':

    font = 'C:/Windows/Fonts/Georgia/georgia.ttf' # add a path to your own font
    tile_bgs_path = '../data/tile_datasets/hexagons'
    output_img_path = '../data/full/generated_synthetic_tiles'
    output_ds_path = '../data/full/compiled_dataset'
    backgrounds_path = '../data/tile_datasets/tile_backgrounds'

    tile_types = ['brick', 'desert', 'sheep', 'ore', 'wheat', 'lumber']
    valid_numbers = ['2', '3', '4', '5', '6', '8', '9', '10', '11', '12']
    backgrounds = os.listdir(backgrounds_path)

    img_size = (243, 256)
    digit_size = (100, 100)
    zoom_to_digit = 3 # factor by which to zoom in to the tile number
    threshold = 85 # used to threshold tile image to obtain digit only

    final_dict = {
        'img_path': [],
        'hex_tensor': [],
        'number_tensor': [],
        'img_label_hexagon': [],
        'img_label_number': []
    }

    bg_index = 0
    for background in backgrounds:

        bg_path = f'{backgrounds_path}/{background}'

        print(f'Reached background image {bg_index} of {len(backgrounds)}')

        for tile in tile_types:
            img_path = f'{tile_bgs_path}/{tile}.png'

            for number in valid_numbers:
                # Generate image
                img = generate_tile_image(img_path, bg_path, number, font, tile)

                # Save image
                img.save(f'{output_img_path}/bg_{bg_index}_{tile}_{number}.png')

                digit = preprocess_image(img, zoom_to_digit, digit_size)

                final_img = img_to_tensor(img, img_size)
                final_digit = img_to_tensor(digit, digit_size)

                # Save relevant information to dictionary
                final_dict['img_path'].append(img_path)
                final_dict['hex_tensor'].append(final_img)
                final_dict['number_tensor'].append(final_digit)
                final_dict['img_label_hexagon'].append(tile)
                final_dict['img_label_number'].append(number)

                if number == '2':
                    sys.exit('Done')

        bg_index += 1

    print('Synthetic samples obtained')

    to_tf_datasets(final_dict, output_ds_path)

    print('Synthetic samples saved to pickle dataset')