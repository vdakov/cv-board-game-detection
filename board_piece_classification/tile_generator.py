from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
import tensorflow as tf
import pickle
import os

def save_to_file(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def to_tf_datasets(ds_dict, output_path):
    input = [x.numpy() for x in ds_dict['img_tensor']]
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
    X = tf.convert_to_tensor(input, dtype=tf.float32)
    y_hexagons = tf.convert_to_tensor(y_hexagons_encoded, dtype=tf.int32)
    y_numbers = tf.convert_to_tensor(y_numbers_encoded, dtype=tf.int32)

    print('Compiled datasets; preparing to store')

    save_path_hexagons = f'{output_path}/synthetic_dataset_hexagons.pkl'
    save_path_numbers = f'{output_path}/synthetic_dataset_numbers.pkl'

    save_to_file(save_path_hexagons, (X, y_hexagons))
    save_to_file(save_path_numbers, (X, y_numbers))


def generate_tile_image(image_path, bg_path, number=None, font=None, tile_type='desert'):
    # Load the background image
    img = Image.open(image_path).convert('RGBA')

    # Get image size
    width, height = img.size

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Desert tiles do not have a number on them
    if tile_type == 'desert':
        bg = Image.open(bg_path).convert('RGBA')
        bg = bg.resize(img.size)
        combined = Image.alpha_composite(bg, img).convert('RGB')

        return combined

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
    text_position = (circle_center[0] - text_width // 2, circle_center[1] - text_height)

    # Draw text (black)
    if number_text == '8' or number_text == '6':
        fill_color = 'red'
    else:
        fill_color = 'black'

    draw.text(text_position, number_text, font=font, fill=fill_color)

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

    final_dict = {
        'img_path': [],
        'img_tensor': [],
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

                # Save relevant information to dictionary
                final_dict['img_path'].append(img_path)
                final_dict['img_tensor'].append(img_to_tensor(img, img_size))
                final_dict['img_label_hexagon'].append(tile)
                final_dict['img_label_number'].append(number)

        bg_index += 1

    print('Synthetic samples obtained')

    to_tf_datasets(final_dict, output_ds_path)

    print('Synthetic samples saved to pickle dataset')