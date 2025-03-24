from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

def to_tf_dataset(ds_dict, output_path):
    input = [x.numpy() for x in ds_dict['img_tensor']]
    labels = ds_dict['img_label']

    # Convert labels (string categories) to integer labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Convert to TensorFlow tensors
    X_tensorflow = tf.convert_to_tensor(input, dtype=tf.float32)
    y_tensorflow = tf.convert_to_tensor(y_encoded, dtype=tf.int32)

    with open(f'{output_path}/generated_synthetic_samples.pkl', 'wb') as f:
        pickle.dump((X_tensorflow.numpy(), y_tensorflow.numpy(), label_encoder), f)

def generate_tile_image(bg_path, number=None, font=None, tile_type='desert'):
    # Load the background image
    image_path = bg_path  # Replace with your image path
    img = Image.open(image_path).convert('RGB')

    # Get image size
    width, height = img.size

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Desert tiles do not have a number on them
    if tile_type == 'desert':
        return img

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

    return img

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

    tile_types = ['brick', 'desert', 'lumber', 'ore', 'sheep', 'wheat']
    valid_numbers = ['2', '3', '4', '5', '6', '8', '9', '10', '11', '12']

    img_size = (135, 121)

    final_dict = {
        'img_path': [],
        'img_tensor': [],
        'img_label': []
    }

    for tile in tile_types:
        img_path = f'{tile_bgs_path}/{tile}.png'

        if tile != 'desert':
            for number in valid_numbers:
                # Generate image
                img = generate_tile_image(img_path, number, font, tile)

                # Save image
                img.save(f'{output_img_path}/{tile}_{number}.png')

                # Save relevant information to dictionary
                final_dict['img_path'].append(img_path)
                final_dict['img_tensor'].append(img_to_tensor(img, img_size))
                final_dict['img_label'].append(tile)
        else:
            # Generate image
            img = generate_tile_image(img_path, font=font, tile_type=tile)

            # Save image
            img.save(f'{output_img_path}/{tile}.png')

            # Save relevant information to dictionary
            final_dict['img_path'].append(img_path)
            final_dict['img_tensor'].append(img_to_tensor(img, img_size))
            final_dict['img_label'].append(tile)

    to_tf_dataset(final_dict, output_ds_path)