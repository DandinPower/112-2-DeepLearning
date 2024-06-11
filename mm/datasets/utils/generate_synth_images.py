import os
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def generate_synthetic_data(output_dir, num_images, text_file, image_dir, image_files, output_file, font_dir):
    # Clear the output file
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read sentences from text file
    with open(text_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]

    for i in tqdm(range(num_images)):
        # Randomly select an image and font
        selected_image = Image.open(os.path.join(image_dir, random.choice(image_files)))
        selected_image = selected_image.resize((1280, 720))
        width, height = 1280, 720
        font_size = (50, 70)

        # Convert to RGB if it's a grayscale image
        selected_image = selected_image.convert("RGB")

        # Create a copy of the selected image
        new_image = selected_image.copy()
        draw = ImageDraw.Draw(new_image)

        # Define function to create text with optional border
        def create_text(text_height, sentence, text_real_width, text_real_height, text_color, selected_font):
            text_position = ((width - text_real_width) // 2, text_height)
            region_right = text_position[0] + text_real_width

            draw.text(text_position, sentence, font=selected_font, fill=text_color)
            
            region_left = text_position[0]
            region_top = text_position[1]
            region_bottom = text_position[1] + text_real_height

            # Find maximum brightness within the region
            max_brightness = 0
            for x in range(region_left, region_right + 1):
                for y in range(region_top, region_bottom + 1):
                    try:
                        pixel_brightness = sum(selected_image.getpixel((x, y))) // 3  # Calculate brightness (average of RGB)
                    except:
                        continue
                    max_brightness = max(max_brightness, pixel_brightness)

            # Check if maximum brightness exceeds threshold
            brightness_threshold = 150  # Adjust this threshold as needed
            if max_brightness > brightness_threshold:
                border_thickness = 1
                for dx in range(-border_thickness, border_thickness + 1):
                    for dy in range(-border_thickness, border_thickness + 1):
                        draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0, 0, 0))
                draw.text(text_position, sentence, font=selected_font, fill=text_color)
            else:
                border_thickness = 1
                if random.randint(0, 3) == 2:
                    thick = random.randint(1, 3)
                    for dx in range(-border_thickness, border_thickness + thick):
                        for dy in range(-border_thickness, border_thickness + thick):
                            draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0, 0, 0))
                draw.text(text_position, sentence, font=selected_font, fill=text_color)

        gt_string = ""
        random.seed()
        num = 0
        out_range = 1

        text_color = (random.randint(250, 255), random.randint(250, 255), random.randint(250, 255))
        font_files = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
        while out_range == 1:
            sentence = random.choice(sentences).replace("\n", "")
            size = random.randint(font_size[0], font_size[1])
            selected_font = ImageFont.truetype(os.path.join(font_dir, random.choice(font_files)), size)
            text_bbox = draw.textbbox((0, 0), sentence, font=selected_font)
            text_real_width = text_bbox[2] - text_bbox[0]
            text_real_height = text_bbox[3] - text_bbox[1]
            line1 = height - int(text_real_height * 3)
            line2 = height - (text_real_height * 2)
            if width - text_real_width >= 0:
                create_text(random.randint(line1, line2), sentence, text_real_width, text_real_height, text_color, selected_font)
                out_range = 0
            else:
                num += 1
                print(f"Skipping line in image {i}")
            if num == 10:
                break
        if num == 10:
            continue
        gt_string += sentence

        output_path = os.path.join(output_dir, f"image_{i + 1}.jpg")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"image_{i + 1}.jpg\t{gt_string}\n")
        
        alpha_value = 255  # You can adjust this value
        Image.blend(selected_image, new_image, alpha=alpha_value/255.0).save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data images with text.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--num_images', type=int, required=True, help='Number of images to generate')
    parser.add_argument('--text_file', type=str, required=True, help='Path to text file with sentences')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with images to use')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output file')
    parser.add_argument('--font_dir', type=str, required=True, help='Directory with font files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    random.seed(args.seed)

    image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    generate_synthetic_data(args.output_dir, args.num_images, args.text_file, args.image_dir, image_files, args.output_file, args.font_dir)

if __name__ == "__main__":
    main()
