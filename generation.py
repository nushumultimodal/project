from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np

# Path to your dataset CSV file
input_file = ''  # Update with your actual file path

# Output directory to save PNG files
output_dir = 'nvshu_png_images/'
os.makedirs(output_dir, exist_ok=True)

# Load the Nüshu text from CSV
df = pd.read_csv(input_file)

# Font path (ensure you have a Nüshu-compatible font)
font_path = 'NotoSansNushu-Regular.ttf'  # Replace with actual font path if you have one
base_font_size = 30  # Base font size
char_spacing = 25  # Adjust the spacing between characters

# Set fixed image size for TrOCR compatibility
target_width = 384
target_height = 384

# Loop through the rows and save each Nüshu sentence as a PNG
for idx, row in df.iterrows():
    nushu_text = row['Nushu']  # Assuming 'Nushu' is the column name
    
    # Calculate dynamic font size based on number of characters
    num_chars = len(nushu_text)
    if num_chars > 10:
        # Reduce font size for longer texts, but not below 20
        font_size = max(40, base_font_size - (num_chars - 10))
    else:
        font_size = base_font_size

    # Initialize the font to calculate character dimensions
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font not found. Please ensure the correct font is specified.")
        continue

    # First pass: calculate character dimensions and layout
    char_dimensions = []
    max_char_height = 0
    for char in nushu_text:
        char_bbox = font.getbbox(char)
        char_width = char_bbox[2] - char_bbox[0]
        char_height = char_bbox[3] - char_bbox[1]
        char_dimensions.append((char_width, char_height))
        max_char_height = max(max_char_height, char_height)

    # Calculate layout
    columns = []
    current_column = []
    current_height = 0
    
    # Process characters in original order
    for char, (width, height) in zip(nushu_text, char_dimensions):
        if current_height + height + char_spacing > target_height:
            columns.append(current_column)
            current_column = []
            current_height = 0
        current_column.append((char, width, height))
        current_height += height + char_spacing
    
    if current_column:
        columns.append(current_column)

    # Reverse the columns to get right-to-left order
    columns = columns[::-1]

    # Calculate actual width needed
    total_width = 0
    for column in columns:
        max_width = max(width for _, width, _ in column)
        total_width += max_width + char_spacing

    # Add padding
    img_width = total_width + 20  # Add padding on both sides

    # Create a new image with white background (RGB mode)
    image = Image.new('RGB', (img_width, target_height), color='white')
    draw = ImageDraw.Draw(image)

    # Start position (right-aligned)
    x_position = img_width - 10  # Start from right with padding
    y_position = 0  # Start from top

    # Calculate maximum vertical position
    max_y_position = target_height - (max_char_height + char_spacing)

    # Draw characters
    for column in reversed(columns):  # Draw columns from right to left
        column_width = max(width for _, width, _ in column)
        for char, width, height in column:  # Keep original order within column
            # Draw the character
            draw.text((x_position - width, y_position), char, font=font, fill='black')
            y_position += height + char_spacing
        
        # Move to next column
        x_position -= (column_width + char_spacing)
        y_position = 0  # Reset vertical position

    # Create a new image with target dimensions
    final_image = Image.new('RGB', (target_width, target_height), color='white')
    
    # Calculate scaling factor to fit the content while maintaining aspect ratio
    scale = min(target_width / img_width, target_height / target_height)
    new_width = int(img_width * scale)
    new_height = int(target_height * scale)
    
    # Resize the original image maintaining aspect ratio
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate position to paste the resized image (centered)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # Paste the resized image onto the final image
    final_image.paste(resized_image, (paste_x, paste_y))
    
    # Save the image with a unique name based on the index in the dataset
    image_name = f"nushu_{idx + 1}.png"  # Create a name based on the row index (1-based)
    final_image.save(os.path.join(output_dir, image_name), 'PNG')  # Save as PNG

    # Only print progress every 50th image
    if (idx + 1) % 50 == 0:
        print(f"Image {idx + 1} saved as {image_name}")

print(f"PNG images generated and saved in '{output_dir}'")
