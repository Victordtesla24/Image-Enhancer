from PIL import Image, ImageDraw
import numpy as np

# Create a 500x500 RGB image with a gradient background
width = 500
height = 500
image = Image.new("RGB", (width, height))
draw = ImageDraw.Draw(image)

# Create gradient background
for y in range(height):
    for x in range(width):
        r = int(255 * x / width)
        g = int(255 * y / height)
        b = 128
        draw.point((x, y), fill=(r, g, b))

# Draw some shapes
draw.ellipse([100, 100, 400, 400], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
draw.rectangle([200, 200, 300, 300], fill=(255, 0, 0), outline=(0, 0, 0), width=2)

# Save the image
image.save("test_image.png", "PNG")
print("Test image created successfully!")
