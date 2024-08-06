import subprocess
from PIL import Image
import os

def png_to_svg(png_file, svg_file):
    """Converts a PNG file to an SVG file using Potrace."""
    # Step 1: Convert PNG to PBM (Portable Bitmap)
    pbm_file = 'temp.pbm'
    image = Image.open(png_file).convert('1')  # Convert image to binary (black and white)
    image.save(pbm_file)
    
    # Step 2: Use Potrace to convert PBM to SVG
    command = ['potrace', pbm_file, '-s', '-o', svg_file]
    subprocess.run(command, check=True)
    
    # Clean up temporary PBM file
    os.remove(pbm_file)

if __name__ == "__main__":
    # Input PNG file
    png_file = 'problems\problems\occlusion2_rec.png'
    
    # Output SVG file
    svg_file = 'output.svg'
    
    # Convert PNG to SVG
    png_to_svg(png_file, svg_file)
