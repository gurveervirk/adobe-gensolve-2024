import cairosvg

def convert_svg_to_png(input_svg_path, output_png_path):
    with open(input_svg_path, 'rb') as svg_file:
        cairosvg.svg2png(file_obj=svg_file, write_to=output_png_path)

# Example usage
convert_svg_to_png(r'problems\problems\frag0.svg', 'output_file.png')
