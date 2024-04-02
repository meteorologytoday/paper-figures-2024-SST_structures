from PIL import Image, ImageFont, ImageDraw
import os
import argparse
import postprocess_tools
import numpy as np

parser = argparse.ArgumentParser(
                    prog = 'postprocess_figures.py',
                    description = 'Use PIL to generate combined image.',
)

parser.add_argument('--input-dir',  type=str, help='Input directory', required=True)
parser.add_argument('--output-dir', type=str, help='Output directory', required=True)
args = parser.parse_args()
print(args)

# ==================================================
print("Merging SST analysis figure")
new_img = postprocess_tools.concatImages([
    os.path.join(args.input_dir, "sst_analysis_map.png"),
    os.path.join(args.input_dir, "sst_analysis_spec.png"),
], "horizontal")

new_img.save(os.path.join(args.output_dir, "merged-sst_analysis.png"), format="PNG")

# ==================================================

print("Merging experiment design figure")
img_left = Image.open(os.path.join(args.input_dir, "experiment_design.png"))
img_right = Image.open(os.path.join(args.input_dir, "input_sounding_woML.png"))

img_left = postprocess_tools.expandCanvas(img_left, 200, "top")
draw = ImageDraw.Draw(img_left)
draw.text((100, 0), "(a)", fill=(0, 0, 0), font=ImageFont.truetype("/usr/share/fonts/open-sans/OpenSans-Regular.ttf", 150))


new_height = 1500


left_old_size  = np.array(img_left.size)
right_old_size  = np.array(img_right.size)

img_left = img_left.resize(
    np.floor( new_height * np.array([
        left_old_size[0] / left_old_size[1],
        1.0
    ])).astype(int)
)

img_left = img_left.crop((0, 0, img_left.size[0]-110, img_left.size[1]))

img_right = img_right.resize(
    np.floor(new_height * np.array([
        right_old_size[0] / right_old_size[1],
        1.0
    ])).astype(int)
)

new_img = postprocess_tools.concatImages([
    img_left, img_right,
], "horizontal", )

new_img.save(os.path.join(args.output_dir, "merged-experiment_design.png"), format="PNG")


    

