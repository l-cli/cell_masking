import os
import glob
import re
import argparse
import numpy as np
import cv2
import scipy.io as sio
import tifffile
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

def make_tiles(img_path, out_dir, dims_log_path, tile_size=256, stride=256):
    """Reads OME-TIFF and saves PNG tiles."""
    print(f"--- Tiling: {img_path} ---")
    os.makedirs(out_dir, exist_ok=True)
    
    img_arr = tifffile.imread(img_path)
    
    if img_arr.ndim == 2:
        H, W = img_arr.shape
    elif img_arr.ndim == 3:
        if img_arr.shape[0] < 10: 
             img_arr = np.moveaxis(img_arr, 0, -1)
        H, W, C = img_arr.shape
    
    # SAVE DIMS OUTSIDE THE TILE LOOP
    # We save this to a specific path so it doesn't pollute the tile folder
    with open(dims_log_path, "w") as f:
        f.write(f"{W},{H}")
    print(f"Dimensions {W}x{H} saved to {dims_log_path}")

    im = Image.fromarray(img_arr)
    if im.mode != "RGB":
        im = im.convert("RGB")

    count = 0
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            box = (x, y, min(x + tile_size, W), min(y + tile_size, H))
            tile_img = im.crop(box)
            
            if tile_img.size != (tile_size, tile_size):
                canvas = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                canvas.paste(tile_img, (0, 0))
                tile_img = canvas
            
            tile_img.save(os.path.join(out_dir, f"y{y:08d}_x{x:08d}.png"))
            count += 1
            
    print(f"Generated {count} tiles in {out_dir}")

def process_mat_files(mat_dir, mask_out_dir):
    print(f"--- Processing .mat files from {mat_dir} ---")
    os.makedirs(mask_out_dir, exist_ok=True)
    mat_files = glob.glob(os.path.join(mat_dir, '*.mat'))
    
    for mat_path in tqdm(mat_files, desc="Converting Mats"):
        data = sio.loadmat(mat_path)
        inst_map = data['inst_map'] 
        binary_mask = np.where(inst_map > 0, 255, 0).astype(np.uint8)
        base_name = os.path.basename(mat_path)
        file_name = os.path.splitext(base_name)[0] + ".png"
        cv2.imwrite(os.path.join(mask_out_dir, file_name), binary_mask)

def stitch_directory(tile_dir, output_path, mode='RGB', crop_to_orig=None):
    print(f"--- Stitching {mode} from {tile_dir} ---")
    
    pat = re.compile(r".*y(\d+)_x(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
    tiles = []
    Wmax, Hmax = 0, 0
    tw, th = None, None

    files = glob.glob(os.path.join(tile_dir, "*"))
    if not files:
        print("No tiles found to stitch.")
        return

    for f in files:
        m = pat.match(os.path.basename(f))
        if not m: continue
        y0, x0 = int(m.group(1)), int(m.group(2))
        
        if tw is None:
            with Image.open(f) as tmp:
                tw, th = tmp.size
        
        Wmax = max(Wmax, x0 + tw)
        Hmax = max(Hmax, y0 + th)
        tiles.append((x0, y0, f))

    bg_color = 0 if mode == 'L' else (0, 0, 0)
    canvas = Image.new(mode, (Wmax, Hmax), bg_color)

    for x0, y0, f in tqdm(tiles, desc="Stitching"):
        try:
            im = Image.open(f)
            if im.mode != mode:
                im = im.convert(mode)
            canvas.paste(im, (x0, y0))
        except Exception as e:
            print(f"Error on {f}: {e}")

    if crop_to_orig:
        print(f"Cropping to original size: {crop_to_orig}")
        canvas = canvas.crop((0, 0, crop_to_orig[0], crop_to_orig[1]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p_tile = subparsers.add_parser('tile')
    p_tile.add_argument('--input', required=True)
    p_tile.add_argument('--out_dir', required=True)
    p_tile.add_argument('--dims_out', required=True) # NEW ARGUMENT

    p_mat = subparsers.add_parser('mats')
    p_mat.add_argument('--mat_dir', required=True)
    p_mat.add_argument('--mask_out_dir', required=True)

    p_stitch = subparsers.add_parser('stitch')
    p_stitch.add_argument('--tile_dir', required=True)
    p_stitch.add_argument('--output', required=True)
    p_stitch.add_argument('--mode', default='RGB', choices=['RGB', 'L'])
    p_stitch.add_argument('--dims_file', default=None)

    args = parser.parse_args()

    if args.command == 'tile':
        make_tiles(args.input, args.out_dir, args.dims_out)
    elif args.command == 'mats':
        process_mat_files(args.mat_dir, args.mask_out_dir)
    elif args.command == 'stitch':
        orig_size = None
        if args.dims_file and os.path.exists(args.dims_file):
            with open(args.dims_file, 'r') as f:
                w, h = map(int, f.read().strip().split(','))
                orig_size = (w, h)
        stitch_directory(args.tile_dir, args.output, args.mode, crop_to_orig=orig_size)