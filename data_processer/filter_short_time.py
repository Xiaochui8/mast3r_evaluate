import argparse
import os
import glob
import tqdm
import numpy as np
import shutil

def filter_short_time(input_path, output_path, threshold_frames):
    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
    
    print(len(images_jpeg_bytes))
    if(len(images_jpeg_bytes) <= threshold_frames):
        shutil.copyfile(input_path, output_path)

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    threshold_frames = args.threshold_frames
    
    already_processed = glob.glob(os.path.join(output_dir, '**', '*'), recursive=True)
    already_processed = [os.path.basename(f) for f in already_processed]
    
    files = glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
    files = [os.path.basename(f) for f in files]
    
    for file in tqdm.tqdm(files):
        if file in already_processed:
            continue
        
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        filter_short_time(input_path, output_path, threshold_frames)
        

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter short time data')
    parser.add_argument('--input_dir', type=str, help='input directory')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--threshold_frames', type=int, help='threshold of time')
    args = parser.parse_args()
    main(args)
    