import numpy as np
import random
from PIL import Image
import io
import argparse
import glob
import os
import tqdm

def get_simple_data():
    with open('./data/drivetrack_example.npz', 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
        
    max_time = 20
        
    images_jpeg_bytes_simple = images_jpeg_bytes[:max_time]
    filter = queries_xyt[:, 2] < max_time
    queries_xyt_simple = queries_xyt[filter]
    tracks_xyz_simple = tracks_xyz[:max_time, filter]
    visibles_simple = visibles[:max_time, filter]
    in_npz_simple = {'images_jpeg_bytes': images_jpeg_bytes_simple, 'queries_xyt': queries_xyt_simple, 'tracks_XYZ': tracks_xyz_simple, 'visibility': visibles_simple, 'fx_fy_cx_cy': intrinsics_params}
    np.savez('./data/drivetrack_example_simple.npz', **in_npz_simple)

def get_pairwise_data_grid(number = 2, ):
    with open('./data/drivetrack_example.npz', 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
        
    max_time = len(images_jpeg_bytes)
    
    indxs = random.sample(range(0, max_time), number)
    indxs.sort()
    if number == 1:
        indxs = [10]
    elif number == 2:
        indxs = [10, 15]
        
    gt_tracks = tracks_xyz

    images_jpeg_bytes = images_jpeg_bytes[indxs]
    img = Image.open(io.BytesIO(images_jpeg_bytes[0])).convert('RGB')
    # 定义网格的范围
    H, W = img.size[0], img.size[1]
    gap = 200
    
    x = np.linspace(0, H - 1, int(H/gap))  
    y = np.linspace(0, W - 1, int(H/gap))  

    # 生成网格
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X, Y], axis=2).reshape(-1, 2)
    queries_xyt.fill(0)
    queries_xyt= np.stack([X, Y, np.zeros_like(X)], axis=2).reshape(-1, 3)
    
    tracks_xyz = np.tile(queries_xyt, (number, 1, 1))
    tracks_xyz[:, :, 2] = 2
    f_u, f_v, c_u, c_v = intrinsics_params
    tracks_xyz[:, :, 0] = (tracks_xyz[:, :, 0] - c_u ) / f_u * 2
    tracks_xyz[:, :, 1] = (tracks_xyz[:, :, 1] - c_v ) / f_v * 2
    visibles = np.ones(tracks_xyz.shape[0:2])
    visibles = visibles == 1
    in_npz_pairwise = {
        'images_jpeg_bytes': images_jpeg_bytes, 
        'queries_xyt': queries_xyt, 
        'tracks_XYZ': tracks_xyz, 
        'visibility': visibles, 
        'fx_fy_cx_cy': intrinsics_params,
        'gt_tracks': gt_tracks,
    }
    np.savez('./data/drivetrack_example_pairwise.npz', **in_npz_pairwise)
    
def get_pairwise_data_gt(input_path, output_path, start_at_frame0 = False):
    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
        
    max_time = len(images_jpeg_bytes)
    
    indxs = [int(max_time/2) - 1, int(max_time/2) + 1]
            
    images_jpeg_bytes = images_jpeg_bytes[indxs]
    
    image = Image.open(io.BytesIO(images_jpeg_bytes[0])).convert('RGB')
    # image2 = Image.open(io.BytesIO(images_jpeg_bytes[1])).convert('RGB')
    # image.save('/mnt/nas/share/home/tjy/mast3r_evaluate/tmp/0.jpg')
    # image2.save('/mnt/nas/share/home/tjy/mast3r_evaluate/tmp/1.jpg')
    H, W = image.size[0], image.size[1]
    
    visibles = visibles[indxs]
    tracks_xyz = tracks_xyz[indxs]
    if start_at_frame0:
        u_d = tracks_xyz[..., 0] / (tracks_xyz[..., 2] + 1e-8)
        v_d = tracks_xyz[..., 1] / (tracks_xyz[..., 2] + 1e-8)

        f_u, f_v, c_u, c_v = intrinsics_params

        u_d = u_d * f_u + c_u
        v_d = v_d * f_v + c_v
        
        mask = (u_d >= 0) & (u_d < H) & (v_d >= 0) & (v_d < W)
        mask = mask[0]
        mask = mask & visibles[0]
        visibles = visibles[:, mask]
        tracks_xyz = tracks_xyz[:, mask]        
        queries_xyt = np.stack([u_d[0][mask], v_d[0][mask], np.zeros_like(u_d[0][mask])], axis=1)
    else:
        if_in_indxs = np.isin(queries_xyt[:, 2], indxs)
        queries_xyt = queries_xyt[if_in_indxs]
        queries_xyt[:, 2] = [indxs.index(i) for i in queries_xyt[:, 2]]
        tracks_xyz = tracks_xyz[:, if_in_indxs]
        visibles = visibles[:, if_in_indxs]
    
    
    if queries_xyt.shape[0] == 0:
        return

    in_npz_pairwise = {
        'images_jpeg_bytes': images_jpeg_bytes, 
        'queries_xyt': queries_xyt, 
        'tracks_XYZ': tracks_xyz, 
        'visibility': visibles, 
        'fx_fy_cx_cy': intrinsics_params,
    }
    np.savez(output_path, **in_npz_pairwise)

def get_sequence_data_gt(input_path, output_path, frames = 8, start_frame = 0, step = 1):
    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
    
    
    max_time = len(images_jpeg_bytes)

    start_frame = int(max_time / 2)
    if max_time < 25:
        start_frame = max(start_frame, max_time - frames * step - 5)
    
    indxs = np.arange(start_frame, start_frame + frames * step, step)
            
    images_jpeg_bytes = images_jpeg_bytes[indxs]
    
    image = Image.open(io.BytesIO(images_jpeg_bytes[0])).convert('RGB')
    W, H = image.size[0], image.size[1]
    
    visibles = visibles[indxs]
    tracks_xyz = tracks_xyz[indxs]
    xy = np.zeros((len(indxs), queries_xyt.shape[0], 2)) # frames, queries, 2
    fx, fy, cx, cy = intrinsics_params
    mask = np.ones(queries_xyt.shape[0], dtype=bool)
    xy[0, ..., 0] = (tracks_xyz[0, ..., 0] / tracks_xyz[0, ..., 2]) * fx + cx 
    xy[0, ..., 1] = (tracks_xyz[0, ..., 1] / tracks_xyz[0, ..., 2]) * fy + cy
    mask = mask & (xy[0, ..., 0] >= 0) & (xy[0, ..., 0] < W) & (xy[0, ..., 1] >= 0) & (xy[0, ..., 1] < H)

    mask = mask & visibles[0, ...]
    visibles = visibles[:, mask]
    tracks_xyz = tracks_xyz[:, mask] 
    queries_xyt = np.stack([xy[0, mask, 0], xy[0, mask, 1], np.zeros_like(xy[0, mask, 0])], axis=1).reshape(-1, 3)

    if input_path.endswith('Apartment_release_multiuser_party_seq134_7.npz'):

        for i in range(len(images_jpeg_bytes)):
            image = Image.open(io.BytesIO(images_jpeg_bytes[i])).convert('RGB')
            image.save('/mnt/nas/share/home/tjy/mast3r_evaluate/tmp/' + 'image' + str(i) + '.png')
            
    
    in_npz_sequence = {
        'images_jpeg_bytes': images_jpeg_bytes, 
        'queries_xyt': queries_xyt, 
        'tracks_XYZ': tracks_xyz, 
        'visibility': visibles, 
        'fx_fy_cx_cy': intrinsics_params,
    }
    print('number of queries : ', queries_xyt.shape[0], 'total length : ', max_time)
    if queries_xyt.shape[0] == 0:
        return
    # np.savez(output_path, **in_npz_sequence)

def get_valid_data_gt(input_path, output_path, frames = 24, start_frame = 10):
    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
    
    max_time = len(images_jpeg_bytes)
    
    if max_time > 48:
        start_frame = int(max_time / 2)
    else:
        start_frame = min(start_frame, max_time - frames + 5)
    

    
    indxs = np.arange(start_frame, min(max_time, start_frame + frames))
            
    images_jpeg_bytes = images_jpeg_bytes[indxs]
    
    image = Image.open(io.BytesIO(images_jpeg_bytes[0])).convert('RGB')
    W, H = image.size[0], image.size[1]
    
    visibles = visibles[indxs]
    tracks_xyz = tracks_xyz[indxs]
    xy = np.zeros((len(indxs), queries_xyt.shape[0], 2)) # frames, queries, 2
    fx, fy, cx, cy = intrinsics_params
    mask = np.ones(queries_xyt.shape[0], dtype=bool)

    xy[0, ..., 0] = (tracks_xyz[0, ..., 0] / tracks_xyz[0, ..., 2]) * fx + cx 
    xy[0, ..., 1] = (tracks_xyz[0, ..., 1] / tracks_xyz[0, ..., 2]) * fy + cy
    mask = mask & (xy[0, ..., 0] >= 0) & (xy[0, ..., 0] < W) & (xy[0, ..., 1] >= 0) & (xy[0, ..., 1] < H)

    mask = mask & visibles[0, ...]
    visibles = visibles[:, mask]
    tracks_xyz = tracks_xyz[:, mask] 
    queries_xyt = np.stack([xy[0, mask, 0], xy[0, mask, 1], np.zeros_like(xy[0, mask, 0])], axis=1).reshape(-1, 3)

    in_npz_sequence = {
        'images_jpeg_bytes': images_jpeg_bytes, 
        'queries_xyt': queries_xyt, 
        'tracks_XYZ': tracks_xyz, 
        'visibility': visibles, 
        'fx_fy_cx_cy': intrinsics_params,
    }
    print('number of queries : ', queries_xyt.shape[0], 'total length : ', max_time)
    if queries_xyt.shape[0] == 0:
        return
    np.savez(output_path, **in_npz_sequence)

def get_pairwise_data_gt_folder(input_path, output_path):
    output_path = output_path + '_pair'
    files = glob.glob(os.path.join(input_path, '**', '*.npz'), recursive=True)
    files = [os.path.basename(f) for f in files]
    already_processed = glob.glob(os.path.join(output_path, '**', '*'), recursive=True)
    already_processed = [os.path.basename(f) for f in already_processed]
    for file in tqdm.tqdm(files):
        # if file in already_processed:
        #     print('already processed', file)
        #     continue
        get_pairwise_data_gt(os.path.join(input_path, file), os.path.join(output_path, file), True)
        
def get_sequence_data_gt_folder(input_path, output_path):
    length = 12
    output_path = output_path + '_seq_' + str(length)
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    files = glob.glob(os.path.join(input_path, '**', '*.npz'), recursive=True)
    files = [os.path.basename(f) for f in files]
    already_processed = glob.glob(os.path.join(output_path, '**', '*'), recursive=True)
    already_processed = [os.path.basename(f) for f in already_processed]
    step = 2
    if input_path.endswith('drivetrack'):
        step = 1
    for file in tqdm.tqdm(files):
        get_sequence_data_gt(os.path.join(input_path, file), os.path.join(output_path, file), length, step = step)

def get_valid_data_gt_folder(input_path, output_path):
    output_path = output_path + '_valid'
    files = glob.glob(os.path.join(input_path, '**', '*.npz'), recursive=True)
    files = [os.path.basename(f) for f in files]
    already_processed = glob.glob(os.path.join(output_path, '**', '*'), recursive=True)
    already_processed = [os.path.basename(f) for f in already_processed]
    for file in tqdm.tqdm(files):
        get_valid_data_gt(os.path.join(input_path, file), os.path.join(output_path, file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/drivetrack_example.npz')
    parser.add_argument('--output_path', type=str, default='./data/drivetrack_example_simple.npz')
    parser.add_argument('--dataset', type=str, default=2)
    parser.add_argument('--number', type=int, default=2)
    args = parser.parse_args()
    
    datasets = args.dataset.split()  
    for dataset in datasets:
        get_sequence_data_gt_folder(os.path.join(args.input_path, dataset), os.path.join(args.output_path, dataset))
    