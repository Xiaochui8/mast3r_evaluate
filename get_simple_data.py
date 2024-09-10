import numpy as np
import random
from PIL import Image
import io
import argparse

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
    
def get_pairwise_data_gt(number, input_path, output_path, start_at_frame0 = False):
    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
        
    max_time = len(images_jpeg_bytes)
    
    indxs = random.sample(range(0, min(max_time, 100)), number)
    indxs.sort()
    if number == 1:
        indxs = [10]
    elif number == 2:
        indxs = [10, 15]
        
    gt_tracks = tracks_xyz

    images_jpeg_bytes = images_jpeg_bytes[indxs]
    
    tracks_xyz = tracks_xyz[indxs]
    if start_at_frame0:
        u_d = tracks_xyz[..., 0] / (tracks_xyz[..., 2] + 1e-8)
        v_d = tracks_xyz[..., 1] / (tracks_xyz[..., 2] + 1e-8)

        f_u, f_v, c_u, c_v = intrinsics_params

        u_d = u_d * f_u + c_u
        v_d = v_d * f_v + c_v
        
        queries_xyt = np.stack([u_d[0], v_d[0], np.zeros_like(u_d[0])], axis=1)
    else:
        if_in_indxs = np.isin(queries_xyt[:, 2], indxs)
        queries_xyt = queries_xyt[if_in_indxs]
        queries_xyt[:, 2] = [indxs.index(i) for i in queries_xyt[:, 2]]
    
    
    visibles = visibles[indxs]

    in_npz_pairwise = {
        'images_jpeg_bytes': images_jpeg_bytes, 
        'queries_xyt': queries_xyt, 
        'tracks_XYZ': tracks_xyz, 
        'visibility': visibles, 
        'fx_fy_cx_cy': intrinsics_params,
        'gt_tracks': gt_tracks,
    }
    np.savez(output_path, **in_npz_pairwise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/drivetrack_example.npz')
    parser.add_argument('--output_path', type=str, default='./data/drivetrack_example_simple.npz')
    parser.add_argument('--number', type=int, default=2)
    args = parser.parse_args()
    get_pairwise_data_gt(args.number, args.input_path, args.output_path)
    