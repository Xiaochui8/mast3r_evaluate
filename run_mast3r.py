from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.point_vis import draw_scenes

import tqdm
import torch
import torch.nn.functional as F

import numpy as np
import io
import PIL
from PIL import Image
import os

import torchvision.transforms as tvf
import argparse
import glob


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_image_pairs(images_jpeg_bytes, query_idx, size=512, save_imgs=False, factor = 16):
    imgs = []
    img_pairs = []
    img_pairs_idxs = []
    idx = 0
    origin_shape = ()
    current_shape = ()
    for image_jpeg_bytes in images_jpeg_bytes:
        idx += 1
        img = Image.open(io.BytesIO(image_jpeg_bytes)).convert('RGB')
        W1, H1 = img.size
        
        # 找到最长边
        max_edge = max(W1, H1)
        
        # 计算缩放比例
        scale = size / max_edge
        
        # 根据比例缩放两边
        W2 = int(W1 * scale)
        H2 = int(H1 * scale)
        
        # 将宽高调整为factor的倍数
        W2 = (W2 ) // factor * factor
        H2 = (H2 ) // factor * factor
        
        img = img.resize((W2, H2))

        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        
        if save_imgs:
            img.save('output/image' + str(idx) + '.png')
        
        if(idx == 1):
            origin_shape = (W1, H1)
            current_shape = (W2, H2)
    
    # 每一个pair --- (任意时刻的frame, 需要track的frame的indx)
    for i in range(len(imgs)):
        for j in query_idx:
            img_pairs.append(tuple([imgs[i], imgs[j]]))
            img_pairs_idxs.append((i, j))
    
    return img_pairs, img_pairs_idxs, origin_shape, current_shape, len(imgs)

def interpolate_at_xy(pts, x, y):
    """
    在 (x, y) 位置对 shape 为 (n, h, w, 3) 的张量进行插值
    
    参数:
    pts: 形状为 (n, h, w, 3) 的张量
    x, y: 单个插值点，分别是 x 和 y 坐标
    
    返回:
    形状为 (n, 3) 的插值结果
    """
    n, h, w, c = pts.shape
    
    # 将坐标归一化到 [-1, 1] 区间
    norm_x = 2.0 * x / (w - 1) - 1.0
    norm_y = 2.0 * y / (h - 1) - 1.0
    
    # 创建归一化的坐标网格, shape为 (1, 1, 2)
    norm_coords = torch.tensor([norm_x, norm_y], dtype=torch.float32, device=pts.device)
    
    # 调整 pts 形状为 (n, 3, h, w) 以适配 grid_sample
    pts = pts.permute(0, 3, 1, 2)  # 转换为 (n, 3, h, w)
    
    # 使用 grid_sample 进行插值, 插值结果 shape 为 (n, 3, 1, 1)
    norm_coords = norm_coords.expand(n, 1, 1, -1)
    interpolated_vals = F.grid_sample(pts, norm_coords, mode='bilinear', align_corners=True)
    
    # 去掉多余的维度，返回形状为 (n, 3) 的结果
    return interpolated_vals.squeeze(-1, -2)



def get_mast3r_output_single_folder(input_path, output_path, device = 'cuda'):
    weights_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    
    already_processed = glob.glob(os.path.join(output_path, '**', '*'), recursive=True)
    already_processed = [os.path.basename(f) for f in already_processed]
    
    files = glob.glob(os.path.join(input_path, '**', '*'), recursive=True)
    files = [os.path.basename(f) for f in files]
    
    for file in tqdm.tqdm(files):
        if file in already_processed:
            print('already processed', file)
            continue
        
        print('processing', file)
        get_mast3r_output_single_file(model, os.path.join(input_path, file), os.path.join(output_path, file), device)
    
    
    

def get_mast3r_output_single_file(model, input_path, output_path, device = 'cuda'):

    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
        gt_tracks = in_npz['tracks_XYZ']
    
    query_idx = list(set(int(idx) for idx in queries_xyt[:, 2]))
    
    image_pairs, image_pairs_idxs, origin_shape, current_shape, t = get_image_pairs(images_jpeg_bytes, query_idx)
    
    


    
    
    output = inference(image_pairs, model, device, batch_size=1, verbose=True)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # turn to the data for TAPVid3D
    prediction_tracks_xyz = np.zeros_like(tracks_xyz)
    prediction_visibles = np.zeros_like(visibles)
    
    
    pts = pred2['pts3d_in_other_view'] # t * #query_frames, h, w, 3
    pts = pts.view(t, len(query_idx), current_shape[1], current_shape[0], 3)
    pts2 = pred1['pts3d'].view(t, len(query_idx), current_shape[1], current_shape[0], 3)
    conf = pred2['conf'].view(t, len(query_idx), current_shape[1], current_shape[0])
    
    
    for i in tqdm.trange(queries_xyt.shape[0]):
        time = int(queries_xyt[i, 2])
        # TODO: x,y是time frame上的坐标，而不是任意帧上的坐标，得找到对应的点才行
        x, y = queries_xyt[i, 0] / origin_shape[0] * current_shape[0], queries_xyt[i, 1] / origin_shape[1] * current_shape[1]
        if x < 0 or x >= current_shape[0] or y < 0 or y >= current_shape[1]:
            print('out of range')
            continue
        temp_query_time = query_idx.index(time)
        prediction_tracks_xyz[:, i] = interpolate_at_xy(pts[:, temp_query_time], x, y)
        prediction_visibles[:, i] = interpolate_at_xy(conf.unsqueeze(-1)[:, temp_query_time], x, y).squeeze(-1)

        

    
    cx, cy = current_shape[0] / 2, current_shape[1] / 2
    x_coords, y_coords = np.meshgrid(np.arange(current_shape[0]), np.arange(current_shape[1]), indexing='xy')  
    
    total_fx, total_fy = 0, 0
    for i in range(pts2.shape[0]):
        for j in range(pts2.shape[1]):
            fx = (x_coords + 1 - cx) / (pts2[i, j, :, :, 0] / (pts2[i, j, :, :, 2] + 1e-8))
            fy = (y_coords + 1 - cy) / (pts2[i, j, :, :, 1] / (pts2[i, j, :, :, 2] + 1e-8))
            total_fx += fx.mean().float()
            total_fy += fy.mean().float()
            
                
    intrinsics_params = [
                    total_fx /(pts2.shape[0] * pts2.shape[1]) * origin_shape[0] / current_shape[0],
                    total_fy /(pts2.shape[0] * pts2.shape[1]) * origin_shape[1] / current_shape[1],
                    cx * origin_shape[0] / current_shape[0],
                    cy * origin_shape[1] / current_shape[1],
                ]
    
    gt_tracks_norm_factor = np.median(np.linalg.norm(gt_tracks, axis = -1),axis = -1)
    tracks_norm_factor = np.median(np.linalg.norm(prediction_tracks_xyz, axis = -1),axis = -1)
    prediction_tracks_xyz = prediction_tracks_xyz * gt_tracks_norm_factor.reshape(-1, 1, 1) / tracks_norm_factor.reshape(-1, 1, 1)
    
    

    output = {
        'tracks_XYZ': prediction_tracks_xyz,
        'visibility': visibles,
        'fx_fy_cx_cy': intrinsics_params,
        'images_jpeg_bytes': images_jpeg_bytes,
        'queries_xyt': queries_xyt,
    }
    np.savez(output_path, **output)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/drivetrack_example.npz')
    parser.add_argument('--output_path', type=str, default='./output/prediction.npz')
    args = parser.parse_args()
    
    get_mast3r_output_single_folder(args.input_path, args.output_path)

    