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


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_images(images):
    imgs = []
    
    for img in images:
        new_size = tuple(int(round(x/16))*16 for x in img.size)
        img = img.resize(new_size, PIL.Image.LANCZOS)
        
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs



def get_matches(output, device = 'cuda', indx = 0):
    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'][indx], pred2['desc'][indx]

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # visualize a few matches
    import numpy as np
    import torch
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl
    
    
    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'][indx] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.savefig('./output/matches.png')
    

    


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def get_image_pairs(images_jpeg_bytes, query_idx, size=512, save_imgs=False, factor = 16):
    imgs = []
    img_pairs = np.empty((len(images_jpeg_bytes), len(query_idx)), dtype=tuple)
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
        for j in range(len(query_idx)):
            img_pairs[i][j] = tuple([imgs[i], imgs[j]])
    
    return img_pairs, origin_shape, current_shape, len(imgs)

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
    norm_coords = torch.stack([torch.tensor(norm_x), torch.tensor(norm_y)], dim=-1).to(pts.device).to(torch.float32)
    
    # 调整 pts 形状为 (n, 3, h, w) 以适配 grid_sample
    pts = pts.permute(0, 3, 1, 2)  # 转换为 (n, 3, h, w)
    
    # norm_coords 扩展为 (n, 1, #query_xy, 2)
    norm_coords = norm_coords.expand(n, 1, -1, -1)
    # 使用 grid_sample 进行插值, 插值结果 shape 为 (n, 3, 1, 1)
    interpolated_vals = F.grid_sample(pts, norm_coords, mode='bilinear', align_corners=True)
    
    # 去掉多余的维度，返回形状为 (n, 3) 的结果
    return interpolated_vals.squeeze(-2)

def check_mast3r_output(input_path, output_path):
    device = 'cuda'
    weights_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    
    with open(input_path, 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
        gt_tracks = in_npz['tracks_XYZ']
    
    query_time = queries_xyt[:, 2] # from indx to time
    query_idx = list(set(int(idx) for idx in queries_xyt[:, 2])) # possible query time
    
    image_pairs, origin_shape, current_shape, t = get_image_pairs(images_jpeg_bytes, query_idx)

    
    
    # turn to the data for TAPVid3D
    prediction_tracks_xyz = np.zeros_like(tracks_xyz)
    prediction_visibles = np.zeros_like(visibles)
    total_fx, total_fy = 0, 0
    cx, cy = current_shape[0] / 2, current_shape[1] / 2
    x_coords, y_coords = np.meshgrid(np.arange(current_shape[0]), np.arange(current_shape[1]), indexing='xy')  
    
    for current_query_idx in tqdm.trange(len(query_idx)):
        current_query_time = query_idx[current_query_idx]
        output = inference(list(image_pairs[:, current_query_idx]), model, device, batch_size=32, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        
        pts = pred2['pts3d_in_other_view'] # t * #query_frames, h, w, 3
        pts = pts.view(t, current_shape[1], current_shape[0], 3)
        pts2 = pred1['pts3d'].view(t, current_shape[1], current_shape[0], 3)
        conf = pred2['conf'].view(t, current_shape[1], current_shape[0])
        
        
        
        # x,y是time frame上的坐标，而不是任意帧上的坐标，得找到对应的点才行
        for i in range(queries_xyt.shape[0]):
            mask = query_time == current_query_time
            x, y = queries_xyt[mask, 0] / origin_shape[0] * current_shape[0], queries_xyt[mask, 1] / origin_shape[1] * current_shape[1]
            prediction_tracks_xyz[:, mask] = interpolate_at_xy(pts, x, y).permute(0, 2, 1)
            prediction_visibles[:, mask] = interpolate_at_xy(conf.unsqueeze(-1), x, y).permute(0, 2, 1).squeeze(-1)


        for i in range(pts2.shape[0]):
            fx = (x_coords + 1 - cx) / (pts2[i, :, :, 0] / (pts2[i, :, :, 2] + 1e-8))
            fy = (y_coords + 1 - cy) / (pts2[i, :, :, 1] / (pts2[i, :, :, 2] + 1e-8))
            if np.isinf(fy.nanmedian()) or np.isinf(fx.nanmedian()):
                print('inf')
            total_fx += fx.nanmedian().float() 
            total_fy += fy.nanmedian().float()
                
                    
    intrinsics_params = [
                    total_fx /(t * len(query_idx)) * origin_shape[0] / current_shape[0],
                    total_fy /(t * len(query_idx)) * origin_shape[1] / current_shape[1],
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
    
    check_mast3r_output(args.input_path, args.output_path)

    