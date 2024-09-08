from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

import torch
import torch.nn.functional as F

import numpy as np
import io
import PIL
from PIL import Image
import os

import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

example_image = PIL.Image.open('data/mouse1.jpg').convert('RGB')
example_image_pairs = [tuple(load_images(['data/mouse1.jpg', 'data/mouse2.jpg'], size=512))]

def get_images(images):
    imgs = []
    
    for img in images:
        new_size = tuple(int(round(x/16))*16 for x in img.size)
        img = img.resize(new_size, PIL.Image.LANCZOS)
        
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    return imgs


def test_mast3r():
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    weights_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    # images = load_images(['data/mouse1.jpg', 'data/mouse2.jpg'], size=512)
    images = [PIL.Image.open('data/mouse1.jpg').convert('RGB'), PIL.Image.open('data/mouse2.jpg').convert('RGB')]
    images = get_images(images)
    
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

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
        rgb_tensor = view['img'] * image_std + image_mean
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
    pl.savefig('output/matches.png')


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def get_image_pairs(images_jpeg_bytes, query_idx, size=512, square_ok=False):
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
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
        
        img.save('output/image' + str(idx) + '.png')
        
        if(idx == 1):
            origin_shape = (W1, H1)
            current_shape = (W2, H2)
    
    # (任意时刻的frame, 需要track的frame的indx)
    for i in range(len(imgs)):
        for j in query_idx:
            img_pairs.append(tuple([imgs[i], imgs[j]]))
            img_pairs_idxs.append((i, j))
    
    return img_pairs, img_pairs_idxs, origin_shape, current_shape, len(imgs)

def check_mast3r_output():
    images_ = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)
    image_pairs_ = [tuple(images_)]
    
    with open('./data/drivetrack_example.npz', 'rb') as in_f:
        in_npz = np.load(in_f, allow_pickle=True)
        images_jpeg_bytes = in_npz['images_jpeg_bytes']
        queries_xyt = in_npz['queries_xyt'] # n, 3
        tracks_xyz = in_npz['tracks_XYZ'] #t, n, 3
        visibles = in_npz['visibility']
        intrinsics_params = in_npz['fx_fy_cx_cy']
        gt_tracks = in_npz['tracks_XYZ']
    
    query_idx = list(set(int(idx) for idx in queries_xyt[:, 2]))
    
    image_pairs, image_pairs_idxs, origin_shape, current_shape, t = get_image_pairs(images_jpeg_bytes, query_idx)
    
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    weights_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    
    output = inference(image_pairs, model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    # turn to the data for TAPVid3D
    prediction_tracks_xyz = np.zeros_like(tracks_xyz)
    prediction_visibles = np.zeros_like(visibles)
    
    pts = pred1['pts3d'] # t * #query_frames, h, w, 3
    pts = pts.view(t, len(query_idx), current_shape[1], current_shape[0], 3)
    conf = pred1['conf'].view(t, len(query_idx), current_shape[1], current_shape[0])
    
    for i in range(queries_xyt.shape[0]):
        time = int(queries_xyt[i, 2])
        x, y = queries_xyt[i, 0] / origin_shape[0] * current_shape[0], queries_xyt[i, 1] / origin_shape[1] * current_shape[1]
        x0 = np.floor(x).astype(int)
        x1 = min(x0 + 1, current_shape[0] - 1)
        y0 = np.floor(y).astype(int)
        y1 = min(y0 + 1, current_shape[1] - 1)

        # 获取四个相邻点的值
        Q_pts_11 = pts[:, query_idx.index(time), y0, x0]
        Q_pts_21 = pts[:, query_idx.index(time), y0, x1]
        Q_pts_12 = pts[:, query_idx.index(time), y1, x0]
        Q_pts_22 = pts[:, query_idx.index(time), y1, x1]
        
        Q_conf_11 = conf[:, query_idx.index(time), y0, x0]
        Q_conf_21 = conf[:, query_idx.index(time), y0, x1]
        Q_conf_12 = conf[:, query_idx.index(time), y1, x0]
        Q_conf_22 = conf[:, query_idx.index(time), y1, x1]

        # 计算插值权重
        wx = x - x0
        wy = y - y0

        # 双线性插值公式
        prediction_tracks_xyz[:, i] = (
            Q_pts_11 * (1 - wx) * (1 - wy) +
            Q_pts_21 * wx * (1 - wy) +
            Q_pts_12 * (1 - wx) * wy +
            Q_pts_22 * wx * wy
        )
        # TODO:confidence的计算
        prediction_visibles[:, i] = (
            Q_conf_11 * (1 - wx) * (1 - wy) +
            Q_conf_21 * wx * (1 - wy) +
            Q_conf_12 * (1 - wx) * wy +
            Q_conf_22 * wx * wy
        )
                               
    pts = pts.reshape(pts.shape[0], -1, pts.shape[-1]).numpy()
    pred_norms = np.sqrt(
      np.maximum(1e-12, np.sum(np.square(pts), axis=-1))
    )
    gt_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(gt_tracks), axis=-1)))     
    scale_factor = np.nanmean(
          gt_norms, axis=(-2, -1), keepdims=True
      ) / np.nanmean(pred_norms, axis=(-2, -1), keepdims=True)
    
    
    gt_xy_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(gt_tracks[:, :, :2]), axis=-1)))
    gt_z_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(gt_tracks[:, :, 2:]), axis=-1)))
    pre_xy_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(pts[:, :, :2]), axis=-1)))
    pre_z_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(pts[:, :, 2:]), axis=-1)))
    gt_xy_div_z = np.nanmean(gt_xy_norms , axis=(-2, -1)) / np.nanmean(gt_z_norms, axis=(-2, -1))
    pre_xy_div_z = np.nanmean(pre_xy_norms , axis=(-2, -1)) / np.nanmean(pre_z_norms, axis=(-2, -1))
    scale_factor = gt_xy_div_z / pre_xy_div_z
    # prediction_tracks_xyz[:, :,0:2] *= scale_factor 
        
    output = {
        'tracks_XYZ': prediction_tracks_xyz,
        'visibility': visibles,
        'fx_fy_cx_cy': intrinsics_params,
        'images_jpeg_bytes': images_jpeg_bytes,
        'queries_xyt': queries_xyt,
    }
    np.savez('./output/prediction.npz', **output)
    
    
     
    
    
    

    

if __name__ == '__main__':
    # test_mast3r()
    check_mast3r_output()
    