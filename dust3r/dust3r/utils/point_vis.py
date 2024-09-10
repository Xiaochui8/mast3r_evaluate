"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=False):
    if isinstance(points, np.ndarray):
        points = points
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # axis_pcd = open3d.create_mesh_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.ones(3) * 255

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        point_color = np.zeros([points.shape[0], 3])
        point_color[:, 1] = 0.9
        point_color[:, 2] = 0.8
        # pts.colors = open3d.utility.Vector3dVector(np.random.randn((points.shape[0], 3)))
        pts.colors = open3d.utility.Vector3dVector(point_color)
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 0))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 0, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    # lines=lines[4:8]
    # lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(1, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        # print(line_set.shape)
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def rotation_points_single_angle(points, angle, axis=0):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [rot_cos, 0, -rot_sin, 0, 1, 0, rot_sin, 0, rot_cos],
            dtype=points.dtype).reshape(3, 3)
    elif axis == 2:
        rot_mat_T = np.array(
            [rot_cos, -rot_sin, 0, rot_sin, rot_cos, 0, 0, 0, 1],
            dtype=points.dtype
        ).reshape(3, 3)
    else:
        rot_mat_T = np.array(
            [1, 0, 0, 0, rot_cos, -rot_sin, 0, rot_sin, rot_cos],
            dtype=points.dtype
        ).reshape(3, 3)

    return points @ rot_mat_T


def points_select(points, npoint=16384, far_filter=40.0):
    if points.shape[0] < npoint:
        # 当前的样本点是少的，进行填充
        choice = np.arange(0, points.shape[0], dtype=np.int32)
        extra_choice = np.random.choice(choice, npoint-points.shape[0], replace=False)      # 是否进行重复采样
        choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
    elif points.shape[0] > npoint:
        # 当前的样本点数较多，需要进行减少
        # 根据距离进行筛选
        pts_depth = points[:, 2]
        pts_near_flag = pts_depth < far_filter
        # far_idxs_choice = np.where(pts_near_flat==0)[0]
        far_idxs = np.where(pts_near_flag==0)[0]
        near_idxs = np.where(pts_near_flag==1)[0]

        if len(near_idxs) < npoint:
            # 近处的点不够
            near_idxs_choice = near_idxs

            far_idxs_choice = np.random.choice(far_idxs, npoint-len(near_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs, far_idxs_choice), aixs=0)
        else:
            choice = np.random.choice(near_idxs, npoint, replace=False)
        
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, points.shape[0], dtype=np.int32)

    points = points[choice, :]
    return points

def points_select_limit(points, dim=1, threshold=40):
    points_value = points[:, dim]
    points_width_flag = (points_value < threshold) & (points_value > (threshold * -1))
    points_selected = points[points_width_flag]
    return points_selected
    
def points_select_front(points):
    points_x = points[:, 0]
    points_index = points_x > 0
    return points[points_index]

def get_topK_points(points, topk=200, dim=2):
    if not isinstance(points, torch.Tensor):
        points = torch.FloatTensor(points)
    points_value = points[:, dim] * -1
    selected_value, selected_index = torch.topk(points_value, k=topk)
    # selected_mean = points[selected_index, 2].mean()
    selected_mean = (selected_value * -1).mean()
    # selected_mean = selected_value.mean()
    print(selected_mean)
    return points[selected_index]

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # device = xyz.device
    N, C = xyz.shape
    B = 1
    xyz = torch.tensor(xyz)
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long)
    farthest = N // 2
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def read_colmap_points(path_to_model_file):
    import struct
    import collections
    Point3D = collections.namedtuple(
        "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)
    
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        points_value = []
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            points_value.append(np.array(binary_point_line_properties[1:7]))
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
#    return points3D
    return np.array(points_value, dtype=np.float32)

if __name__ == '__main__':
    
    points_pro = np.load('points_0_1_proj.npy')
    points_1 = np.load('points_1_0_proj.npy')
    import pdb; pdb.set_trace()
    points = np.concatenate((points_pro, points_1), axis=0)
    OPEN3D_FLAG = True
    draw_scenes(
            points=points[:, :3] , ref_boxes=None, point_colors=points[:, 3:]/255.
            )