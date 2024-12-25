import numpy as np
import random
from PIL import Image
import io
import argparse
import glob
import os
import tqdm




# input_path = '/mnt/nas/share/home/tjy/mast3r_evaluate/visualize_data/crossing'
# output_path = '/mnt/nas/share/home/tjy/mast3r_evaluate/evaluate_data/tapvid_datasets/visual/crossing.npz'

# if __name__ == '__main__':
#     files = os.listdir(input_path)
#     image_files = [f for f in files if f.endswith('.jpg')]
#     image_files.sort()
#     mask_folser_path = os.path.join(input_path, 'mask')
    
#     mask_files = os.listdir(mask_folser_path)
#     mask_file = mask_files[0]
    
#     images = []
#     images_jpeg_bytes = []
#     for image_file in image_files:
#         image_path = os.path.join(input_path, image_file)
#         image = Image.open(image_path).convert('RGB')

#         with io.BytesIO() as output:
#             image.save(output, format="JPEG")
#             jpeg_bytes = output.getvalue()
#             images_jpeg_bytes.append(jpeg_bytes)
        
#         # images.append(image)
        
#     mask_path = os.path.join(mask_folser_path, mask_file)
#     mask = Image.open(mask_path).convert('RGB')
#     mask_array = np.array(mask)

#     # 获取mask的尺寸
#     height, width, _ = mask_array.shape

#     # 设置网格步长
#     step = 20  # 你可以根据需要调整步长

#     # 生成网格点
#     grid_x, grid_y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
#     grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

#     # 过滤网格点：只有在白色区域内的点才保留
#     valid_points = []
#     for point in grid_points:
#         x, y = point
#         if np.array_equal(mask_array[y, x], [255, 255, 255]):  # 判断是否是白色点
#             valid_points.append(np.array([x, y, 0]))  # 加上时间维度0

#     # 将结果转换为[n, 3]的数组
#     valid_points = np.array(valid_points)



#     images_jpeg_bytes = images_jpeg_bytes[:15]
#     in_npz_sequence = {
#         'images_jpeg_bytes': images_jpeg_bytes, 
#         'queries_xyt': valid_points, 
#     }
#     np.savez_compressed(output_path, **in_npz_sequence)
    
#     pass

#######################################################################################################################################


# if __name__ == '__main__':
#     files = os.listdir(input_path)
#     image_files = [f for f in files if f.endswith('.jpg')]
#     image_files.sort()
#     mask_folser_path = os.path.join(input_path, 'mask')
    
#     mask_files = os.listdir(mask_folser_path)
#     mask_file = mask_files[0]
    

#     images_jpeg_bytes = []
    
#     mask_path = os.path.join(mask_folser_path, mask_file)
#     mask = Image.open(mask_path).convert('RGB')
#     with io.BytesIO() as output:
#         mask.save(output, format="JPEG")
#         jpeg_bytes = output.getvalue()
#         images_jpeg_bytes.append(jpeg_bytes)
    
    
#     for image_file in image_files:
#         image_path = os.path.join(input_path, image_file)
#         image = Image.open(image_path).convert('RGB')

#         with io.BytesIO() as output:
#             image.save(output, format="JPEG")
#             jpeg_bytes = output.getvalue()
#             images_jpeg_bytes.append(jpeg_bytes)
    
#     # image_path = os.path.join(input_path, image_files[0])
#     # image = Image.open(image_path).convert('RGB')
#     # with io.BytesIO() as output:
#     #     image.save(output, format="JPEG")
#     #     jpeg_bytes = output.getvalue()
#     #     images_jpeg_bytes.append(jpeg_bytes)  
#     # image_path = os.path.join(input_path, image_files[6])
#     # image = Image.open(image_path).convert('RGB')
#     # with io.BytesIO() as output:
#     #     image.save(output, format="JPEG")
#     #     jpeg_bytes = output.getvalue()
#     #     images_jpeg_bytes.append(jpeg_bytes)   
    
#     images_jpeg_bytes = images_jpeg_bytes[:6]
    
#     in_npz_sequence = {
#         'images_jpeg_bytes': images_jpeg_bytes, 
#     }
#     np.savez_compressed(output_path, **in_npz_sequence)
    
#     pass


#######################################################################################################################################


input_path = '/mnt/nas/share/home/tjy/mast3r_evaluate/visualize_data/car-turn'
output_path = '/mnt/nas/share/home/tjy/mast3r_evaluate/evaluate_data/tapvid_datasets/visual/car-turn.npz'

if __name__ == '__main__':
    files = os.listdir(input_path)
    image_files = [f for f in files if f.endswith('.jpg')]
    image_files.sort()
    mask_folser_path = os.path.join(input_path, 'mask')
    
    start_frame = 7
    max_frames = 15
    
    mask_files = os.listdir(mask_folser_path)
    mask_files.sort()
    mask_files = mask_files[::2]
    image_files = image_files[::2]
    mask_files = mask_files[start_frame: start_frame + min(len(mask_files), max_frames)]
    image_files = image_files[start_frame: start_frame + min(len(image_files), max_frames)]
    
    images = []
    images_jpeg_bytes = []
    for image_file in image_files:
        image_path = os.path.join(input_path, image_file)
        image = Image.open(image_path).convert('RGB')

        with io.BytesIO() as output:
            image.save(output, format="JPEG")
            jpeg_bytes = output.getvalue()
            images_jpeg_bytes.append(jpeg_bytes)
        
        # images.append(image)
    
    query_xyts = []
    w, h = 30, 30
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folser_path, mask_file)
        mask = Image.open(mask_path).convert('RGB')
        mask_array = np.array(mask)

        # 获取mask的尺寸
        height, width, _ = mask_array.shape

        # 设置网格步长
        step = 5  # 你可以根据需要调整步长

        # 生成网格点
        grid_x, grid_y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        # 过滤网格点：只有在白色区域内的点才保留
        valid_points = []
        for point in grid_points:
            x, y = point
            if np.array_equal(mask_array[y, x], [255, 255, 255]):  # 判断是否是白色点
                valid_points.append(np.array([x, y, 0]))  # 加上时间维度0
        valid_points = np.array(valid_points)
        # valid_points需要在正方形内
        # 计算中点
        center = np.mean(valid_points, axis=0)

        # 定义长方形的边界
        half_w = w / 2
        half_h = h / 2

        # 过滤掉长方形外的点
        filtered_points = valid_points[
            (valid_points[:, 0] >= center[0] - half_w) & (valid_points[:, 0] <= center[0] + half_w) &  # x 坐标在范围内
            (valid_points[:, 1] >= center[1] - half_h) & (valid_points[:, 1] <= center[1] + half_h)    # y 坐标在范围内
        ]
        
        # 将结果转换为[n, 3]的数组
        valid_points = np.array(filtered_points)
        query_xyts.append(valid_points)




    in_npz_sequence = {
        'images_jpeg_bytes': images_jpeg_bytes, 
        'queries_xyt': query_xyts, 
    }
    np.savez(output_path, **in_npz_sequence)
    
    pass






