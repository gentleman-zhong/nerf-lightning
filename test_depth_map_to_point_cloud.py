import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import os
import json
from path import Path
from tqdm import tqdm
import open3d as o3d


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def depth_image_to_point_cloud(rgb, depth, K, c2w, scale=1.0):
    # scale缩放因子，K内参矩阵，c2w相机转世界
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(c2w, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B)))



    # 创建一个PointCloud对象
    pcd = o3d.geometry.PointCloud()
    # 去除了白色的点
    non_white_points = points[~np.all(points[:, 3:6] == 255, axis=1)]
    # 设置点云的坐标、颜色
    pcd.points = o3d.utility.Vector3dVector(non_white_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(non_white_points[:, 3:6])  # 颜色通常在[0, 255]范围内，我的输出不在，需要归一化到[0, 1]

    # 使用统计离群值去除滤波器,nb_neighbors 参数表示用于计算每个点周围邻域的点数越大滤波越厉害；std_ratio 参数表示标准差的倍数，越小滤波越厉害
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)

    return pcd


def test_depth():
    # 获取图像和深度图
    depth_filename = './results/blender/test/depth_000.pfm'
    rgb_filename = './results/blender/test/000.png'
    root_dir = './data/nerf_synthetic/lego'
    # 下面读出来是BGR的颜色，转换成rgb
    image = cv2.imread(rgb_filename)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_data, scale = read_pfm(depth_filename)
    H, W = image.shape[:2]

    # 获取相机内外参数
    with open(os.path.join(root_dir,
                           f"transforms_test.json"), 'r') as f:
        meta = json.load(f)

    focal = 0.5 * W / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ])
    c2w = np.array(meta['frames'][0]['transform_matrix'])

    points_pcd = depth_image_to_point_cloud(rgb=image_rgb, depth=depth_data, scale=scale, K=K, c2w=c2w)


    # # 创建一个画板
    # fig = plt.figure()
    # # 添加子图2：显示深度图
    # min_depth = np.min(depth_data)
    # max_depth = np.max(depth_data)
    # # 将深度图像归一化到 [0, 1]
    # normalized_depth = (depth_data - min_depth) / (max_depth - min_depth)
    # ax2 = fig.add_subplot(111)
    # ax2.imshow(normalized_depth,  cmap='viridis')
    # ax2.set_title('Depth Map')
    # plt.show()

    o3d.visualization.draw_geometries([points_pcd])

if __name__ == '__main__':
    ply_file_path = "./results/blender/test/point_cloud_004.ply"
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    o3d.visualization.draw_geometries([point_cloud])









