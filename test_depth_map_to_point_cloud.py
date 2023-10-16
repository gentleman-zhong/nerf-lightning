import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import os
import json
from path import Path
from tqdm import tqdm


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


def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


def depth_image_to_point_cloud(rgb, depth, scale, K, c2w):
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
    # R = np.ravel(rgb[:, :, 0])
    # G = np.ravel(rgb[:, :, 1])
    # B = np.ravel(rgb[:, :, 2])
    points = np.transpose(np.vstack((position[0:3, :], R, G, B)))

    return points


def build_point_cloud(dataset_path, scale, view_ply_in_world_coordinate):
    K = np.fromfile(os.path.join(dataset_path, "K.txt"), dtype=float, sep="\n ")
    K = np.reshape(K, newshape=(3, 3))
    image_files = sorted(Path(os.path.join(dataset_path, "images")).files('*.png'))
    depth_files = sorted(Path(os.path.join(dataset_path, "depth_maps")).files('*.png'))

    if view_ply_in_world_coordinate:
        poses = np.fromfile(os.path.join(dataset_path, "poses.txt"), dtype=float, sep="\n ")
        poses = np.reshape(poses, newshape=(-1, 4, 4))
    else:
        poses = np.eye(4)

    for i in tqdm(range(0, len(image_files))):
        image_file = image_files[i]
        depth_file = depth_files[i]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i])
        else:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses)
        save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"
        save_ply_path = os.path.join(dataset_path, "point_clouds")

        if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)


def draw_point_cloud(ax, points, title, axes=[0, 1, 2],point_size = 0.1, xlim3d=None, ylim3d=None, zlim3d=None):
    axes_limits = [
        [-20,80],
        [-20,20],
        [-3,3]
    ]
    axes_str = ['X','Y','Z']
    ax.grid(False)  #不会画出网格
    ax.scatter(*np.transpose(points[:, axes]), s=point_size, c=points[:, 3], cmap='gray')
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)


if __name__ == '__main__':
    # 获取图像和深度图
    depth_filename = './results/blender/test/depth_000.pfm'
    rgb_filename = './results/blender/test/000.png'
    root_dir = './data/nerf_synthetic/lego'
    # 下面读出来是BGR的颜色，转换成rgb
    image = cv2.imread(rgb_filename)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_data, scale = read_pfm(depth_filename)
    H, W = image.shape[:2]

    # 获取点云
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
    points = depth_image_to_point_cloud(rgb=image_rgb, depth=depth_data, scale=scale, K=K, c2w=c2w)

    # 创建一个画板
    fig = plt.figure()

    # 添加子图1：显示图像
    ax1 = fig.add_subplot(131)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Image')

    # 添加子图2：显示深度图
    min_depth = np.min(depth_data)
    max_depth = np.max(depth_data)
    # 将深度图像归一化到 [0, 1]
    normalized_depth = (depth_data - min_depth) / (max_depth - min_depth)
    ax2 = fig.add_subplot(132)
    ax2.imshow(normalized_depth,  cmap='viridis')
    ax2.set_title('Depth Map')

    # 添加子图3：显示点云
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.view_init(90, 90)  # 指定看的视角,俯仰角和方位角
    # 筛选出非白色的点
    non_white_points = points[~np.all(points[:, 3:6] == 255, axis=1)]
    # 绘制非白色的点云
    ax3.scatter(non_white_points[:, 0], non_white_points[:, 1], non_white_points[:, 2],
                c=non_white_points[:, 3:6] / 255.0)
    # ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
    #             c=points[:, 3:6] / 255.0)
    ax3.set_title('Point Cloud')

    plt.show()








