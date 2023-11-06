import os
import cv2

from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils.utils import *

from load_datasets import dataset_dict
from load_datasets.depth_utils import *

torch.backends.cudnn.benchmark = True
import json

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='./data/my_airplane_data/airplane',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='airplane',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[960, 544],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, default='./ckpts/airplane/model-epoch=01-val/psnr=27.16.ckpt',
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=True, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    parser.add_argument('--save_point_cloud', default=True, action="store_true",
                        help='whether to save point cloud')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)
        # 返回的是这些信息
        # results[f'weights_{typ}'] = weights
        # results[f'opacity_{typ}'] = weights_sum
        # results[f'z_vals_{typ}'] = z_vals
        # results[f'rgb_{typ}'] = rgb_map
        # results[f'depth_{typ}'] = depth_map


        #下面这个步骤就是将结果从gpu转移到cpu，对上面返回的字典没有丝毫改变
        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    nerf_coarse = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=6*args.N_emb_dir+3)
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}

    if args.N_importance > 0:
        nerf_fine = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                         in_channels_dir=6*args.N_emb_dir+3)
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine

    imgs, depth_maps, psnrs = [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(0, len(dataset), 4)):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        # 就是获取图像的颜色值
        img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)

        if args.save_depth:
            # 获取深度值
            depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(os.path.join(dir_name, f'depth_{i:03d}'), 'wb') as f:
                    f.write(depth_pred.tobytes())

        # if args.save_point_cloud:
        #     # 获取相机内外参数
        #     with open(os.path.join(args.root_dir,
        #                            f"transforms_{args.split}.json"), 'r') as f:
        #         meta = json.load(f)
        #     focal = 0.5 * w / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
        #     K = np.array([
        #         [focal, 0, 0.5 * w],
        #         [0, focal, 0.5 * h],
        #         [0, 0, 1]
        #     ])
        #     c2w = np.array(meta['frames'][0]['transform_matrix'])
        #     depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
        #     points_pcd = depth_image_to_point_cloud(rgb=img_pred, depth=depth_pred, K=K, c2w=c2w)
        #     o3d.io.write_point_cloud(os.path.join(dir_name, f'point_cloud_{i:03d}.ply'), points_pcd)


        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [psnr(img_gt, img_pred).item()]

    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)

    if args.save_depth:
        min_depth = np.min(depth_maps)
        max_depth = np.max(depth_maps)
        # 获取完深度的最大值最小值，然后进行归一化，有助于可视化深度信息
        depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
        depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_depth.gif'), depth_imgs_, fps=30)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
