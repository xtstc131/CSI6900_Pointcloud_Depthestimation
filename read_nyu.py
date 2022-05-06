from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import h5py
import open3d as o3d
from random import randint
from tqdm import tqdm
# data path
path_to_depth = './dataset/nyuv2/nyu_depth_v2_labeled.mat'
path_to_train = './dataset/nyuv2/train/'
# read mat file
def read_and_save_from_mat(save_files=False):
    f = h5py.File(path_to_depth)
    for idx, img in enumerate(f['images']):
        # read 0-th image. original format is [3 x 640 x 480], uint8
        # reshape
        idx_str = f'{idx:04d}'

        img_ = np.empty([480, 640, 3])
        img_[:, :, 0] = img[0, :, :].T
        img_[:, :, 1] = img[1, :, :].T
        img_[:, :, 2] = img[2, :, :].T
        img__ = img_.astype(np.uint8)

        # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
        depth = f['depths'][idx]
        # reshape for imshow
        depth_ = np.empty([480, 640])
        depth_ = depth.T

        depth_arr = (depth_ * 1e3).astype(np.uint16)
        depth_img = Image.fromarray(depth_arr)

        color = o3d.geometry.Image(img__)
        depth = o3d.geometry.Image(
            np.ascontiguousarray(depth_).astype(np.float32))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        if not save_files:
            plt.subplot(1, 2, 1)
            plt.title('NYU color image')
            plt.imshow(rgbd_image.color)
            plt.subplot(1, 2, 2)
            plt.title('NYU depth image')
            plt.imshow(rgbd_image.depth, cmap="magma_r")

            plt.show()

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                      [0, 0, -1, 0], [0, 0, 0, 1]])

        origin_points = np.asarray(pcd.points)
        downpcd = pcd.random_down_sample(0.00166667)
        down_points = np.asarray(downpcd.points)
        if not save_files:
            o3d.visualization.draw_geometries([downpcd])
            print(origin_points.shape)
            print(down_points.shape)
        if save_files:
            img_path = path_to_train + idx_str + '.png'
            depth_path = path_to_train + idx_str + '_depth.png'
            pcd_path = path_to_train + idx_str + '.pcd'
            print(img_path, depth_path, pcd_path)
            plt.imsave(img_path, img_/255.0)
            depth_img.save(depth_path)
            o3d.io.write_point_cloud(pcd_path, downpcd)


def read_and_save_from_mat_rand(save_files=False):
    f = h5py.File(path_to_depth)
    idx = randint(0,1448)
    # read 0-th image. original format is [3 x 640 x 480], uint8
    # reshape
    img = f['images'][idx]
    idx_str = f'{idx:04d}'
    img_ = np.empty([480, 640, 3])
    img_[:, :, 0] = img[0, :, :].T
    img_[:, :, 1] = img[1, :, :].T
    img_[:, :, 2] = img[2, :, :].T
    img__ = img_.astype(np.uint8)

    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
    depth = f['depths'][idx]
    # reshape for imshow
    depth_ = np.empty([480, 640])
    depth_ = depth.T

    depth_arr = (depth_ * 1e3).astype(np.uint16)
    depth_img = Image.fromarray(depth_arr)

    color = o3d.geometry.Image(img__)
    depth = o3d.geometry.Image(
        np.ascontiguousarray(depth_).astype(np.float32))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)
    if not save_files:
        plt.subplot(1, 2, 1)
        plt.title('NYU color image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('NYU depth image')
        plt.imshow(rgbd_image.depth, cmap="magma_r")

        plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, 1]])

    origin_points = np.asarray(pcd.points)
    downpcd = pcd.random_down_sample(0.00166667)
    down_points = np.asarray(downpcd.points)
    if not save_files:
        o3d.visualization.draw_geometries([downpcd])
        print(origin_points.shape)
        print(down_points.shape)
    if save_files:
        img_path = path_to_train + idx_str + '.png'
        depth_path = path_to_train + idx_str + '_depth.png'
        pcd_path = path_to_train + idx_str + '.pcd'
        print(img_path, depth_path, pcd_path)
        plt.imsave(img_path, img_/255.0)
        depth_img.save(depth_path)
        o3d.io.write_point_cloud(pcd_path, downpcd)


def create_files_list():
    txt_file_name = 'nyu_train_file_list.txt'
    lines = []
    for i in range(1449):
        idx_str = f'{i:04d}'
        img_name = idx_str + '.png'
        pcd_name = idx_str + '.pcd'
        depth_name = idx_str + '_depth.png'
        line = img_name + ' ' + depth_name + ' ' + pcd_name + '\n'
        lines.append(line)
    with open(txt_file_name, 'w') as fd:
        fd.writelines(lines)


read_and_save_from_mat_rand()
# create_files_list()
