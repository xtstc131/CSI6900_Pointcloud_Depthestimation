# examples/Python/Basic/rgbd_nyu.py

import open3d as o3d
import numpy as np
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


if __name__ == "__main__":
    print("Read NYU dataset")
    # Open3D does not support ppm/pgm file yet. Not using o3d.io.read_image here.
    # MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
    color_raw = mpimg.imread("dataset/nyuv2/test/r-1295838170.688547-3677604826.ppm")
    depth_raw = read_nyu_pgm("dataset/nyuv2/test/d-1295838170.679708-3677158570.pgm")
    print(color_raw.dtype)
    print(depth_raw.dtype)
    color = o3d.geometry.Image(color_raw)
    depth = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_nyu_format(color, depth)
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('NYU grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('NYU depth image')
    plt.imshow(rgbd_image.depth,cmap="magma_r")
    plt.show()
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    origin_points = np.asarray(pcd.points)
    downpcd = pcd.random_down_sample(0.002)
    down_points = np.asarray(downpcd.points)
    # downpcd.paint_uniform_color([0, 0, 0])
    print(origin_points.shape)
    print(down_points.shape)
    o3d.visualization.draw_geometries([downpcd])
