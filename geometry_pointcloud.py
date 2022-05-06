import glob
import os
import struct
import subprocess
import sys
from random import randrange
from statistics import geometric_mean

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import objectron.dataset.box as Box
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
from objectron.schema import object_pb2 as object_protocol

# matplotlib.use( 'tkagg' )
# I'm running this Jupyter notebook locally. Manually import the objectron module.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The AR Metadata captured with each frame in the video
# The annotations are stored in protocol buffer format.


def get_geometry_data(geometry_filename):
    sequence_geometry = []
    with open(geometry_filename, 'rb') as pb:
        proto_buf = pb.read()

        i = 0
        frame_number = 0

        while i < len(proto_buf):
            # Read the first four Bytes in little endian '<' integers 'I' format
            # indicating the length of the current message.
            msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
            i += 4
            message_buf = proto_buf[i:i + msg_len]
            i += msg_len
            frame_data = ar_metadata_protocol.ARFrame()
            frame_data.ParseFromString(message_buf)

            transform = np.reshape(frame_data.camera.transform, (4, 4))
            projection = np.reshape(
                frame_data.camera.projection_matrix, (4, 4))
            view = np.reshape(frame_data.camera.view_matrix, (4, 4))
            depth_map = frame_data.depth_data.depth_data_map
            if len(depth_map) != 0:
                print(depth_map)

            current_points = [np.array([v.x, v.y, v.z])
                              for v in frame_data.raw_feature_points.point]
            current_points = np.array(current_points)

            sequence_geometry.append(
                (transform, projection, view, current_points))
    return sequence_geometry


def get_frame_annotation(annotation_filename):
    """Grab an annotated frame from the sequence."""
    result = []
    instances = []
    with open(annotation_filename, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())

        object_id = 0
        object_rotations = []
        object_translations = []
        object_scale = []
        num_keypoints_per_object = []
        object_categories = []
        annotation_types = []

        # Object instances in the world coordinate system, These are stored per sequence,
        # To get the per-frame version, grab the transformed keypoints from each frame_annotation
        for obj in sequence.objects:
            rotation = np.array(obj.rotation).reshape(3, 3)
            translation = np.array(obj.translation)
            scale = np.array(obj.scale)
            points3d = np.array([[kp.x, kp.y, kp.z] for kp in obj.keypoints])
            instances.append((rotation, translation, scale, points3d))

        # Grab teh annotation results per frame
        for data in sequence.frame_annotations:
            # Get the camera for the current frame. We will use the camera to bring
            # the object from the world coordinate to the current camera coordinate.
            transform = np.array(data.camera.transform).reshape(4, 4)
            view = np.array(data.camera.view_matrix).reshape(4, 4)
            intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)
            projection = np.array(data.camera.projection_matrix).reshape(4, 4)

            keypoint_size_list = []
            object_keypoints_2d = []
            object_keypoints_3d = []
            for annotations in data.annotations:
                num_keypoints = len(annotations.keypoints)
                keypoint_size_list.append(num_keypoints)
                for keypoint_id in range(num_keypoints):
                    keypoint = annotations.keypoints[keypoint_id]
                    object_keypoints_2d.append(
                        (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
                    object_keypoints_3d.append(
                        (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
                num_keypoints_per_object.append(num_keypoints)
                object_id += 1
            result.append((object_keypoints_2d, object_keypoints_3d,
                          keypoint_size_list, view, projection))

    return result, instances


def grab_frame(video_file, frame_ids):
    """Grab an image frame from the video file."""
    frames = []
    capture = cv2.VideoCapture(video_file)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    frame_size = width * height * 3

    for frame_id in frame_ids:
        frame_filter = r'select=\'eq(n\,{:d})\''.format(frame_id)
        command = [
            'ffmpeg', '-i', video_file, '-f', 'image2pipe', '-vf', frame_filter,
            '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-vsync', 'vfr', '-', '-loglevel', 'quiet'
        ]
        pipe = subprocess.Popen(
            command, stdout=subprocess.PIPE, bufsize=151 * frame_size)
        current_frame = np.frombuffer(
            pipe.stdout.read(frame_size), dtype='uint8').reshape(height, width, 3)
        pipe.stdout.flush()

        frames.append(current_frame)
    return frames


def project_points(points, projection_matrix, view_matrix, width, height):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

    p_3d = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1).T

    p_3d_cam = np.matmul(view_matrix, p_3d)

    p_2d_proj = np.matmul(projection_matrix, p_3d_cam)
    # Project the points
    p_2d_ndc = p_2d_proj[:-1, :] / p_2d_proj[-1, :]
    p_2d_ndc = p_2d_ndc.T

    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    # ndc = [Y, X, Z]
    x = p_2d_ndc[:, 1]
    y = p_2d_ndc[:, 0]
    z = p_2d_ndc[:, 2]
    # print( (z - z.min()) / (z.max() - z.min()))
    zNorm = (z - z.min()) / (z.max() - z.min()) * 255
    pc_depth_map = zNorm.astype(int)
    pixels = np.copy(p_2d_ndc)
    pixels[:, 0] = ((1 + x) * 0.5) * width
    pixels[:, 1] = ((1 + y) * 0.5) * height
    pixels = pixels.astype(int)
    return pixels, pc_depth_map


def show_point_cloud(geometry_pb, annotaion_pb, video_file):
    # Grab some frames from the video file.
    frame_ids = [101, 105, 123, 144]
    num_frames = len(frame_ids)
    sequence_geometry = get_geometry_data(geometry_pb)
    frames = grab_frame(video_file, frame_ids)
    annotation_data, instances = get_frame_annotation(annotaion_pb)
    fig, ax = plt.subplots(2, num_frames, figsize=(60, 60))

    for i in range(len(frame_ids)):
        frame_id = frame_ids[i]
        image = frames[i]
        height, width, _ = image.shape
        points_2d, points_3d, num_keypoints, frame_view_matrix, frame_projection_matrix = annotation_data[
            frame_id]
        num_instances = len(num_keypoints)
        points_depth = []
        # As covered in our previous tutorial, we can directly grab the 2D projected points from the annotation
        # file. The projections are normalized, so we scale them with the image's height and width to get
        # the pixel value.
        # The keypoints are [x, y, d] where `x` and `y` are normalized (`uv`-system)\
        # and `d` is the metric distance from the center of the camera. Convert them
        # keypoint's `xy` value to pixel.
        points_2d = np.split(points_2d, np.array(np.cumsum(num_keypoints)))
        points_2d = [points.reshape(-1, 3) for points in points_2d]
        points_depth = [keypoint for keypoint in points_2d]
        points_2d = [
            np.multiply(keypoint, np.asarray(
                [width, height, 1.], np.float32)).astype(int)
            for keypoint in points_2d
        ]

        points_2d = []
        # Now, let's compute the box's vertices in 3D, then project them back to 2D:
        for instance_id in range(num_instances):
            # The annotation contains the box's transformation and scale in world coordinate system
            # Here the instance_vertices_3d are the box vertices in the "BOX" coordinate, (i.e. it's a unit box)
            # and has to be transformed to the world coordinate.
            instance_rotation, instance_translation, instance_scale, instance_vertices_3d = instances[
                instance_id]

            box_transformation = np.eye(4)
            box_transformation[:3, :3] = np.reshape(instance_rotation, (3, 3))
            box_transformation[:3, -1] = instance_translation
            vertices_3d = instance_vertices_3d * instance_scale.T
            # Homogenize the points
            vertices_3d_homg = np.concatenate(
                (vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T
            # Transform the homogenious 3D vertices with the box transformation
            box_vertices_3d_world = np.matmul(
                box_transformation, vertices_3d_homg)

            # If we transform these vertices to the camera frame, we get the 3D keypoints in the annotation data
            # i.e. vertices_3d_cam == points_3d
            vertices_3d_cam = np.matmul(
                frame_view_matrix, box_vertices_3d_world)
            vertices_2d_proj = np.matmul(
                frame_projection_matrix, vertices_3d_cam)

            # Project the points
            points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
            points2d_ndc = points2d_ndc.T

            # Convert the 2D Projected points from the normalized device coordinates to pixel values
            x = points2d_ndc[:, 1]
            y = points2d_ndc[:, 0]
            points2d = np.copy(points2d_ndc)
            points2d[:, 0] = ((1 + x) * 0.5) * width
            points2d[:, 1] = ((1 + y) * 0.5) * height
            points_2d.append(points2d.astype(int))
            # points2d are the projected 3D points on the image plane.

        # We can also use the above pipeline to visualize the scene point-cloud on the image.
        # First, let's grab the point-cloud from the geometry metadata
        transform, projection, view, scene_points_3d = sequence_geometry[frame_id]
        scene_points_2d, pc_depth_map = project_points(
            scene_points_3d, projection, view, width, height)
        # Note these points all have estimated depth, which can double as a sparse depth map for the image.
        print(scene_points_2d.shape)
        for point_id in range(scene_points_2d.shape[0]):
            point_depth = int(pc_depth_map[point_id])
            pcolor = (point_depth, point_depth, point_depth)
            cv2.circle(image, (scene_points_2d[point_id, 0], scene_points_2d[point_id, 1]), 10,
                       pcolor, -1)
        ax[0, i].grid(False)
        ax[0, i].imshow(image)
        ax[0, i].get_xaxis().set_visible(False)
        ax[0, i].get_yaxis().set_visible(False)
        ax[1, i].hist(pc_depth_map)
    fig.tight_layout()
    plt.show()
    frame_id = 100

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    transform, projection, view, points_3d = sequence_geometry[frame_id]
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])


if __name__ == '__main__':
    train_folders = glob.glob('./dataset/train/camera/**/**/')

    rand_idx = randrange(len(train_folders))
    geometric_pb = train_folders[rand_idx] + 'geometry.pbdata'
    annotation_pb = train_folders[rand_idx] + 'annotation.pbdata'
    video_file = train_folders[rand_idx] + 'video.MOV'
    show_point_cloud(geometric_pb, annotation_pb, video_file)
    # for i in range(len(bike_train_folders)):
    #     geometric_pb = bike_train_folders[i] + 'geometry.pbdata'
    #     get_geometry_data(geometric_pb)
    # print('finished!')
