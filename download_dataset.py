import os
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


def download_dataset(public_url, video_ids, is_train=True):
    if is_train:
        print("Download trainning data")
    else:
        print("Download testing data")

    for i in tqdm(range(len(video_ids))):
        video_filename = public_url + "/videos/" + video_ids[i] + "/video.MOV"
        metadata_filename = public_url + "/videos/" + \
            video_ids[i] + "/geometry.pbdata"
        annotation_filename = public_url + \
            "/annotations/" + video_ids[i] + ".pbdata"

        if is_train:
            folder_path = Path('dataset/train')
        else:
            folder_path = Path('dataset/test')

        folder_path = folder_path.joinpath(video_ids[i])

        video_path = folder_path.joinpath("video.MOV")
        metadata_path = folder_path.joinpath("geometry.pbdata")
        annotation_path = folder_path.joinpath("annotation.pbdata")
        # video.content contains the video file.
        video = requests.get(video_filename)
        metadata = requests.get(metadata_filename)

        # Please refer to Parse Annotation tutorial to see how to parse the annotation files.
        annotation = requests.get(annotation_filename)
        folder_path.mkdir(parents=True, exist_ok=True)

        file = open(video_path, "wb")
        file.write(video.content)
        file.close()

        file = open(metadata_path, "wb")
        file.write(metadata.content)
        file.close()

        file = open(annotation_path, "wb")
        file.write(annotation.content)
        file.close()


if __name__ == '__main__':
    public_url = "https://storage.googleapis.com/objectron"

    class_list = ["bike", "book", "bottle", "camera",
                  "cereal_box", "chair", "cup", "laptop", "shoe"]

    sub_url_train = "/v1/index/" + class_list[5] + "_annotations_train"
    sub_url_test = "/v1/index/" + class_list[5] + "_annotations_test"
    blob_path_train = public_url + sub_url_train
    blob_path_test = public_url + sub_url_test

    video_ids_train = requests.get(blob_path_train).text
    video_ids_train = video_ids_train.split('\n')

    video_ids_test = requests.get(blob_path_test).text
    video_ids_test = video_ids_test.split('\n')
    print(len(video_ids_test))
    download_dataset(public_url, video_ids_train, True)
    download_dataset(public_url, video_ids_test, False)
