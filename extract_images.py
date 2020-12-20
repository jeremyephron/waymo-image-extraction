"""
File: extract_images.py
-----------------------
This script extracts camera images from the Waymo open dataset, and allows the 
user to specify from which camera they want to save images, at what rate they 
want to sample the images, and which split (train, test, or val) they want to
save images from.

Example usage:

    python3 extract_images.py waymo-images/train --views front side_left \
        --split training --sample 10

This will:
    - save every 10 frames
    - from the front and side left cameras
    - from the training set
    - in the folder 'waymo-images/train'

The video is shot at 10 fps, so every 10 frames is equivalent to sampling 1 
frame per second.

"""

import argparse
import os
import subprocess
import tempfile
from urllib.parse import urlparse

from PIL import Image
import tensorflow.compat.v1 as tf
import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset

tf.enable_eager_execution()

TEMP_DIR = tempfile._get_default_tempdir()
WAYMO_DATASET_BUCKET = 'gs://waymo_open_dataset_v_1_2_0_individual_files'

CODE_TO_VIEW = {
    1: 'front',
    2: 'front_left',
    3: 'front_right',
    4: 'side_left',
    5: 'side_right'
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_path',
        type=str,
        help='directory path in which to output the images'
    )
    parser.add_argument(
        '--views',
        nargs='+',
        default=CODE_TO_VIEW.values(),
        choices=CODE_TO_VIEW.values(),
        help='which camera views to use'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='training',
        choices=['training', 'testing', 'validation'],
        help='use the training, testing, or validation split'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=1,
        help='frame sample increment (video is 10 fps)'
    )
    return parser.parse_args()


def main(output_path, views, split, sample):
    os.makedirs(output_path, exist_ok=True)

    urls = get_record_urls(split)
    for url in tqdm.tqdm(urls, total=len(urls), desc='urls'):
        fname = urlparse(url)
        fname = os.path.basename(fname.path)

        # Download the record
        temp_record_path = os.path.join(
            TEMP_DIR, next(tempfile._get_candidate_names())
        )
        subprocess.call(
            ['gsutil', 'cp', url, temp_record_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        extract_images_from_record(
            temp_record_path, sample, views, output_path
        )
        
        # Delete the record
        subprocess.call(['rm', '-f', temp_record_path])


def get_record_urls(split):
    """Returns the GCS urls of each record in the specified split."""

    stream = os.popen(f'gsutil ls {WAYMO_DATASET_BUCKET}/{split}')
    urls = list(filter(None, stream.read().split('\n')))
    return urls


def extract_images_from_record(record_path, sample, views, output_path):
    """Extracts images from the specified record at a given sample rate."""

    dataset = tf.data.TFRecordDataset(record_path, compression_type='')
    for i, data in enumerate(dataset):
        if i % sample != 0:
            continue

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        extract_images_from_frame(frame, views, output_path)


def extract_images_from_frame(frame, views, output_path):
    """Extracts images from the given frame object for the requested views."""

    for camera_image in frame.images:
        view = CODE_TO_VIEW[camera_image.name]
        if (camera_image.name not in CODE_TO_VIEW or view not in views):
            continue

        save_image(
            camera_image.image, view, frame.timestamp_micros, output_path
        )


def save_image(img, view, ts, output_dir):
    """Saves an image with the name format '<TIMESTAMP>_<VIEW>.jpeg'."""

    img = tf.image.decode_jpeg(img, channels=3)
    img = Image.fromarray(img.numpy(), 'RGB')
    img.save(os.path.join(output_dir, f'{ts}_{view}.jpeg'))


if __name__ == '__main__':
    main(**vars(get_args()))
