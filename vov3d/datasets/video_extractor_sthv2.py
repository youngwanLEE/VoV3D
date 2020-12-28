#!/usr/bin/env python3
# Copyright Youngwan Lee (ETRI). All Rights Reserved.

import os
import threading
import argparse
import sys

NUM_THREADS = 100


def parse_args():

    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--video_dir",
        dest="video_dir",
        help="something-something V2 video directory path",
        default="/home/lsrock1/data/sth-sth-v2/20bn-something-something-v2",
        type=str,
    )
    parser.add_argument(
        "--frame_dir",
        dest="frame_dir",
        help="extracted frame directory path",
        default="/home/lsrock1/data/sth-sth-v2/20bn-something-something-v2-SF-frames",
        type=str,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def extract(video, args):

    # following the instruction in slowfast/dataset/DATASET.md
    # `ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"`
    cmd = (
        'ffmpeg -i "{}/{}" -threads 1 -r 30 -q:v 1 "{}/{}/{}_%06d.jpg"'.format(
            args.video_dir, video, args.frame_dir, video[:-5], video[:-5]
        )
    )
    os.system(cmd)


def target(video_list, args):
    for video in video_list:
        os.makedirs(os.path.join(args.frame_dir, video[:-5]))
        extract(video, args)


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.video_dir):
        raise ValueError("Please download videos and set VIDEO_ROOT variable.")
    if not os.path.exists(args.frame_dir):
        os.makedirs(args.frame_dir)

    video_list = os.listdir(args.video_dir)
    splits = list(split(video_list, NUM_THREADS))

    threads = []
    for i, split in enumerate(splits):
        thread = threading.Thread(target=target, args=(split, args))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
