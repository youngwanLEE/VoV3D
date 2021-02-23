# Dataset Preparation

## Kinetics-400
Since some urls of [Kinetics-400 dataset](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics) are expired, we cannot collect full dataset.\
Alternatively, we downloaded the dataset via a link from the [facebookresearch/video-nonlocal-net#67](https://github.com/facebookresearch/video-nonlocal-net/issues/67).\
However, the link is also expired and thus we share our dataset link.

[[`Kinetics-400 dataset link`](https://dl.dropbox.com/s/419u0zljf2brsbt/compress.tar.gz)]
- training set :  234,619 videos
- validation set : 19,716 videos 

**Important** : This dataset is smaller than the original Kinetics-400 dataset.

After all the videos were downloaded, prepare the csv files for training and validation as `train.csv`, `val.csv`.
The format of the csv file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

For an example sample, we also share our csv.files including our absolute path. 
So you have to replace our absolute path to your path.

- [[`train.csv`](https://dl.dropbox.com/s/1c1yweibf6rbpff/train.csv)]
- [[`val.csv`](https://dl.dropbox.com/s/88tz1mwqjzewcrp/val.csv)]

For dataset specific issues, please reach out to the [dataset provider](https://deepmind.com/research/open-source/kinetics).


## Something-Something-V1 (SSv1)


**Step 1.** Download the dataset and annotations from [dataset provider](https://20bn.com/datasets/something-something/v1). \
 *Note that unlike SSv2 (video files), SSv1 is comprised of extracted RGB frames.
 
 ```shell
cd path/to/the/data/ssv1
cat 20bn-something-something-v1-?? | tar zx
```

**Step 2.** Download the *frame list* from the following links: ([train](https://dl.dropbox.com/s/7jk9s5syt925epo/train.csv), [val](https://dl.dropbox.com/s/y4x0yewm6wwvak3/val.csv)).


Please put all annotation json files and the frame lists in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.

```shell
cp *.csv 20bn-something-something-v1/
```

## Something-Something-V2 (SSv2)

**Step 1.** Download the dataset and annotations from [dataset provider](https://20bn.com/datasets/something-something).

 ```shell
cd path/to/the/data/ssv2
cat 20bn-something-something-v2-?? | tar zx
```

**Step 2.** Download the *frame list* from the following links: ([train](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv), [val](https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv)).

**Step 3.** Extract the frames at 30 FPS by using [`video_extractor_sthv2.py`](vov3d/datasets/video_extractor_sthv2.py). \
 *Note that check the number of the extracted frames : **`25,209,271`** \
 *If you get the fewer number of frames than mine, remove the folder and try it again.
```shell
# Extract videos to frames
python vov3d/datasets/video_extractor_sthv2.py \
--video_dir path/to/the/data/ssv2/20bn-something-something-v2 \
--frame_dir path/to/the/data/ssv2/20bn-something-something-v2-SF-frame

# Check the number of extracted frames : 25209271
cd path/to/the/data/ssv2/20bn-something-something-v2-SF-frames
find . -type f | wc -l
```
 
 Please put all annotation json files and the frame lists in the same folder, and set `DATA.PATH_TO_DATA_DIR` to the path. Set `DATA.PATH_PREFIX` to be the path to the folder containing extracted frames.


## ETC
For AVA and Charades, please follow PySlowFast's [instruction](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md#ava).
