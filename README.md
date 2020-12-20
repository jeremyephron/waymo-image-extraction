waymo-image-extraction
======================

This repo contains a script for extracting images from the Waymo open dataset. 
Read the file docstring of [extract_images.py](extract_images.py) and run 
`python3 extract_images.py --help` (after setup) for more information on how 
to use the script.

Prerequisites
-------------

First, you'll need to request access to Waymo's dataset from their 
[website](https://waymo.com/open/). This script downloads the files 
from Google Cloud Storage (GCS), so you'll need to ensure you have been 
granted GCS access before running.

Second, you'll need to have `gsutil` installed. Visit Google's 
[site](https://cloud.google.com/storage/docs/gsutil_install) for more 
information.

Third, you'll need to have Python 3 (>= 3.8 recommended) and `pip` installed.

Quickstart
----------

Clone this repository, navigate to it, and run 
```
pip3 install -r requirements.txt
```

If you want to sample every 10 frames (which is 1 frame per second) from only 
the front camera, from the training set, and output them in the directory 
`waymo-images/train`, run

```
python3 extract_images.py waymo-images/train --views front --split training \
    --sample 10
```
