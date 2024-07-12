# HierarCaps Data

## Train set

`hierarcaps_train.csv.gz`: Over 70K items consisting of positive and negative caption hierarchies, constructed from image captions from <a href="https://huggingface.co/datasets/google-research-datasets/conceptual_captions">Conceptual Captions (CC)</a>. Our method only uses textual data for training, but corresponding images can be retrieved by cross-referencing captions in CC.

## Test set
`hierarcaps_test.csv`: 1K paired images and caption hierarchies, used for calculating quantitative metrics. Images are provided as URL links to images from <a href="https://cocodataset.org/">MS-COCO</a>.

`hierarcaps_test_expanded.csv.gz`: Over 23K automatically-generated (uncurated) caption hierarchies based on the MS-COCO val set. All (positive) textual items from these are used as candidates for our qualitative results.