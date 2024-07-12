# HierarCaps Code

## Setup

```
conda create -n hierarcaps python=3.9
conda activate hierarcaps
pip install -r requirements.txt
```

## Fine-tuning

```
python train.py
```

Run with `--help / -h` to see all arguments and default values.

You may also download [fine-tuned CLIP-B and CLIP-L checkpoints](https://drive.google.com/drive/folders/1s-f2L0pFzZXs2jCaIyfMa207iiNYkua3?usp=sharing).

## Inference

To run inference on HierarCaps:

```
python eval.py (-bc ...) (-w ...)    # quantitative evaluation (1K-item test set)
python qual.py -i imgs_dir (-bc ...) (-w ...)    # qualitative evaluation (expanded test candidate set)
```

Use optional `-bc` and `-w` flags to change which model is loaded (base pretrained model and fine-tuned weights respectively). For qualitative tests, `imgs_dir` is a directory containing the image files to evaluate on. Run with `--help / -h` to see all arguments and default values.
