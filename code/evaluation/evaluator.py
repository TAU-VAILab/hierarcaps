import os
from transformers import CLIPModel, AutoProcessor
import numpy as np


class Evaluator:

    def __init__(self, args, dataset_name, log):
        self.test_csv_filename = args.test_csv_filename
        self.test_images_dir = args.test_images_dir
        self.weights_fn = args.weights
        self.dataset_name = dataset_name
        self.log = log
        self.steps = args.steps
        self.base_checkpoint = args.base_checkpoint
        self.device = args.device

        self._load_model()

        self._load_data()

    def _load_model(self):
        print('Loading CLIP model for evaluation')
        if self.weights_fn is None:
            ckpt = self.base_checkpoint
            print('Using base checkpoint:', self.base_checkpoint)
        else:
            print('Loading CLIP weights for evaluation from:', self.weights_fn)
            ckpt = self.weights_fn
        self.clip = CLIPModel.from_pretrained(ckpt).to(self.device)
        self.clip.eval()
        self.proc = AutoProcessor.from_pretrained(ckpt)
        print('CLIP model loaded')

    def _report_metrics(self, metrics, display_metrics=None, skip_header=False):
        # metrics: dict {metric_name=>val}
        # display_metrics: if printed value should be formatted differently than saved value
        if not skip_header:
            print(f'Metrics for {self.dataset_name}:')
        for k, v in (metrics if display_metrics is None else display_metrics).items():

            if type(v) is float or type(v) is np.float64 or type(v) is np.float32:
                print(f"\t{k.title()}: {v:.4f}")
            else:
                print(f"\t{k.title()}: {v}")

        for k, v in metrics.items():
            self.log[k] = v

    def _load_data(self):
        raise NotImplementedError

    def run(self):

        raise NotImplementedError
