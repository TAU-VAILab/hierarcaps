import os
from transformers import CLIPModel, AutoProcessor
import torch
from glob import glob
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


class QualitativeEvaluator:

    def __init__(self, args):
        self.weights_fn = args.weights
        self.imgs_dir = args.imgs_dir
        self.logfile = args.logfile
        self.csvfile = args.csvfile
        self.steps = args.steps
        self.candidates_filename = args.candidates_filename

        self._load_imgs()
        self._load_captions()
        self._load_model()

    def _load_imgs(self):
        print("Loading images")
        self.img_fns = glob(os.path.join(self.imgs_dir, '*'))
        assert len(self.img_fns) > 0, 'No images found'
        self.imgs = [Image.open(fn).convert('RGB') for fn in self.img_fns]
        print("Images loaded:", len(self.imgs))

    def _load_captions(self):
        print("Loading captions (HierarCaps; expanded candidates)")
        data_fn = self.candidates_filename
        assert os.path.exists(data_fn), f'Missing file: {data_fn}'
        df = pd.read_csv(data_fn)
        self.caps = list({x.strip() for row in df.itertuples()
                          for x in row.positive.split('=>')})
        print('Unique captions:', len(self.caps))

    def _load_model(self):
        print('Loading CLIP model for qual eval')
        ckpt = 'openai/clip-vit-base-patch32'
        if self.weights_fn is not None:
            print('Loading CLIP weights for evaluation from:', self.weights_fn)
            ckpt = self.weights_fn
        self.clip = CLIPModel.from_pretrained(ckpt).to('cuda')
        self.clip.eval()
        self.proc = AutoProcessor.from_pretrained(ckpt)
        print('CLIP model loaded')

    @torch.no_grad()
    def run(self):

        log = ""

        print(f'Embedding {len(self.imgs)} images...')
        inp = self.proc(images=self.imgs, return_tensors="pt").to('cuda')
        VE = self.clip.get_image_features(**inp)
        VE = VE / VE.norm(dim=-1)[:, None]
        print('Images embedded; embedding shape:', VE.shape)

        print(f'Scoring {len(self.caps)} captions...')
        sims = []
        radii = []
        radii_mean = []

        inp = self.proc(text=[''], padding=True,
                        truncation=True, return_tensors="pt").to('cuda')
        root = self.clip.get_text_features(**inp)
        root = root / root.norm(dim=-1)[:, None]
        root = root[0]

        for cap in tqdm(self.caps, desc="Scoring captions"):
            inp = self.proc(text=[cap], padding=True,
                            truncation=True, return_tensors="pt").to('cuda')

            TE = self.clip.get_text_features(**inp)
            TE = TE / TE.norm(dim=-1)[:, None]
            TE = TE[0]

            radius = 1 - (TE @ root).item()
            radii.append(radius)

            sims.append((VE @ TE).cpu().numpy())
        S = np.stack(sims)  # size: (n_caps, n_imgs)
        print('Similarity matrix:', S.shape)

        print("Building table...")
        df = pd.DataFrame({'text': self.caps})
        if len(radii) > 0:
            df['radius'] = radii
        else:
            df['radius'] = 1.
        if len(radii_mean) > 0:
            df['radii_mean'] = radii_mean
        for i in range(len(self.imgs)):
            df[f'sim_{i}'] = S[:, i]
        print("Table built")

        a, b = df.radius.min(), df.radius.max()
        for i in range(len(self.imgs)):
            print()
            fn = os.path.basename(self.img_fns[i])
            log += f"Hierarchical predictions for image {i}: ({fn})\n"

            ddf = df.sort_values(by=f'sim_{i}', ascending=False)

            last = None
            for thresh in np.linspace(a, b, self.steps):
                subres_df = ddf[ddf.radius <= thresh]
                if len(subres_df) == 0:
                    log += str(round(thresh, 3)) + '\t---\n'
                else:
                    t = subres_df.text.iloc[0].strip()
                    s = subres_df[f'sim_{i}'].iloc[0]
                    if t == last:
                        pass  # print(round(thresh, 3), '\t', '...')
                    else:
                        log += f'{thresh:.3f}\t{s:.3f}\t{t}\n'
                    last = t
            log += '\n'

        print(log)

        print("Saving logs to:", self.logfile)
        with open(self.logfile, 'w') as f:
            f.write(log)
        print("Saved")

        print("Saving table to:", self.csvfile)
        df.to_csv(self.csvfile, index=False)
        print("Saved")
