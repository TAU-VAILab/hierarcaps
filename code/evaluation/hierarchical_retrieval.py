import os
import pandas as pd
from tqdm.auto import tqdm
import torch
import numpy as np
from glob import glob
from PIL import Image
from itertools import groupby
from collections import defaultdict
from scipy.stats import kendalltau

from evaluation.evaluator import Evaluator


def calculate_Ro(preds, gts):
    # calculate order-aware recall for a single item

    indices = np.array([list(gts).index(p) for p in preds if p in gts])
    # ^ indices from 0 to 3
    if len(indices) == 0:
        return 0.
    signs = np.diff(indices) > 0
    # True for each increasing step
    # now look for longest string of Trues (+1):
    groups = [len(list(g)) + 1 for k, g in groupby(signs) if k]
    longest = 1 if len(groups) == 0 else max(groups)
    # normalize to be between 0 and 1:
    return longest / 4


def calculate_Ro_multi(preds, gts):
    # calculate order-aware recall for a single item
    # multi-reference: gts is list of 4-tuple chains
    s_gts = set(y for x in gts for y in x)

    def get_pred_index(p):
        for x in gts:
            if p in x:
                return list(x).index(p)
    indices = np.array([get_pred_index(p) for p in preds if p in s_gts])
    # ^ indices from 0 to 3
    if len(indices) == 0:
        return 0.
    # de-dup adjacents:
    indices_ = np.array([k for k, _ in groupby(indices)])
    signs = np.diff(indices_) > 0
    # True for each increasing step
    # now look for longest string of Trues (+1):
    groups = [len(list(g)) + 1 for k, g in groupby(signs) if k]
    longest = 1 if len(groups) == 0 else max(groups)
    # normalize to be between 0 and 1:
    return longest / 4


def fn2id(fn):
    base = os.path.splitext(os.path.basename(fn))[0]
    return int(base.split('_')[-1])


class HCEvaluator(Evaluator):
    def _load_data(self):

        data_fn = self.test_csv_filename
        assert os.path.exists(data_fn), f'Missing file: {data_fn}'

        img_dir = self.test_images_dir
        assert os.path.exists(img_dir), f'Missing image dir: {img_dir}'

        print('Loading data...')
        fns = glob(f'{img_dir}/*')
        id2fn = {
            fn2id(fn): fn
            for fn in fns
        }
        print(len(fns), 'images found')

        self.df = pd.read_csv(data_fn)
        print(f'{len(self.df)} records')

        self.df['image_id'] = self.df.image_url.str.extract(
            '_(\d+)')[0].astype(int)
        assert self.df.image_id.isin(id2fn).all(
        ), 'COCO image ID(s) missing image files'
        self.df['img_fn'] = self.df.image_id.map(id2fn)
        # ^ note: same image appears multiple times (has multiple references in COCO)

        print("HierarCaps data loaded")

    @torch.no_grad()
    def run(self):

        print("Running HierarCaps evaluation")

        inp = self.proc(text="", padding=True, truncation=True,
                        return_tensors="pt").to('cuda')
        root = self.clip.get_text_features(**inp)[0]
        root /= root.norm()

        def row2texts(row):
            return [x.strip() for x in row.captions.split('=>')]

        def embed(row):
            texts = row2texts(row)
            assert len(texts) == 4, f'Invalid number of val texts: {len(texts)}'
            inp = self.proc(text=texts, truncation=True,
                            padding=True, return_tensors='pt').to('cuda')
            E = self.clip.get_text_features(**inp)
            E = E / E.norm(dim=-1)[:, None]
            return E

        def embed_img(img):
            inp = self.proc(images=img, return_tensors='pt').to('cuda')
            E = self.clip.get_image_features(**inp)[0]
            E = E / E.norm()
            return E

        def get_dists(E):
            return (E - root[None]).norm(dim=-1)

        R = []  # radii
        Es = []
        VEs = []
        img_fns_uniq = self.df.img_fn.drop_duplicates()

        img_fn2idx = {}
        for i, img_fn in enumerate(tqdm(img_fns_uniq, desc="Embedding image data")):
            img = Image.open(img_fn)
            VE = embed_img(img)
            VEs.append(VE)
            img_fn2idx[img_fn] = i

        t2v_index = {}
        v2t_indices = defaultdict(list)
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Embedding text data"):
            E = embed(row)
            Es.append(E)
            j = img_fn2idx[row.img_fn]
            t2v_index[i] = j
            v2t_indices[j].append(i)

        Es = torch.stack(Es)
        VEs = torch.stack(VEs)
        # Es shape: (n, 4, 512)
        # VEs shape: (m, 512)

        Es_pos = Es[:, :4, :]
        Rs = (Es_pos - root).norm(dim=-1)
        # Rs = 1 - Es @ root
        # Rs (radii) shape: (n, 4)
        rmin, rmax = Rs.min().item(), Rs.max().item()
        levels = np.linspace(rmin, rmax, self.steps)
        masks = [Rs <= thresh for thresh in levels]

        T = np.array([row2texts(row)[:4] for _, row in self.df.iterrows()])
        # T: all (pos) texts
        # shape: (n, 4)
        T_maskeds = [T[mask.cpu()] for mask in masks]

        recalls = []
        precisions = []

        for i, img_fn in enumerate(tqdm(img_fns_uniq, desc="Calculating image metrics")):
            VE = VEs[i]

            # hierarchical retrieval:
            S = Es_pos @ VE
            # S shape: (n, 4)
            preds = []
            vals = []
            for mask, Tm in zip(masks, T_maskeds):
                idx = S[mask].argmax().item()
                t = Tm[idx]
                if len(preds) == 0 or preds[-1] != t:
                    preds.append(t)
                    vals.append(S[mask].max().item())

            t_indices = v2t_indices[i]
            gts = [T[j][:4] for j in t_indices]
            flat_gts = [t for x in gts for t in x]
            # list of lists of 4 positive texts

            s_preds = set(preds)
            s_gts = set(flat_gts)

            tp = len(s_preds & s_gts)
            fp = len(s_preds - s_gts)

            precision = tp / max(1, tp + fp)

            # multi-reference recall: check any for each level
            r1 = any(x[0] in s_preds for x in gts)
            r2 = any(x[1] in s_preds for x in gts)
            r3 = any(x[2] in s_preds for x in gts)
            r4 = any(x[3] in s_preds for x in gts)
            recall = (r1 + r2 + r3 + r4) / 4.

            precisions.append(precision)
            recalls.append(recall)

        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Calculating text metrics"):
            E = Es[i]
            radii = get_dists(E[:4]).cpu()
            R.append(radii)

        R = torch.stack(R)
        # ^ shape: (n, 4)

        ranks = R.argsort(dim=-1)
        corrs = [
            kendalltau(row.cpu().numpy(), [1, 2, 3, 4]).correlation
            for row in ranks
        ]
        dcorr = np.mean(corrs)
        precision = np.mean(precisions)
        recall = np.mean(recalls)

        metrics = {
            'd_corr': dcorr,
            'precision': precision,
            'recall': recall
        }
        self._report_metrics(metrics)

        print("HierarCaps evaluation done")
