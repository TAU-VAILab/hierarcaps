import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
from transformers import AutoProcessor, CLIPModel, CLIPTextModelWithProjection
import torch
from transformers import logging as transformers_logging
from argparse import ArgumentParser
import os
import json
from collections import namedtuple


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_file',
                        default='../data/hierarcaps_train.csv.gz', type=str)
    parser.add_argument('-e', '--epochs', default=1, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-l', '--learning_rate', default=1e-7, type=float)
    parser.add_argument('-o', '--output_dir', default='output', type=str)
    parser.add_argument('-wf', '--output_weights_fn',
                        default='clip_ft', type=str)
    parser.add_argument('-lf', '--output_loss_fn',
                        default='loss_logs.json', type=str)
    parser.add_argument('-lp', '--lambda_p', default=10.0, type=float)
    parser.add_argument('-le', '--lambda_e', default=1.0, type=float)
    parser.add_argument('-lm', '--lambda_m', default=1.0, type=float)
    parser.add_argument('-v', '--val_p', default=0.0, type=float)
    parser.add_argument('-vf', '--val_freq', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-bc', '--base_checkpoint',
                        default='openai/clip-vit-base-patch32', type=str)
    parser.add_argument('-ml', '--max_length', default=77, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    return parser.parse_args()


class DS(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        poss = row.positive
        negs = row.negative
        return poss.split(' => '), negs.split(' => ')


def load_model_and_tokenizer(args):
    model = CLIPModel.from_pretrained(
        args.base_checkpoint).to(args.device).train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.text_model.parameters():
        param.requires_grad = True
    for param in model.text_projection.parameters():
        param.requires_grad = True

    proc = AutoProcessor.from_pretrained(args.base_checkpoint)

    return model, proc


def load_data(args):
    print('Reading data from file:', args.data_file)
    df = pd.read_csv(args.data_file)
    print(f'Data read; {len(df)} items')

    if args.debug:
        print('Debug mode: using subset of data')
        df = df.head(10000)
        print(f'Subset used: {len(df)} items')

    do_val = args.val_p > 0

    if do_val:
        df['split'] = 'train'
        if args.val_p > 0:
            n_val = int(len(df) * args.val_p)
            df.split.iloc[-n_val:] = 'val'
        print('Train/val split:')
        print(df.split.value_counts())
    else:
        print('Not using val split')

    ds = DS(df[df.split == 'train'] if do_val else df)
    dl = DataLoader(
        ds,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=0
    )

    if do_val:
        ds_val = DS(df[df.split == 'val'])
        dl_val = DataLoader(
            ds_val,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=0
        )
    else:
        dl_val = None

    return dl, dl_val, do_val


def print_hparams(args):
    print('Hyperparameters:')
    print('\tEpochs:', args.epochs)
    print('\tLR:', args.learning_rate)
    print('\tBatch size:', args.batch_size)
    print('\tlambda_p:', args.lambda_p)
    print('\tlambda_e:', args.lambda_e)
    print('\tlambda_m:', args.lambda_m)


class PullbackLossScorer:

    def __init__(self, base_checkpoint, device):
        self.model = CLIPTextModelWithProjection.from_pretrained(
            base_checkpoint)
        self.model.eval()
        self.model.to(device)

    def calculate_loss(self, inp, E):
        with torch.no_grad():
            orig = self.model(**inp).text_embeds
            orig = orig / orig.norm(dim=-1)[:, None]

        sim = torch.einsum('bi,bi->b', E, orig).mean()
        return (1 - sim) / 2.


BatchLosses = namedtuple("BatchLosses", "ploss eloss total_loss")


def batch2losses(args, model, proc, pullback, B):

    inp = proc(text="",
               padding=True, truncation=True, return_tensors="pt",
               max_length=args.max_length
               ).to(args.device)
    root = model.get_text_features(**inp)[0]
    root = root / root.norm()

    # B: [poss, negs]
    # poss: [(sample1_item1, sample2_item1, ..., samplebsz_item1), (sample1_item2, ...), ...]
    # (negs: same as poss)
    poss = B[0]
    negs = B[1]
    assert len(poss) == len(negs) == 4
    bsz = len(poss[0])
    all_texts = [y for x in poss + negs for y in x]
    # flattened: [sample1_item1, sample2_item1, ..., samplebsz_item1, sample1_item2, ...]

    inp = proc(text=all_texts,
               return_tensors="pt", padding=True, truncation=True,
               max_length=args.max_length
               ).to(args.device)
    E = model.get_text_features(**inp)
    E = E / E.norm(dim=-1)[:, None]

    ploss = pullback.calculate_loss(inp, E)

    Ev = E.view(2, 4, bsz, -1)
    # shape: (chain vs. hal, item, b, 512)
    Evv = Ev.view(8, bsz, -1)
    # shape: (8, b, 512)

    def ij2ext(i, j):
        a = Evv[i] - root
        b = Evv[j] - root
        b_ = b - a

        an = a.norm(dim=-1)
        bn = b_.norm(dim=-1)
        ext_c = (a * b_).sum(dim=-1) / (an * bn)
        # ^ cos-ext angle
        ext_a = ext_c.clip(min=-1., max=1.).acos()
        # ext angle
        return ext_a

    pos_indices = [(i, i+1) for i in range(3)]
    neg_indices = [(i, i+4) for i in range(3)]

    P = torch.stack([ij2ext(*indices) for indices in pos_indices])
    N = torch.stack([-ij2ext(*indices) for indices in neg_indices])
    # size: (3, b)

    eloss = P.mean() + N.mean()
    Pr = P.ravel()
    Nr = N.ravel()
    PNr = Pr + Nr
    i, j = Pr.argmax(), Nr.argmax()
    mloss = PNr[i] + PNr[j]

    total_loss = (
        args.lambda_p * ploss
        + args.lambda_e * eloss
        + args.lambda_m * mloss
    )
    L = BatchLosses(ploss, eloss, total_loss)

    return L


def train(args, model, proc, dl, dl_val, do_val):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    pullback = PullbackLossScorer(
        base_checkpoint=args.base_checkpoint, device=args.device)

    print_hparams(args)

    print('Training...')

    loss_logs = {'train': []}
    if do_val:
        loss_logs['val'] = []

    total_iter_counter = 0
    for i in trange(args.epochs, desc="Epoch"):
        pbar = tqdm(dl, desc="Train iteration")
        for j, B in enumerate(pbar):

            if do_val and total_iter_counter % args.val_freq == 0:
                with torch.no_grad():
                    for j, B in enumerate(tqdm(dl_val, desc="Validating")):
                        L = batch2losses(args, model, proc, pullback, B)
                        L_dict = {k: v.item() for k, v in L._asdict().items()}
                        L_dict['epoch'] = i
                        L_dict['iter'] = j
                        L_dict['total_iter'] = total_iter_counter
                        loss_logs['val'].append(L_dict)

            optimizer.zero_grad()

            L = batch2losses(args, model, proc, pullback, B)
            L_dict = {k: v.item() for k, v in L._asdict().items()}
            L_dict['epoch'] = i
            L_dict['iter'] = j
            loss_logs['train'].append(L_dict)

            L.total_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_iter_counter += 1

            pbar.set_description(f"Train iter (L={L.total_loss.item():.4f})")

    return loss_logs


def save(args, model, proc, loss_logs):

    print('Making directory (if missing):', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print('Saving weights (& proc)...')
    out_fn = os.path.join(args.output_dir, args.output_weights_fn)
    if os.path.exists(out_fn):
        print('Warning: weights file exists, overwriting:', out_fn)
    model.save_pretrained(out_fn)
    proc.save_pretrained(out_fn)
    print('Weights (& proc) saved to:', os.path.join(
        args.output_dir, args.output_weights_fn))

    print('Saving loss logs....')
    log_fn = os.path.join(args.output_dir, args.output_loss_fn)
    if os.path.exists(log_fn):
        print('Warning: log file exists, overwriting:', log_fn)
    with open(log_fn, 'w') as f:
        json.dump(loss_logs, f, indent=4)
    print('Loss logs saved to:', log_fn)


def main():

    args = get_opts()

    assert args.data_file is not None, 'Missing data filename'
    assert args.val_p >= 0 and args.val_p <= 1, f'Invalid value for val_p: {args.val_p}'
    if args.val_p > 0:
        assert args.val_freq > 0, 'If validation is used, val_freq must be a positive number of steps'

    print('Loading model and tokenizer...')
    print('Base checkpoint:', args.base_checkpoint)
    model, proc = load_model_and_tokenizer(args)
    print('Model and tokenizer loaded')

    print('Loading data...')
    dl, dl_val, do_val = load_data(args)
    print('Data loaded')

    print("Training...")
    loss_logs = train(args, model, proc, dl, dl_val, do_val)
    print("Done training")

    print('Saving outputs')
    save(args, model, proc, loss_logs)
    print('Outputs saved')


if __name__ == "__main__":
    transformers_logging.set_verbosity_error()
    main()
