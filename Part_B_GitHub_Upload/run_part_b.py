
import argparse
import csv
import fcntl
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torchvision


ROOT = Path(__file__).resolve().parent
CONFIG = 'config/SN-GAN_CIFAR10_CNN_200k.txt'
STATS_IDS = {'train': '143zk96jHE0ZIUSDwmca-B-Q_qjU-uQG0',
             'test': '1_K66b9VJC1WdZRtKDVIbRBulKsIUfw53'}


def check_stats(path):
    with np.load(path, allow_pickle=False) as data:
        for key, shape in [('mu', (2048,)), ('sigma', (2048, 2048))]:
            arr = data[key]
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f'Invalid {key} in {path}')


def prepare_stats(existing):
    folder = ROOT/'stats'
    folder.mkdir(exist_ok=True)
    for split, file_id in STATS_IDS.items():
        target = folder/f'cifar10.{split}.npz'
        if existing:
            src = existing.resolve()/target.name
            check_stats(src)
            if target.exists() and not target.samefile(src):
                import hashlib
                if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
                    raise ValueError(f'{target} differs from {src}; use a fresh extraction')
            elif not target.exists():
                target.symlink_to(src)
        if not target.exists():
            temp = target.with_suffix('.download')
            urlretrieve(f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t', temp)
            check_stats(temp)
            os.replace(temp, target)
        check_stats(target)


def evaluate(out, checkpoint, weights, count):
    path = out/f'eval_{checkpoint.stem}_{weights}_{count}.txt'
    cmd = [sys.executable, '-u', 'eval_reproduction.py', '--flagfile='+CONFIG,
           '--checkpoint_path='+str(checkpoint), '--weights='+weights, '--num_images='+str(count)]
    if not path.exists():
        tmp = path.with_suffix('.tmp')
        with tmp.open('w') as stream:
            subprocess.run(cmd, stdout=stream, check=True)
        os.replace(tmp, path)
    results = {}
    for line in path.read_text().splitlines():
        if line.startswith('RESULT '):
            row = dict(v.split('=',1) for v in line.split()[1:])
            results[row['split']] = row
    if set(results) != {'train','test'}:
        raise ValueError(f'Incomplete evaluation in {path}')
    return results


def run(out):
    import pytorch_gan_metrics  # Check metric dependencies before starting the long run.
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for this server experiment.')
    env = dict(gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
               torch=str(torch.__version__), torchvision=str(torchvision.__version__),
               cuda=torch.version.cuda, platform=platform.platform())
    print(env,flush=True)
    with (out/'environments.jsonl').open('a') as f:
        f.write(json.dumps(env)+'\n')
    final = out/'200000.pt'
    latest = out/'model.pt'
    if not final.exists() or not latest.exists():
        cmd = [sys.executable,'-u','train.py','--flagfile='+CONFIG,'--num_workers=4','--logdir='+str(out)]
        if latest.exists():
            cmd.append('--resume')
        subprocess.run(cmd,check=True)
    rows = []
    baseline = list(csv.DictReader((ROOT/'experiments/gn_baseline/results.csv').open()))
    for label, ck, weights in [('Best-G',out/'best_model.pt','net_G'),
                               ('Final-G',final,'net_G'),('Final-EMA',final,'ema_G')]:
        big = evaluate(out,ck,weights,50000)['train']
        small = evaluate(out,ck,weights,10000)['test']
        rows.append(dict(model='SN-GAN-'+label,step=big['step'],IS_50k=big['IS'],
                         IS_std_50k=big['IS_std'],FID_train_50k=big['FID'],FID_test_10k=small['FID']))
    for row in rows:
        if not all(math.isfinite(float(v)) for k,v in row.items() if k not in ('model','step')):
            raise ValueError('Nonfinite score')
    # Load only checkpoints produced by this experiment.
    state=torch.load(final,map_location='cpu')
    (out/'timing.json').write_text(json.dumps(dict(loop_seconds=state['loop_seconds'],
        scope='Training loop including sampling and intermediate metric passes; excludes setup, final checkpoint writes and standalone final evaluations.'),indent=2))
    for name, values in [('results.csv',rows),('comparison.csv',baseline+rows)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(values)
    print((out/'comparison.csv').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT/'logs/SN-GAN_CIFAR10_CNN_200k')
    p.add_argument('--stats-dir',type=Path,help='Existing CIFAR-10 NPZ pair from the GN server')
    args=p.parse_args()
    out=args.out.resolve()
    stats=args.stats_dir.resolve() if args.stats_dir else None
    os.chdir(ROOT)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_stats(stats)
        run(out)
