#!/usr/bin/env python3
"""Read a folder of images and write one row per supported image to a CSV."""
import argparse
import csv
import os
from pathlib import Path
import sys
import tempfile
import time

FIELDS = ['image_id','attack_score','predicted_attack','threshold','score_whole','score_native',
          'z_whole','z_native','dominant_branch','width','height','status','error']


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Folder of images; labels/counterparts are not needed.')
    p.add_argument('--output',type=Path,required=True,help='Output CSV path.')
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'artifacts/fusion_model.pt')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--threads',type=int,default=4,help='CPU threads; default 4.')
    p.add_argument('--recursive',action='store_true',help='Include images in subfolders.')
    p.add_argument('--overwrite',action='store_true',help='Allow replacement of an existing output CSV.')
    return p


def run(args):
    from pad.model import PADPredictor, EXTENSIONS
    root=args.input.resolve()
    if not root.is_dir(): raise ValueError(f'Input directory does not exist: {root}')
    if args.output.exists() and not args.overwrite:
        raise ValueError(f'Output exists: {args.output}; use --overwrite to replace it.')
    paths=sorted((p for p in (root.rglob('*') if args.recursive else root.iterdir())
                  if p.is_file() and p.suffix.lower() in EXTENSIONS),key=lambda p:p.relative_to(root).as_posix())
    if not paths: raise ValueError('No supported images found in input folder.')
    predictor=PADPredictor(args.model,args.device,args.threads)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    errors=0;start=time.perf_counter()
    fd,tmp=tempfile.mkstemp(prefix='.pad_scores_',suffix='.csv',dir=args.output.parent)
    try:
        with os.fdopen(fd,'w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for i,path in enumerate(paths):
                row={'image_id':path.relative_to(root).as_posix()}
                try: row.update(predictor.predict(path),status='ok',error='')
                except (OSError,ValueError,RuntimeError) as exc:
                    errors+=1;row.update(status='error',error=str(exc))
                writer.writerow(row);f.flush()
                print(f'{i+1}/{len(paths)}: {row["image_id"]} [{row["status"]}]',file=sys.stderr,flush=True)
        os.replace(tmp,args.output)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(f'Wrote {len(paths)} rows to {args.output}; errors={errors}; elapsed={time.perf_counter()-start:.1f}s',file=sys.stderr)
    return 2 if errors else 0


if __name__=='__main__':
    try: sys.exit(run(parser().parse_args()))
    except (OSError,ValueError,RuntimeError,ImportError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
