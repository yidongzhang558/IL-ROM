"""Train clean-layout Case1 FNO variants or VIDON from initialization."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['masked_fno','masked_fno_small','voronoi_fno','vidon'],required=True)
    ap.add_argument('--seed',type=int,choices=[2026,2027,2028],required=True)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();torch.set_num_threads(1)
    out=a.out or ROOT/'outputs/case1'/a.method/('smoke' if a.smoke else '')/f'seed_{a.seed}'
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise FileExistsError(out)
    if a.method=='vidon':
        from baselines import vidon
        vidon.OUTPUT_OVERRIDE=out
        vidon.training('case1',a.seed,a.smoke);return
    from baselines import masked_fno as m
    m.seed_everything(a.seed)
    case=m.ConvCase(12,a.seed,'voronoi' if a.method=='voronoi_fno' else 'sparse')
    cfg=dict(epochs=300,batch_size=32,lr=.0008,eval_every=10,patience_evals=12,min_epochs=100,width=32,modes=16,layers=4)
    if a.method=='masked_fno_small':cfg.update(width=16,modes=8,layers=3)
    if a.smoke:cfg.update(epochs=1,eval_every=1,min_epochs=1)
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(method=a.method,seed=a.seed,from_scratch=True,smoke=a.smoke,**cfg),indent=2))
    model,training=m.train(case,a.seed,**cfg)
    report=m.evaluate(model,case,'test')
    arrays={'truth':case.truth('test')}
    for name,layout in case.layouts.items():arrays['pred_'+name]=case.decode(m.predict_latent(model,case,'test',layout))
    np.savez_compressed(out/'predictions.npz',**arrays)
    rep=case.rep.__dict__
    torch.save(dict(model=model.state_dict(),representation=rep,normalization={k:getattr(case.data,k) for k in ['t_mean','t_std','u_mean','u_std']},config=cfg,seed=a.seed,input_mode=case.input_mode),out/'deployment.pt')
    m.save_json(out/'metrics.json',dict(method=a.method,seed=a.seed,training=training,results=report))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
