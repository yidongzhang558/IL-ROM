"""Train the Case 1 fixed-input baselines or original set encoder from scratch.

Example: python scripts/train_case1_fixed.py --method deeponet --seed 2026
The fixed-input methods evaluate C1 only; set_encoder evaluates C1--C5.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from ilrom_basic import case1
from ilrom_basic.encoders import MultiModalSensorSetEncoder
from baselines import case1_training as train


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def all_metrics(pred, truth):
    values=case1.regression_metrics(pred,truth)
    delta=np.asarray(pred,dtype=np.float64)-np.asarray(truth,dtype=np.float64)
    values['relative_rms']=float(np.sqrt(np.sum(delta**2)/max(np.sum(np.asarray(truth,dtype=np.float64)**2),1e-16)))
    return values


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['deeponet','pod_gp','fno','set_encoder'],required=True)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--out',type=Path,help='New output directory (must be empty).')
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs',type=int,help='Optional verification-only budget override; recorded as non-paper budget.')
    return p.parse_args()


def main():
    args=parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    if args.epochs is not None and (args.epochs<1 or args.method=='pod_gp'):
        raise ValueError('--epochs must be positive and applies only to neural methods.')
    output=args.out or ROOT/'outputs'/'case1'/args.method/f'seed_{args.seed}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a nonempty run: {output}')
    output.mkdir(parents=True,exist_ok=True)
    started=time.time()
    config=json.loads((ROOT/'configs/case1_fixed_baselines.json').read_text(encoding='utf8'))
    settings=dict(config[args.method])
    if args.epochs is not None: settings['epochs']=args.epochs
    actual={'case':'case1','method':args.method,'seed':args.seed,'pod_rank':config['pod_rank'],
            'settings':settings,'split':config['split'],'device':args.device,
            'paper_budget':args.epochs is None or args.epochs==config[args.method].get('epochs'),
            'initialized_from_scratch':True,'pretrained_inputs':False}
    write_json(output/'config.json',actual)
    device=torch.device(args.device)
    train.DEVICE=device
    case1.DEVICE=device
    case1.seed_everything(args.seed)
    data=case1.CaseData('case2',data_root=args.data_root)
    truth_all=np.asarray(data.flux,dtype=np.float32)
    rep=case1.PODCurveRepresentation(config['pod_rank']).fit(truth_all[case1.TRAIN_IDX])
    layouts=case1.fixed_layouts('case2')
    c1=layouts['C1_standard']
    train_x=data.fixed_vector(case1.TRAIN_IDX,c1)
    val_x=data.fixed_vector(case1.VAL_IDX,c1)
    test_x=data.fixed_vector(case1.TEST_IDX,c1)
    truth=truth_all[case1.TEST_IDX]
    val_truth=truth_all[case1.VAL_IDX]
    normalized=train.target_normalized(rep,truth_all[case1.TRAIN_IDX])
    predictions={}
    print(f'Start {args.method} seed={args.seed}; train/val/test={len(case1.TRAIN_IDX)}/{len(case1.VAL_IDX)}/{len(case1.TEST_IDX)}',flush=True)
    if args.method=='deeponet':
        model,history,best=train.train_deeponet(train_x,val_x,normalized,val_truth,rep,args.seed,settings['epochs'])
        predictions['C1_standard']=train.predict_curve(model,test_x,rep)
        model_args={'n_obs':train_x.shape[1],'n_output':truth.shape[1]}
        completed_epochs=int(history[-1][0])
    elif args.method=='fno':
        model,history=train.train_fno(data,rep,c1,settings['epochs'],args.seed)
        best=case1.regression_metrics(train.predict_fno(model,data,rep,case1.VAL_IDX,c1),val_truth)['rel_l2']
        predictions['C1_standard']=train.predict_fno(model,data,rep,case1.TEST_IDX,c1)
        model_args={'sensor_num':train_x.shape[1],'fc_size':64,'out_size':data.grid,'modes':16,'width':48}
        completed_epochs=len(history['train_loss'])
    elif args.method=='pod_gp':
        model,best=train.select_pod_gp(train_x,truth_all[case1.TRAIN_IDX],val_x,val_truth,config['pod_rank'])
        predictions['C1_standard']=model.predict(test_x).mean
        model_args={'rank':model.rank,'length_scale':model.length_scale,'noise':model.noise}
        history={'selected_validation_rel_l2':float(best),'selection_grid':settings}
        completed_epochs=0
    else:
        # Preserve the historical initialization and layout-order seed offsets.
        case1.seed_everything(args.seed+1)
        model_args={'n_modalities':2,'latent_dim':config['pod_rank'],'d_model':64,'nhead':4,'nlayers':2,'condition_dim':0}
        initial=MultiModalSensorSetEncoder(**model_args).state_dict()
        model,history,best=train.train_original_set(data,rep,rep.encode(truth_all),initial,settings['epochs'],args.seed+11)
        for name,layout in layouts.items():
            _,predictions[name]=train.predict_student(model,data,rep,case1.TEST_IDX,layout)
        completed_epochs=len(history['train_loss'])
    for name,pred in predictions.items():
        if pred.shape!=truth.shape or not np.isfinite(pred).all():
            raise ValueError(f'Invalid prediction for {name}: {pred.shape}')
    state=model.state_dict()
    if args.method!='pod_gp':
        state={k:v.detach().cpu() for k,v in state.items()}
    stats={k:float(getattr(data,k)) for k in ['t_mean','t_std','u_mean','u_std','condition_mean','condition_std']}
    deployment={'format':'ilrom_basic_case1_baseline_v1','case':'case1','method':args.method,'seed':args.seed,
                'model':state,'model_args':model_args,'representation':rep.state_dict(),
                'normalization':stats,'layouts':{k:asdict(layouts[k]) for k in predictions},
                'training':actual,'selected_validation_rel_l2':float(best),'initialized_from_scratch':True}
    torch.save(state,output/'model.pt')
    torch.save(deployment,output/'deployment.pt')
    np.savez_compressed(output/'predictions.npz',truth=truth,test_indices=case1.TEST_IDX,
                        **{'pred_'+name:pred for name,pred in predictions.items()})
    write_json(output/'history.json',history)
    source_files=[Path(__file__),ROOT/'src/baselines/case1_training.py',ROOT/'src/baselines/deeponet.py',
                  ROOT/'src/baselines/fno.py',ROOT/'src/baselines/pod_gp.py',ROOT/'src/ilrom_basic/case1.py',
                  ROOT/'src/ilrom_basic/encoders.py',ROOT/'configs/case1_fixed_baselines.json']
    payload={'case':'case1','method':args.method,'seed':args.seed,'configuration':actual,
             'training':{'epochs_run':completed_epochs,'best_validation_rel_l2':float(best),'selection_layout':'C1_standard'},
             'test':{name:all_metrics(pred,truth) for name,pred in predictions.items()},
             'parameters':sum(p.numel() for p in model.parameters()) if args.method!='pod_gp' else None,
             'data_shapes':{'temperature':list(data.temperature.shape),'target':list(data.flux.shape)},
             'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
             'elapsed_seconds':time.time()-started,'completed_utc':datetime.now(timezone.utc).isoformat(),
             'status':'complete','pretrained_inputs':False}
    write_json(output/'metrics.json',payload)
    print(json.dumps({'status':'complete','method':args.method,'seed':args.seed,
                      'test':payload['test'],'output':str(output)},indent=2),flush=True)


if __name__=='__main__':
    main()
