"""Reproduce validation, 20 final runs, locked tests, and figures."""
import argparse,json,platform,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np,pandas as pd,gymnasium as gym
from taxi_rl.core import train,reset_states,evaluate,aggregate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    out=Path(__file__).parent/('smoke_outputs' if args.smoke else 'outputs');out.mkdir(exist_ok=True)
    episodes=100 if args.smoke else 10000
    validation=reset_states(range(20000,20100));test=reset_states(range(30000,30300))
    # Shared per-episode start lists. Action RNGs remain algorithm-local.
    devseeds=range(1) if args.smoke else range(3)
    sweep=[]
    for algo in ['q_learning','sarsa']:
        for alpha in [.1,.3]:
            for floor in [.01,.05]:
                scores=[]
                for seed in devseeds:
                    starts=reset_states(np.arange(episodes)+seed*100000)
                    q,*_=train(algo,alpha,floor,seed,starts,validation,episodes,False)
                    scores.append(aggregate(evaluate(q,validation,7000+seed)))
                row=dict(algorithm=algo,alpha=alpha,epsilon_floor=floor,success_rate=float(np.mean([x['success_rate'] for x in scores])),mean_return=float(np.mean([x['mean_return'] for x in scores])))
                sweep.append(row);print('validation',row,flush=True)
    pd.DataFrame(sweep).to_csv(out/'validation_sweep.csv',index=False)
    selected={}
    for algo in ['q_learning','sarsa']:
        rows=[r for r in sweep if r['algorithm']==algo]
        # Predeclared tie break: lower alpha then higher epsilon floor.
        selected[algo]=sorted(rows,key=lambda r:(-r['success_rate'],-r['mean_return'],r['alpha'],-r['epsilon_floor']))[0]
    (out/'selected_configs.json').write_text(json.dumps(selected,indent=2))
    runs=[];checkpoints=[];episodes_log=[];test_records=[]
    seeds=range(100,101) if args.smoke else range(100,110)
    for seed in seeds:
        starts=reset_states(seed*100000+np.arange(episodes))
        for algo in ['q_learning','sarsa']:
            cfg=selected[algo]
            q,log,check,eff,snapshot=train(algo,cfg['alpha'],cfg['epsilon_floor'],seed,starts,validation,episodes)
            np.save(out/f'{algo}_seed{seed}.npy',q)
            if snapshot is not None and seed==100:np.save(out/f'{algo}_seed100_episode500.npy',snapshot)
            records=evaluate(q,test,6000+seed)
            for r in records:test_records.append(dict(algorithm=algo,seed=seed,**r))
            rate=[c['success_rate'] for c in check];reached=None
            for j in range(len(check)-2):
                if all(v>=.95 for v in rate[j:j+3]):reached=check[j]['episode'];break
            runs.append(dict(algorithm=algo,seed=seed,**aggregate(records),**eff,threshold_episode=reached))
            checkpoints.extend(dict(algorithm=algo,seed=seed,**c) for c in check)
            episodes_log.extend(dict(algorithm=algo,seed=seed,**c) for c in log)
            print('final',runs[-1],flush=True)
    for kind in ['random','planner']:
        for seed in seeds:
            records=evaluate(None,test,6000+seed,kind)
            test_records.extend(dict(algorithm=kind,seed=seed,**r) for r in records)
            runs.append(dict(algorithm=kind,seed=seed,**aggregate(records)))
    for name,rows in [('run_metrics',runs),('checkpoints',checkpoints),('training_episodes',episodes_log),('test_episodes',test_records)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    metadata=dict(python=sys.version,gymnasium=gym.__version__,numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),episodes=episodes,training_seeds=list(seeds),development_seeds=list(devseeds),validation_reset_seeds=[20000,20099],test_reset_seeds=[30000,30299],time_limit=200,gamma=.99,action_mask=False,cache='Gymnasium deterministic P table',tie_break='lower alpha then higher floor',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    if not args.smoke:
        from analyze import analyze
        analyze(out)
if __name__=='__main__':main()
