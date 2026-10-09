import json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze(out):
    out=Path(out);df=pd.read_csv(out/'run_metrics.csv');cp=pd.read_csv(out/'checkpoints.csv')
    rng=np.random.default_rng(12345);summary={}
    metrics=['success_rate','mean_return','illegal_per_episode','successful_steps','excess_steps','training_seconds','transitions','threshold_episode']
    for algo,g in df.groupby('algorithm'):
        summary[algo]={}
        for metric in metrics:
            if metric not in g:continue
            x=g[metric].dropna().to_numpy()
            if not len(x):continue
            samples=rng.choice(x,(10000,len(x)),replace=True).mean(axis=1)
            summary[algo][metric]=dict(mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else 0,ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),n=len(x))
    q=df[df.algorithm=='q_learning'].set_index('seed');s=df[df.algorithm=='sarsa'].set_index('seed')
    diff=(q.mean_return-s.mean_return).to_numpy();b=rng.choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
    summary['paired_return_difference']=dict(mean=float(diff.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    colors={'q_learning':'#167d9a','sarsa':'#dc7e35'}
    for metric,label in [('success_rate','Greedy validation success (%)'),('mean_return','Greedy validation return')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for algo,g in cp.groupby('algorithm'):
            for seed,run in g.groupby('seed'):ax.plot(run.episode,run[metric]*(100 if metric=='success_rate' else 1),color=colors[algo],alpha=.12)
            means=g.groupby('episode')[metric].mean()
            ax.plot(means.index,means*(100 if metric=='success_rate' else 1),color=colors[algo],lw=2.8,label=algo.replace('_',' ').title())
        ax.set(xlabel='Training episodes',ylabel=label);ax.legend();fig.tight_layout();fig.savefig(out/f'{metric}_curve.png',dpi=180);plt.close(fig)
    # training curve differs from frozen greedy evaluation due to exploration.
    logs=pd.read_csv(out/'training_episodes.csv');fig,ax=plt.subplots(figsize=(10,5))
    for algo,g in logs.groupby('algorithm'):
        curve=g.groupby('episode')['return_'].mean().rolling(100,min_periods=100).mean()
        ax.plot(curve.index,curve,color=colors[algo],label=algo.replace('_',' ').title())
    ax.set(xlabel='Training episodes',ylabel='Exploratory training return (100-episode mean)');ax.legend();fig.tight_layout();fig.savefig(out/'training_return_curve.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':analyze(Path(__file__).parent/'outputs')
