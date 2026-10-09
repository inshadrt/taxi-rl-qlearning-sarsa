"""Tabular control using transition arrays derived from Gymnasium Taxi-v3.
All six actions remain available. No action masking or reward shaping.
"""
from collections import deque
import numpy as np
import gymnasium as gym

def dynamics():
    env = gym.make('Taxi-v3')
    nxt = np.zeros((500, 6), dtype=np.int64)
    rew = np.zeros((500, 6), dtype=np.float64)
    end = np.zeros((500, 6), dtype=bool)
    for s in range(500):
        for a in range(6):
            prob, ns, r, t = env.unwrapped.P[s][a][0]
            assert prob == 1
            nxt[s,a], rew[s,a], end[s,a] = ns,r,t
    env.close()
    return nxt, rew, end

NEXT, REWARD, TERMINAL = dynamics()

def reset_states(seeds):
    env=gym.make('Taxi-v3')
    states=np.array([env.reset(seed=int(s))[0] for s in seeds],dtype=int)
    env.close()
    return states

def choose(q, s, rng, epsilon=0):
    if epsilon and rng.random() < epsilon:
        return int(rng.integers(6))
    row=q[s]; candidates=np.flatnonzero(row==row.max())
    return int(candidates[0] if len(candidates)==1 else rng.choice(candidates))

def epsilon_at(episode, floor):
    return max(floor, 1-(1-floor)*episode/8000)

def td_target(algorithm,q,next_state,reward,terminated,gamma,next_action=None):
    if terminated: return float(reward)
    future=q[next_state].max() if algorithm=='q_learning' else q[next_state,next_action]
    return float(reward+gamma*future)

def planner_actions(state):
    """BFS over full state transitions; excludes invalid/no-op actions.
    In this deterministic task a minimum-step delivery also maximizes return.
    This model-informed baseline never changes the learners.
    """
    queue=deque([(int(state),[])])
    seen={int(state)}
    while queue:
        s,path=queue.popleft()
        for a in range(6):
            ns=NEXT[s,a]
            if TERMINAL[s,a]: return path+[a]
            if ns != s and REWARD[s,a] != -10 and ns not in seen:
                seen.add(int(ns)); queue.append((int(ns),path+[a]))
    raise RuntimeError('No delivery path')

OPTIMAL_STEPS={int(s):len(planner_actions(s)) for s in reset_states(range(1000))}

def evaluate(q,states,seed=0,kind='learned'):
    rng=np.random.default_rng(seed); records=[]
    for initial in states:
        s=int(initial); total=0.; illegal=0; path=planner_actions(s) if kind=='planner' else None
        for step in range(200):
            if kind=='random': a=int(rng.integers(6))
            elif kind=='planner': a=path[step]
            else: a=choose(q,s,rng)
            r=REWARD[s,a]; total+=r; illegal+=int(r==-10)
            done=TERMINAL[s,a]; s=int(NEXT[s,a])
            if done: break
        n=step+1; success=bool(done)
        optimal=OPTIMAL_STEPS.get(int(initial))
        if optimal is None:optimal=len(planner_actions(int(initial)))
        records.append(dict(initial_state=int(initial),return_=float(total),success=int(success),
                            steps=n,illegal=illegal,excess_steps=n-optimal if success else None))
    return records

def aggregate(records):
    success=[r for r in records if r['success']]
    return dict(success_rate=np.mean([r['success'] for r in records]),
                mean_return=np.mean([r['return_'] for r in records]),
                illegal_per_episode=np.mean([r['illegal'] for r in records]),
                successful_steps=np.mean([r['steps'] for r in success]) if success else float('nan'),
                excess_steps=np.mean([r['excess_steps'] for r in success]) if success else float('nan'))

def train(algorithm,alpha,floor,seed,starts,validation,episodes=10000,checkpoints=True):
    import time
    q=np.zeros((500,6));rng=np.random.default_rng(seed)
    logs=[];check=[];elapsed=0.;transitions=0; snapshot=None
    if checkpoints:check.append(dict(episode=0,**aggregate(evaluate(q,validation,9000))))
    for episode in range(episodes):
        tic=time.perf_counter();s=int(starts[episode]);eps=epsilon_at(episode,floor)
        a=choose(q,s,rng,eps);total=0.;illegal=0
        for step in range(200):
            ns=int(NEXT[s,a]);r=REWARD[s,a];terminated=bool(TERMINAL[s,a])
            # SARSA selects a next action even at external time-limit cutoff,
            # because truncation does not make the underlying state terminal.
            na=choose(q,ns,rng,eps) if algorithm=='sarsa' and not terminated else None
            target=td_target(algorithm,q,ns,r,terminated,.99,na)
            q[s,a]+=alpha*(target-q[s,a])
            total+=r;illegal+=int(r==-10);transitions+=1;s=ns
            if terminated:break
            a=na if algorithm=='sarsa' else choose(q,s,rng,eps)
        elapsed+=time.perf_counter()-tic
        logs.append(dict(episode=episode+1,return_=total,steps=step+1,success=int(terminated),illegal=illegal,epsilon=eps))
        if episode+1==500:snapshot=q.copy()
        if checkpoints and (episode+1)%500==0:
            check.append(dict(episode=episode+1,**aggregate(evaluate(q,validation,9000+episode+1))))
    return q,logs,check,dict(training_seconds=elapsed,transitions=transitions),snapshot
