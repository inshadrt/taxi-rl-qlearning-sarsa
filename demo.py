"""Record native Taxi frames; annotations identify actual model, step and return."""
import os,sys,json,subprocess,tempfile
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np,gymnasium as gym
from PIL import Image
from taxi_rl.core import choose
ROOT=Path(__file__).parent;OUT=ROOT/'outputs'

def main():
 clips=[];trajectories=[]
 cases=[('Untrained policy',np.zeros((500,6)),20),('Q-learning after 500 episodes',np.load(OUT/'q_learning_seed100_episode500.npy'),20),('Q-learning after 10000 episodes',np.load(OUT/'q_learning_seed100.npy'),200),('SARSA after 10000 episodes',np.load(OUT/'sarsa_seed100.npy'),200)]
 with tempfile.TemporaryDirectory() as tmp:
  for k,(label,q,limit) in enumerate(cases):
   work=Path(tmp)/str(k);work.mkdir()
   env=gym.make('Taxi-v3',render_mode='rgb_array');s,_=env.reset(seed=30001);rng=np.random.default_rng(6100)
   total=0;illegal=0;done=False;frames=[];rows=[]
   frames.append(env.render())
   for i in range(limit):
    a=choose(q,s,rng);ns,r,done,trunc,info=env.step(a);total+=r;illegal+=int(r==-10)
    rows.append(dict(step=i+1,state=int(s),action=int(a),next_state=int(ns),reward=int(r),return_=total,terminated=bool(done)))
    frames.append(env.render());s=ns
    if done or trunc:break
   for n,frame in enumerate(frames):Image.fromarray(frame).save(work/f'{n:04d}.png')
   if k==2:Image.fromarray(frames[len(frames)//2]).save(OUT/'demo_screenshot.png')
   status='Delivered' if done else f'First {len(rows)} steps only, not a full episode'
   desc=f'{label}    Reset seed 30001    {len(rows)} steps    Return {total:.0f}    {status}'
   # Hold initial/final frames. Intermediate native steps play at two per second.
   clip=Path(tmp)/f'clip{k}.mp4'
   subprocess.run(['ffmpeg','-y','-framerate','2','-i',str(work/'%04d.png'),'-vf',f"scale=600:400:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0xF5F7FA,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{label}':x=60:y=45:fontsize=32:fontcolor=0x102536,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Reset seed 30001  -  Model seed 100':x=60:y=95:fontsize=22:fontcolor=0x344B5E,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{len(rows)} steps   Return {total:.0f}   Illegal actions {illegal}':x=60:y=605:fontsize=24:fontcolor=0x102536,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{status}':x=60:y=648:fontsize=22:fontcolor=0x344B5E,tpad=start_duration=2:stop_duration=2",'-r','24','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   clips.append(clip);trajectories.append(dict(label=label,seed=30001,steps=len(rows),return_=total,illegal=illegal,delivered=bool(done),shown_only_first_steps=not done,trajectory=rows));env.close()
  manifest=Path(tmp)/'concat.txt';manifest.write_text('\n'.join("file '"+str(c)+"'" for c in clips))
  subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',str(manifest),'-c','copy',str(OUT/'Taxi_Agent_Demo.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 (OUT/'demo_trajectories.json').write_text(json.dumps(trajectories,indent=2))
 print(json.dumps([{k:v for k,v in r.items() if k!='trajectory'} for r in trajectories],indent=2))
if __name__=='__main__':main()
