import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,gymnasium as gym
from taxi_rl.core import NEXT,REWARD,TERMINAL,td_target,planner_actions,evaluate,reset_states,epsilon_at
class CoreTests(unittest.TestCase):
 def test_cached_transition_equivalence(self):
  env=gym.make('Taxi-v3')
  for s in range(500):
   for a in range(6):
    env.reset(seed=0);env.unwrapped.s=s
    ns,r,end,trunc,_=env.step(a)
    self.assertEqual((ns,r,end),(NEXT[s,a],REWARD[s,a],TERMINAL[s,a]))
  env.close()
 def test_targets_and_truncation(self):
  q=np.zeros((500,6));q[1]=[0,2,4,0,0,0]
  self.assertAlmostEqual(td_target('q_learning',q,1,-1,False,.9),2.6)
  self.assertAlmostEqual(td_target('sarsa',q,1,-1,False,.9,1),.8)
  self.assertEqual(td_target('q_learning',q,1,20,True,.9),20)
 def test_planner_and_reward(self):
  states=reset_states(range(30000,30300))
  for r in evaluate(None,states,kind='planner'):
   self.assertEqual(r['success'],1);self.assertEqual(r['illegal'],0)
   self.assertEqual(r['return_'],21-r['steps'])
 def test_epsilon(self):
  self.assertEqual(epsilon_at(0,.05),1)
  self.assertAlmostEqual(epsilon_at(8000,.05),.05)
if __name__=='__main__':unittest.main()
