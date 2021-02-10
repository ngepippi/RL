import logging
import math
import gym
import time
from gym import spaces
from gym .utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class GridWorld(gym.Env):

    def __init__(self):
        self.world_map = [      #探索用マップ
            [0, 1, 1, 1, 1],
            [1, 1, 2 ,2, 1],
            [1, 1, 1, 2, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 2, 1, 3]
        ]

        self.world_size = len(self.world_map) #マップサイズ(今は正方形であることを仮定)
        self.x_threshold = self.world_size - 1#x方向への限界値
        self.y_threshold = self.world_size - 1#y方向への限界値
        self.sx ,self.sy = 0,0                #スタート地点
        self.x,self.y = self.sx,self.sy       #エージェントの現在地

        self.colors = {                       #床の要素の色
            0:[0,0,1], 2:[0,0,0], 3:[1,0,0]
        }
        self.agent_color = [0,1,0]            #エージェントの色
        self.world_attr = {"START":0,"ROAD":1,"WALL":2,"GOAL":3}#マップの要素
        self.map_state = 0

        self.action_space = spaces.Discrete(4)#行動数の設定(4個の行動をとる)

        #状態の範囲設定(方向情報を増やすなら増やそう)
        high=np.array([#最大値
            self.x_threshold,
            self.y_threshold
        ])
        low = np.array([#最小値
            0,
            0
        ])
        self.observation_space = spaces.Box(low,high)

        self._seed()
        self.viewer = None
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.x = self.sx
        self.y = self.sy
        self.state = self.world_attr["START"]
        self.state = np.array([self.x,self.y])

        return self.state

    def _step(self,action):
        if self.isGoal():
            #reward = 1.0#step数を報酬にするならばいらない呼び出し側で計算(maxstep - endstep)
            done = True
        self.Move(action)
        self.state = np.array([self.x,self.y])
        reward = 0.0
        done = False
        #print(self.x,self.y,self.world_map[self.y][self.x])#debug
        return self.state, reward, done, {}

    def _render(self,mode="human",close = False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        map_width = 300
        map_height = 300
        wx_min = (screen_width - map_width) / 2
        wx_max = wx_min + map_width
        wy_min = (screen_height - map_height) / 2
        wy_max = wy_min + map_height
        vw = map_width / self.world_size
        vh = map_height / self.world_size

        #デカい4角形にスタート:青,ゴール:赤,壁:黒で描画したい
        if self.viewer is None:#初回呼び出し時
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            #マップ全体の枠の表示
            world_Box = rendering.FilledPolygon([(wx_min,wy_min),(wx_max,wy_min),(wx_max,wy_max),(wx_min,wy_max)])
            world_Box.set_color(.5,.5,.5)
            self.viewer.add_geom(world_Box)
            for i,row in enumerate(self.world_map):
                for j,c in enumerate(row):
                    tx = wx_min + vw*j
                    ty = wy_max - vh*i
                    if c == self.world_attr["ROAD"]:
                        continue
                    r,g,b = self.colors[c]
                    attr = rendering.FilledPolygon([(tx,ty),(tx+vw,ty),(tx+vw,ty-vh),(tx,ty-vh)])
                    attr.set_color(r,g,b)
                    self.viewer.add_geom(attr)
            agent = rendering.make_circle(vw/5*2)
            r,g,b = self.agent_color
            agent.set_color(r,g,b)
            self.agent_trans = rendering.Transform()#動くからtransformが必要
            agent.add_attr(self.agent_trans)
            self.viewer.add_geom(agent)
            if self.state is None:return None

        self.agent_trans.set_translation(wx_min+vw*self.x+vw/2,wy_max-vh*self.y-vw/2)
        time.sleep(0.5)#動きが速すぎて見えないのでsleep、でも処理止まっちゃうのでなんかうまくやりたい
        return self.viewer.render(return_rgb_array = mode == "rgb_array")

    def Move(self,action):
        action_move = [[0,-1], [1,0], [0,1], [-1,0]]#行動それぞれでの座標の増分
        next_x , next_y = self.x + action_move[action][0],self.y + action_move[action][1]
        #print(next_x,next_y)#debug
        isLimit = 0 <= next_x <= self.x_threshold and 0 <= next_y <= self.y_threshold
        isWall = self.world_map[next_y][next_x] == self.world_attr["WALL"] if isLimit else True

        if isLimit and not isWall:
            self.x , self.y = next_x,next_y
            self.map_state = self.world_map[next_y][next_x]

    def isGoal(self):
        return self.map_state == self.world_attr["GOAL"]
