"""
python3
バンディット問題を強化学習するプログラム
スロットマシンは２つ存在していて、それぞれの確率は[0.4, 0.6]
当たると報酬が１もらえる
方策はgreedyとε-greedyとε減衰の３つ(このコードにはgreedyしか実装されていない)
残りの二つを実装するのが課題　(このコードから編集するなら)
"""
"""
    Q値の更新を学習率を1/n(試行回数)にするといい。しかし定常環境のみ使用可能
    期待値を使う
    シミュレーションしてみるQ値がどう変動するのかを見る
    最大確率の腕を選んでいる回数/試行回数を見てみるといい
"""
import numpy as np
import matplotlib.pyplot as plt
import random

# バンディットタスク
class Bandit(object):
    def __init__(self):
        # バンディットの設定
        self.probability = np.asarray([[0.4, 0.6]])
        # スタート地点
        self.start = 0
        # ゴール地点
        self.goal = len(self.probability)

    # 報酬を評価
    def get_reward(self, current_state, action):
        # 受け取るアクションは0か1の2値
        # アタリなら１を返す
        if random.random() <= self.probability[current_state, action]:
            return 1
        # 外れなら0を返す
        else:
            return 0

    # 状態の数を返す
    def get_num_state(self):
        return len(self.probability)

    # 行動の数を返す
    def get_num_action(self):
        return len(self.probability[0])

    # スタート地点の場所を返す(初期化用)
    def get_start(self):
        return self.start

    # ゴール地点の場所を返す
    def get_goal_state(self):
        return self.goal

    # 行動を受け取り、次状態を返す
    def get_next_state(self, current_state, current_action):
        return current_state + 1

class KArmBandit(Bandit):
    def __init__(self, prob=[0.4,0.6]):
        self.probability = np.array([prob])
        self.start = 0
        # ゴール地点
        self.goal = len(self.probability)

# Q学習のクラス
class Q_learning():
    # 学習率、割引率、状態数、行動数を定義する
    def __init__(self, learning_rate=0.1, discount_rate=0.9, num_state=None, num_action=None):
        self.learning_rate = learning_rate  # 学習率
        self.discount_rate = discount_rate  # 割引率
        self.num_state = num_state  # 状態数
        self.num_action = num_action  # 行動数
        # Qテーブルを初期化
        self.Q = np.zeros((self.num_state+1, self.num_action))

    # Q値の更新
    # 現状態、選択した行動、得た報酬、次状態を受け取って更新する
    def update_Q(self, current_state, current_action, reward, next_state):
        # TD誤差の計算
        TD_error = (reward
                    + self.discount_rate
                    * max(self.Q[next_state])
                    - self.Q[current_state, current_action])
        # Q値の更新
        self.Q[current_state, current_action] += self.learning_rate * TD_error

    # Q値の初期化
    def init_params(self):
        self.Q = np.zeros((self.num_state+1, self.num_action))

    # Q値を返す
    def get_Q(self):
        return self.Q

    def update_alpha(self,n):
        self.learning_rate = 1 / n

# 方策クラス
class Greedy():  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def serect_action(self, value, current_state):
        a =np.argmax(value[current_state])

        return a
    def reset(self):
        return

# 方策クラス
class E_Greedy():  # greedy方策
    def __init__(self,eps=0.1):
        self.eps = eps
    # 行動価値を受け取って行動番号を返す
    def serect_action(self, value, current_state):
        rnd = random.uniform(0.0,1.0)
        if rnd > self.eps:
            return np.argmax(value[current_state])
        else:
            return np.random.choice(range(len(value[0])))
    def reset(self):
        return

class E_Greedy_Decay():  # greedy方策
    def __init__(self,eps=1.0):
        self.eps = eps
    # 行動価値を受け取って行動番号を返す
    def serect_action(self, value, current_state):
        rnd = random.uniform(0.0,1.0)
        if rnd > self.eps:
            self.eps -= 0.05
            return np.argmax(value[current_state])
        else:
            self.eps -= 0.05
            return np.random.choice(range(len(value[0])))
    def reset(self):
        self.e = 1.0

class PS_policy(object):
    def __init__(self,R = 0.7):
        self.R = R

    def serect_action(self, value, current_state):
        maxidx = np.argmax(value[current_state])
        if  self.R > value[current_state,maxidx]:
            rnd = random.random()
            if rnd < 0.5:
                return maxidx
            else:
                return np.random.choice(range(len(value[0])))
        else:
            return maxidx

class RS_policy(object):
    def __init__(self,num_state,num_action,R = 0.65):
        self.R = R
        self.tau = np.zeros((num_state+1,num_action))
        self.num_state = num_state
        self.num_action = num_action
        self.RSs = []
    def serect_action(self, value, current_state):
        self.RSs = [self.getRSvalue(value,current_state,next_action) for next_action in range(self.num_action)]
        action  = random.choice(np.where(self.RSs == np.max(self.RSs))[0])
        self.update_tau(current_state,action)
        return action

    def getRSvalue(self,value,current_state,next_action):
        return self.tau[current_state,next_action] *(value[current_state,next_action] - self.R)

    def update_tau(self,cuurent_state,current_action):
        self.tau[cuurent_state,current_action] += 1

    def reset(self):
        self.tau = np.zeros((self.num_state+1,self.num_action))

# エージェントクラス
class Agent():
    def __init__(self, value_func="Q_learning", policy="greedy", learning_rate=0.1, discount_rate=0.9, n_state=None, n_action=None):
        # 価値更新方法の選択
        if value_func == "Q_learning":
            self.value_func = Q_learning(num_state=n_state, num_action=n_action)

        # 方策の選択
        if policy == "greedy":
            self.policy = Greedy()
        elif policy == "e-greedy":
            self.policy = E_Greedy()
        elif policy == "e-greedy-decay":
            self.policy = E_Greedy_Decay()
        elif policy == "PS":
            self.policy = PS_policy()
        elif policy == "RS":
            self.policy = RS_policy(n_state,n_action)

    # パラメータ更新(基本呼び出し)
    def update(self, current_state, current_action, reward, next_state):
        self.value_func.update_Q(current_state, current_action, reward, next_state)

    # 行動選択(基本呼び出し)
    def serect_action(self, current_state):
        return self.policy.serect_action(self.value_func.get_Q(), current_state)

    # 行動価値の表示
    def print_value(self):
        print(self.value_func.get_Q())

    # 所持パラメータの初期化
    def init_params(self):
        self.value_func.init_params()
    def reset(self):
        if type(self.policy) == E_Greedy_Decay:
            self.policy = E_Greedy_Decay()
        elif type(self.policy) == RS_policy:
            self.policy.reset()

# メイン関数
def main():
    # ハイパーパラメータ等の設定
    #task = Bandit()  # タスク定義
    task = KArmBandit([0.2,0.4,0.5,0.8])
    SIMULATION_TIMES = 2000 # シミュレーション回数
    EPISODE_TIMES = 1000 # エピソード回数
    #atype = "e-greedy-decay"

    atype = "RS"
    agent = Agent(policy=atype,n_state=task.get_num_state(), n_action=task.get_num_action())  # エージェントの設定
    act_reward = [0.0, 0.0,0.0,0.0] #各腕の報酬の累計
    act_count = [0,0,0,0]       #各腕の選んだ回数
    sumreward_graph = np.zeros(EPISODE_TIMES)  # グラフ記述用の報酬記録
    sum_q = np.zeros((task.get_num_state()+1, task.get_num_action()))

    if type(agent.policy) == RS_policy:
        fp = open("RS value.csv","w")
    # トレーニング開始
    print("トレーニング開始")
    for simu in range(SIMULATION_TIMES):
        agent.init_params()  # エージェントのパラメータを初期化
        agent.reset()
        for epi in range(EPISODE_TIMES):
            current_state = task.get_start()  # 現在地をスタート地点に初期化
            while True:
                action = agent.serect_action(current_state)
                # 報酬を観測
                reward = task.get_reward(current_state, action)
                if type(agent.policy) == RS_policy:
                    row = [x for x in agent.policy.RSs]
                    row.extend([action,reward])
                    times = [simu,epi]
                    times.extend(row)
                    fp.write(",".join(map(str,times)))
                    fp.write("\n")
                act_reward[action] += reward
                act_count[action] += 1
                a = 0
                sumreward_graph[epi] += reward
                # 次状態を観測
                next_state = task.get_next_state(current_state, action)
                # Q価の更新
                agent.value_func.update_alpha(epi+1)
                agent.update(current_state, action, reward, next_state)
                # 次状態が終端状態であれば終了
                if next_state == task.get_goal_state():
                    break
        sum_q += agent.value_func.get_Q()
        #print("\n")
    if type(agent.policy) == RS_policy:
        fp.close()

    print("最終シミュレーションのQ値の表示")
    agent.print_value()
    print("全シミュレーションのQ値の平均の表示")
    avg_q = sum_q / SIMULATION_TIMES
    print(avg_q)

    print("グラフ表示")
    plt.plot(sumreward_graph/SIMULATION_TIMES, label=atype)  # グラフ書き込み

    plt.legend()  # 凡例を付ける
    plt.title("reward")  # グラフタイトルを付ける
    plt.xlabel("episode")  # x軸のラベルを付ける
    plt.ylabel("sum reward")  # y軸のラベルを付ける
    plt.show()  # グラフを表示

main()
