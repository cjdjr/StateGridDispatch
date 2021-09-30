* export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data1/wangmr/StateGridDispatch/gridsim_master_0910/lib64

* obs.gen_p返回的各个时刻潮流计算后的各机组有功出力，仿真器关于有功出力的逻辑是：每个回合的初始时刻，机组有功出力的初始值是从gen_p.csv中随机读取一行（reset()提供了接口，参赛者可以固定读某一行，方便参赛者复现实验），随后各时刻的机组有功出力是根据智能体给的一系列动作，计算出来的，和gen_p.csv就没有关系了。建议参赛者去直接调用obs.xxxx，step()，reset()等强化学习层面的接口。

* 现在训练阶段，允许agent调用grid.env_feedback (潮流计算函数，用来simulation)。但提交作品线上评测时，agent只能和environment进行直接交互

* 20个episode，每个episode 288个time step

* grid.env_feedback 需要设置workdir为gridsim_master_0910

* 目前的env不是gym.env形式的，故不能继承gym.wrapper进行包装

* 数字表示当前电网断线条数，DPF failed表示潮流不收敛

* os.environ['PARL_BACKEND'] = 'torch' 要放到import parl之前

* adjust_gen_v 学不出来

* 现在的一个问题是，当初始停机的过多，会在timestep为40的时候集中开机，导致agent的决策过于奔放，会在timestep=40的时候直接平衡机越限。

* 加上 steps_to_close_gen 学不出来