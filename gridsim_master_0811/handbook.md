# 文件说明

**/Agent** 智能体相关代码。

**/Agent/BaseAgent** 智能体基类。用户提交的智能体类必须继承该基类。

**/Agent/DoNothingAgent** 空动作智能体类。每次调用返回一个不做任何操作的空动作。

**/Agent/RandomAgent** 随机动作智能体类。每次调用返回一个属于合法动作空间中的随机动作。

**/Environment** 环境相关代码。

**/Environment/base_env** 定义本次比赛使用的电网运行仿真环境类。

**/Observation** 观测值相关代码。

**/Observation/observation** 定义本次比赛使用的电网运行仿真环境观测类。

**/Reward** 奖励值相关代码。

**/Reward/rewards** 定义本次比赛使用的各项奖励值方法。

**/utilize** 辅助功能相关代码。

**/utilize/exceptions** 定义不同种类非法动作的异常类。

**/utilize/parameters** 参数配置相关代码。

**/utilize/parameters/static.yml** 记录静态参数

**/utilize/parameters/dynamic.yml** 记录动态参数

**/utilize/parameters/main.yml** 记录所有参数包括静态和动态。

**/utilize/parameters/yml_creator.py** 参数配置主程序，调用该程序生成上述三个yml文件

**/utilize/form_action.py** 将多个动作封装成一个字典类变量。

**/utilize/line_cutting** 获得电流过载和随机故障导致断开的线路名称。

**/utilize/read_forecast_value** 获得机组有功出力和负荷的未来预测值。

**/utilize/action_space** 合法动作空间类变量

**/main.py** 实现仿真器环境和智能体交互的主程序。

**/example.so** 电网模型文件。禁止用户操作。

**/model_jm** 电网模型文件。禁止用户操作。

**/lib64.zip** example.so文件依赖的第三方库。禁止用户操作。

**/README.py** 说明文档



# 环境准备

代码运行需要linux主机，python3.6和gcc 4.8以上版本。

## 推荐的环境

操作系统 CentOS 3.10.0

python3.6.9

gcc 4.8.5



# 名词解释

（1）有功出力注入值：等于上一时间步观测量给出的有功出力实际值加上当前动作中的有功出力调整值，也等于计算潮流前的机组有功出力。

（2）有功最大出力：不同于有功出力上限，新能源机组的最大出力值会随着时间步发生变化，通过调用`obs.curstep_renewable_gen_p_max`获取。



# 变量声明

## 动作空间

|   变量命名   |  数据类型   | 数据大小 |        含义        | 说明 |
| :----------: | :---------: | :------: | :----------------: | ---- |
| adjust_gen_p | list[float] |   $54$   | 机组有功出力调整值 |      |
| adjust_gen_v | list[float] |   $54$   |   机组电压调整值   |      |

## 观测空间

|           变量命名           |  数据类型   |                含义                |            说明             |
| :--------------------------: | :---------: | :--------------------------------: | :-------------------------: |
|           timestep           |     int     |            当前时间步数            |                             |
|            vTime             |   string    |              当前时刻              |                             |
|            gen_p             | list[float] |            机组有功出力            |                             |
|            gen_q             | list[float] |            机组无功出力            |                             |
|            gen_v             | list[float] |           发电机电压幅值           |                             |
|       target_dispatch        | list[float] |       计算潮流前机组有功出力       |                             |
|       actual_dispatch        | list[float] |       计算潮流后机组有功出力       |                             |
|            load_p            | list[float] |              负荷有功              |                             |
|            load_q            | list[float] |              负荷无功              |                             |
|            load_v            | list[float] |        负荷所在节点电压幅值        |                             |
|             p_or             | list[float] |           支路起始端有功           |                             |
|             q_or             | list[float] |           支路起始端无功           |                             |
|             v_or             | list[float] |           支路起始端电压           |                             |
|             a_or             | list[float] |           支路起始端电流           |                             |
|             p_ex             | list[float] |            支路末端有功            |                             |
|             q_ex             | list[float] |            支路末端无功            |                             |
|             v_ex             | list[float] |            支路末端电压            |                             |
|             a_ex             | list[float] |            支路末端电流            |                             |
|         line_status          | list[bool]  |              线路状态              |                             |
|          grid_loss           | list[float] |         电网损耗（网损值）         |                             |
|             flag             |  list[int]  |         各线路潮流收敛情况         |   1表示不收敛，0表示收敛    |
|         unnameindex          |    dict     |    机组名称和机组编号的对应关系    |                             |
|         action_space         |             |      下一时间步的合法动作空间      |                             |
|   steps_to_reconnect_line    |  list[int]  |  已断开支路恢复连接的剩余时间步数  |                             |
|  count_soft_overflow_steps   |  list[int]  |     支路已连续软过载的时间步数     |                             |
|             rho              | list[float] |           支路电流负载率           |                             |
|          gen_status          |  list[int]  |           机组开关机状态           |    1表示开机，0表示关机     |
|     steps_to_recover_gen     |  list[int]  |   关机机组允许重启的剩余时间步数   | 如果机组状态为开机，则值为0 |
|      steps_to_close_gen      |  list[int]  |   重启机组允许关机的剩余时间步数   |                             |
|           busname            |    dict     |             各节点名称             |                             |
|           bus_gen            |    dict     |        各节点相连的机组名称        |                             |
|           bus_load           |    dict     |        各节点相连的负荷名称        |                             |
|          bus_branch          |    dict     |        各节点相连的支路名称        |                             |
| curstep_renewable_gen_p_max  | list[float] | 当前时间步新能源机组的最大有功出力 |                             |
| nextstep_renewable_gen_p_max | list[float] | 下一时间步新能源机组的最大有功出力 |                             |
|       nextstep_load_p        | list[float] | 下一时间步的负荷                |                             |

## 静态参数

|            参数命名             |     数据类型     |                 含义                 |              说明              |
| :-----------------------------: | :--------------: | :----------------------------------: | :----------------------------: |
|            机组相关             |                  |                                      |                                |
|             num_gen             |       int        |                机组数                |                                |
|     n_steps_to_recover_gen      |    list[int]     |  关机机组允许重启所需的最小时间步数  |                                |
|      n_steps_to_close_gen       |    list[int]     |  重启机组允许关机所需的最小时间步数  |                                |
|            gen_type             |    list[int]     |               机组类型               | 5表示新能源机组，1表示火电机组 |
|            max_gen_p            |   list[float]    |           机组有功出力上限           |                                |
|            min_gen_p            |   list[float]    |           机组有功出力下限           |                                |
|            max_gen_q            |   list[float]    |           机组无功出力上限           |                                |
|            min_gen_q            |   list[float]    |           机组无功出力上限           |                                |
|            max_gen_v            |   list[float]    |             机组电压上限             |                                |
|            min_gen_v            |   list[float]    |             机组电压下限             |                                |
|            max_bus_v            |   list[float]    |             节点电压上限             |                                |
|            min_bus_v            |   list[float]    |             节点电压下限             |                                |
|            ramp_rate            |      float       |              机组爬坡率              |                                |
|           thermal_ids           |    list[int]     |             火电机组编号             |                                |
|        renewable_gen_id         |    list[int]     |            新能源机组编号            |                                |
|          balanced_ids           |    list[int]     |             平衡机组编号             |                                |
|                                 |                  |                                      |                                |
|            支路相关             |                  |                                      |                                |
|            num_line             |       int        |                支路数                |                                |
|       soft_overflow_bound       |       int        | 支路发生软过载前的支路电流负载率上限 |                                |
|       hard_overflow_bound       |       int        | 支路发生硬过载前的支路电流负载率上限 |                                |
|   max_steps_to_reconnect_line   |       int        |   已断开支路重连所需的最大时间步数   |                                |
|       prob_disconnection        |      float       |         随机断线故障发生概率         |                                |
|     max_steps_soft_overflow     |       int        |   支路连续软过载的最大允许时间步数   |                                |
|       disconnection_name        |      string      |             断线线路名称             |                                |
|                                 |                  |                                      |                                |
|         奖励值计算相关          |                  |                                      |                                |
| r_factor_renewable_consumption  |      float       |         新能源消纳占比项系数         |                                |
|        r_factor_overflow        |      float       |            线路越限项系数            |                                |
|      r_factor_running_cost      |      float       |            运行费用项系数            |                                |
|          gen_name_list          |      string      |     机组费用计算：name_list参数      |                                |
|          startup_cost           |      float       |        机组费用计算：开机费用        |                                |
|          constant_cost          |      float       |         机组费用计算：常数项         |                                |
|        first_order_cost         |      float       |        机组费用计算：一阶系数        |                                |
|        second_order_cost        |      float       |        机组费用计算：二阶系数        |                                |
|                                 |                  |                                      |                                |
|              其他               |                  |                                      |                                |
|           num_sample            |       int        |                断面数                |                                |
|          action_space           |       dict       |       下一时间步的合法动作空间       |                                |
| white_list_random_disconnection |      string      |            随机断线白名单            |                                |
|         max_ramp_adjust         |   list[float]    |           火电机组爬坡速率           |                                |
|           智能体交互            |                  |                                      |                                |
|            last_obs             | Observation Type |       上一个时间步的环境观测量       |                                |
|            is_legal             |       bool       |             动作是否合法             |                                |
|         illegal_reasons         |      string      |             违规停止理由             |                                |



# 运行规则

（1）**机组有功出力上下限约束**：任意机组（除了平衡机）的有功出力注入值不能大于有功出力上限，也不能小于有功出力下限。如果违反，仿真器提示“动作非法”，强制结束该回合。

（2）**新能源机组最大出力约束**：在任意时间步中，新能源机组的有功出力注入值不能大于最大出力值。如果违反，仿真器提示“动作非法”，强制结束该回合。

（3）**机组爬坡约束**：任意火电机组的有功出力调整值必须小于爬坡速率。如果违反，仿真器提示“动作非法”，强制结束该回合。

（4）**机组启停约束**：火电机组停运规则为机组停运前机组有功出力必须调整至出力下限，再调整至0。机组停机后连续40个时间步内不允许重新启动。火电机组启动规则为机组开启前有功出力必须调整至出力下限。机组重新启动后连续40个时间步内不允许停机。

（5）**支路越限约束**：若支路的电流值超过其热稳限值，表示支路电流越限。若支路电流越限但未超热稳限值的135%，表示支路“软过载”。若支路电流超热稳限值的135%，表示支路“硬过载”。任意支路连续4个时间步发生“软过载”，则该支路停运。发生“硬过载”则支路立即停运。支路停运16个时间步之后，重新投运。

（6）**随机故障**：每个时间步中，会有1%联络线支路停运概率，停运16个时间步之后，重新投运。

（7）**机组无功出力上下限约束**：当智能体调整机端电压时，机组的无功出力值超过其上下限则获得负奖励。

（8）**电压上下限约束**：节点电压超过其上下限则获得负奖励。

（9）**平衡机上下限约束**：系统设置一台平衡机，用于分担控制策略不合理导致的系统不平衡功率。潮流计算后，平衡机有功出力大于上限但小于上限的110%，或者，小于下限但大于下限的90%，则获得负奖励。出力大于上限的110%或者小于下限的90%，则回合终止。



# 合法动作空间

## 机组有功出力调整

|            当前时间步的机组有功出力实际值            |        其他条件         |            下一时间步有功出力调整值的合法取值范围            |         下一时间步的机组有功出力注入值         |
| :--------------------------------------------------: | :---------------------: | :----------------------------------------------------------: | :--------------------------------------------: |
|                       火电机组                       |                         |                                                              |                                                |
|                         p=0                          | steps_to_recover_gen=0  |                      {0}U(0,min_gen_p]                       |                {0}U{min_gen_p}                 |
|                         p=0                          | steps_to_recover_gen!=0 |                             {0}                              |                       0                        |
|                     p=min_gen_p                      |  steps_to_close_gen=0   |     [-min_gen_p, 0)U[0,min(max_gen_p-p,max_ramp_adjust)]     |           {0}U[min_gen_p,max_gen_p]            |
|                     p=min_gen_p                      |  steps_to_close_gen!=0  |             [0,min(max_gen_p-p,max_ramp_adjust)]             |             [min_gen_p,max_gen_p]              |
|             p$\in$(min_gen_p, max_gen_p]             |  steps_to_close_gen=0   |  [-min(p,max_ramp_adjust),min(max_gen_p-p,max_ramp_adjust)]  |             [min_gen_p,max_gen_p]              |
|             p$\in$(min_gen_p, max_gen_p]             |  steps_to_close_gen!=0  | [-min(p-min_gen_p,max_ramp_adjust),min(max_gen_p-p,max_ramp_adjust)] |             (min_gen_p,max_gen_p]              |
|                      新能源机组                      |                         |                                                              |                                                |
| p$\in$[0,min(max_gen_p,curstep_renewable_gen_p_max)] |            /            |      [-p,min(max_gen_p,nextstep_renewable_gen_p_max)-p]      | [0,min(max_gen_p,curstep_renewable_gen_p_max)] |
|                       平衡机组                       |                         |                                                              |                                                |
|                          p                           |            /            |                          [-inf,inf]                          |                   [-inf,inf]                   |

## 补充说明

（1）新能源机组的有功出力受到的影响除了固定上限值max_gen_p，还有一个每个时间步会变化的最大出力renewable_gen_p_max

（2）重启或关闭火电机组时，有功出力调整值不受机组爬坡约束。

（3）当某火电机组处于关机状态且允许该机组重启时，智能体给上调出力动作，仿真器会被默认执行开机动作，机组出力变成下限值。

（4）某火电机组出力等于下限值且允许该机组关闭时，智能体给下调出力动作，仿真器会被默认执行关机动作，机组出力变成零。

（5）平衡机有功出力实际值和智能体给出的有功出力调整值并无直接关系，仿真器不对平衡机的有功出力调整动作的合法性进行判定。



# 奖励和得分

**奖励（reward）**作为智能体算法的优化目标，具体形式可由用户自行定义。仿真器提供了几种奖励值的具体形式供参赛者选择：

（1）线路越限情况（正奖励）
$$
r_1=1-\frac{1}{n_{line}}\sum_{i=1}^{n_{line}}{min\left(\frac{I_i}{T_i+\epsilon},1\right)}
$$
其中$n_{line}$表示电网支路个数，$I_i$和$T_i$表示支路$i$的电流和热极限，$\epsilon$为一常数取值为0.1，避免出现分母为零的情况。

（2）新能源机组消纳量（正奖励）
$$
r_2=\frac{\sum_{i=1}^{n_{new}}p_i}{\sum_{i=1}^{n_{new}}\overline{p}_i}
$$
其中$n_{new}$表示新能源机组个数，$p_i$表示新能源机组$i$的实际有功出力，$\overline{p}_i$表示新能源机组$i$在当前时间步的最大出力。

（3）平衡机功率越限（负奖励）
$$
r_4=\frac{10}{n_{balanced}} \sum_{i=1}^{n_{balanced}}1-\frac{p}{p^{max}} \quad if \quad p^{max}<p<1.1\times p^{max}
$$
$$
r_4=\frac{10}{n_{balanced}} \sum_{i=1}^{n_{balanced}}\frac{p}{p^{min}}-1 \quad if \quad 0.9 \times p^{min}<p<p^{min}
$$



其中$n_{balanced}$表示平衡机个数，$p$表示平衡机的实际有功出力，$p^{max}$表示平衡机的出力上限。

（4）机组运行费用（负奖励）
$$
r_4=-\sum_{i=1}^{n}{(a{p_i}^2+bp_i+c)-d(if 火电机组有启停)}
$$

其中$n$表示机组总个数，$q_i$表示机组$i$的实际有功出力，$a,b,c$表示系数。新能源和平衡机没有关机状态，一直保持开机。火电机组的关机状态通过判断机组有功出力是否为零来确定。

（5）无功出力越限（负奖励）
$$
r_5=\frac{1}{n_{gen}}\sum_{i=1}^{n_{gen}} 1-\frac{q_i}{q_{i}^{max}} \quad if \quad q_i>q_i^{max}
$$

$$
r_5=\frac{1}{n_{gen}}\sum_{i=1}^{n_{gen}}1-\frac{q_{i}^{min}}{q_i} \quad if \quad q_i<q_i^{min}
$$

其中$n$表示机组总个数，$q_i$表示机组的实际无功出力，$q_i^{max}$表示机组的无功出力上限，$q_i^{min}$表示机组的无功出力下限。	

（6）节点电压越限（负奖励）
$$
r_6=\frac{1}{n_{sub}}\sum_{i=1}^{n_{sub}}1-\frac{v_i}{v_{i}^{max}} \quad if \quad v_i>v_i^{max}
$$

$$
r_6=\frac{1}{n_{sub}}\sum_{i=1}^{n_{sub}}1-\frac{v_{i}^{min}}{v_i} \quad if \quad v_i<v_i^{min}
$$

其中$n_{sub}$表示电网节点个数，$v_i$表示节点$i$的电压值，$v_i^{max}$表示节点$i$的电压上限，$v_i^{min}$表示节点$i$的电压下限。	

对奖励项$r_4$、$r_5$、$r_6$进行归一化，公式如下
$$
r=e^{r}-1
$$
由上可知，奖励项$r_1$、$r_2$的域值为$[0,1]$，奖励项$r_3$、$r_4$、$r_5$、$r_6$的域值为$[-1,0]$



仿真器默认使用的奖励，公式如下
$$
R=a_1r_1+a_2r_2+a_3r_3+a_4r_4+a_5r_5+a_6r_6
$$
其中$r_i$表示归一化后的各奖励项，$a_i$表示各奖励项系数，根据赛事考验侧重点进行取值，本次赛事取值为
$$
a_1=1,a_2=2,a_3=1,a_4=3,a_5=1,a_6=1
$$



**得分（score）**作为比赛测试阶段评价智能体算法优劣的唯一依据，定义如下：
$$
score=\frac{\sum_{t=1}^{t_{over}}R_t}{|t_{over}|}
$$
其中$R_t$表示智能体在$t$时刻获得的奖励值大小，$t_{over}$表示回合结束的最后时刻，$|t_{over}|$表示该回合结束时的总时刻个数。



# 回合结束条件

（1） 潮流不收敛，由环境实例obs的done属性返回当前时刻潮流的收敛情况；

（2） 训练达到最大时间步数；

（3） 读取到.csv文件中的最后一个断面。





# 调用方法

```python
environment.base_env.reset(seed=None, timestep=None)
```

初始化环境。

参数：seed(int)——随机种子

​			timestep(int)——初始断面的

返回：init_obs(Observation)——初始观测值



```python
environment.base_env.step(act)
```

在环境中执行一步动作。

参数：act()——动作

返回：obs(Observation Type)——观测值

​			reward(double)——奖励值

​			done(bool)——回合是否结束标志。done=True表示当前回合结束。

​			info(dict)——其他信息，包括动作非法原因、各因素奖励值、潮流是否收敛、常规机组开机容量占比。



```python
environment.base_env._update_gen_status(injection_prod_p)
```

更新各机组的开关机状态。

参数：injection_prod_p(double)——有功出力注入值

返回：无





