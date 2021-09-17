import numpy as np

def wrap_action(adjust_gen_p):
    act = {
        'adjust_gen_p': adjust_gen_p,
        'adjust_gen_v': np.zeros_like(adjust_gen_p)
    }
    return act

def feature_process(settings, obs):
    loads = []
    loads.append(obs.load_p)
    loads.append(obs.load_q)
    loads.append(obs.load_v)
    loads = np.concatenate(loads)

    # prods
    prods = []
    prods.append(obs.gen_p)
    prods.append(obs.gen_q)
    prods.append(obs.gen_v)
    prods = np.concatenate(prods)
    
    # rho
    rho = np.array(obs.rho) - 1.0

    next_load = obs.nextstep_load_p

    # action_space
    action_space_low = obs.action_space['adjust_gen_p'].low.tolist()
    action_space_high = obs.action_space['adjust_gen_p'].high.tolist()
    for id in settings.renewable_ids:
        action_space_low[id] = action_space_high[id]
    action_space_low[settings.balanced_id] = 0.0
    action_space_high[settings.balanced_id] = 0.0
    
    # steps_to_reconnect_line = obs.steps_to_reconnect_line.tolist()
    steps_to_recover_gen = obs.steps_to_recover_gen.tolist()
    # gen_status = obs.gen_status.tolist()
    # 1 stands for can be opened
    gen_status = ((obs.gen_status == 0) & (obs.steps_to_recover_gen == 0)).astype(float).tolist()
    steps_to_close_gen = obs.steps_to_close_gen.tolist()

    gen_features = np.concatenate([gen_status, prods, action_space_low, action_space_high, steps_to_recover_gen])
    gen_features = np.transpose(gen_features.reshape((7,-1))).reshape(7*settings.num_gen)
    
    features = np.concatenate([
        gen_features.tolist(),
        loads,
        rho.tolist(), next_load
        # gen_status
    ])

    return features

def action_process(settings, obs, model_output_act):
    N = len(model_output_act)

    # model_output_act, mask = model_output_act[:N//2], model_output_act[N//2:]
    # gen_status = ((self.env.raw_obs.gen_status == 0) & (self.env.raw_obs.steps_to_recover_gen == 0)).astype(float)
    # idx = ((mask <=0 ) & (gen_status == 1))
    # model_output_act[np.where(idx==1)] = -1
    # gen_status = ((obs.gen_status == 0) & (obs.steps_to_recover_gen == 0)).astype(float)
    # idx = ((model_output_act <=0 ) & (gen_status == 1))
    # model_output_act[np.where(idx==1)] = -1

    gen_p_action_space = obs.action_space['adjust_gen_p']

    gen_p_low_bound = gen_p_action_space.low
    gen_p_high_bound = gen_p_action_space.high

    # gen_v_action_space = self.env.raw_obs.action_space['adjust_gen_v']

    # gen_v_low_bound = gen_v_action_space.low
    # gen_v_high_bound = gen_v_action_space.high

    # low_bound = np.concatenate([gen_p_low_bound, gen_v_low_bound])
    # high_bound = np.concatenate([gen_p_high_bound, gen_v_high_bound])
    low_bound = gen_p_low_bound
    high_bound = gen_p_high_bound

    for id in settings.renewable_ids:
        low_bound[id] = high_bound[id]
        
    mapped_action = low_bound + (model_output_act - (-1.0)) * (
        (high_bound - low_bound) / 2.0)
    mapped_action[settings.balanced_id] = 0.0
    # mapped_action[N//2 + self.settings.balanced_id] = 0.0
    mapped_action = np.clip(mapped_action, low_bound, high_bound)
    return wrap_action(mapped_action)