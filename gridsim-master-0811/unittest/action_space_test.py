from Environment.base_env import Environment
from utilize.action_space import ActionSpace
from utilize.settings import settings
from Agent.RandomAgent import RandomAgent

class TestActionSpace():
    def run(self):
        print("test action space")
        action_space_cls = ActionSpace(settings)
        env = Environment(settings, "EPRIReward")
        obs = env.reset()
        action = my_agent.act(obs, 0, False)
        obs, reward, done, info = env.step(action)
        print("action space = ", obs.action_space)
        print(info)

if __name__ == "__main__":
    my_agent = RandomAgent(settings.num_gen)
    test = TestActionSpace()
    test.run()
