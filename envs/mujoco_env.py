import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class MujocoEnv(gym.Env):
    def __init__(self, task_name, action_repeat=1, stack=1, seed=0):
        """
        包装 Gymnasium 的 MuJoCo 环境以适配 FlowRL 的代码接口。
        """
        # 尝试创建环境，增加 render_mode 以避免部分新版 gym 警告
        self._env = gym.make(task_name)

        self._action_repeat = action_repeat
        self._stack = stack
        self._seed = seed

        # 1. 设置标准的 action_space
        self.action_space = self._env.action_space

        # [关键修复]：main.py 第142行使用了 env._action_space
        # 我们手动创建一个别名指向 action_space
        self._action_space = self.action_space

        # 2. 处理观察空间 (Observation Space)
        if stack > 1:
            self._frames = deque(maxlen=stack)
            # 创建一个新的 Box 空间，增加一个维度用于堆叠 (stack, feature_dim)
            low = np.repeat(self._env.observation_space.low[np.newaxis, ...], stack, axis=0)
            high = np.repeat(self._env.observation_space.high[np.newaxis, ...], stack, axis=0)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=self._env.observation_space.dtype
            )
        else:
            self.observation_space = self._env.observation_space

        # [兼容性] 设置 observation_space 的别名（虽然 main.py 似乎没用 _observation_space，但以防万一）
        self._observation_space = self.observation_space

    def seed(self, seed):
        """适配 main.py 中的 env.seed(config.seed) 调用"""
        self._seed = seed

    def reset(self):
        """
        适配 main.py 接口：
        1. 使用保存的 seed 进行重置
        2. 只返回 observation (去掉 info)
        """
        # Gymnasium 的 reset 返回 (obs, info)
        obs, info = self._env.reset(seed=self._seed)

        if self._stack > 1:
            # 初始化堆叠帧
            for _ in range(self._stack):
                self._frames.append(obs)
            return np.stack(self._frames)

        return obs

    def step(self, action):
        """
        适配 main.py 接口：
        1. 执行动作重复
        2. 合并 terminated 和 truncated 为 done
        3. 返回 (obs, reward, done, info) 四个值
        """
        total_reward = 0.0
        done = False
        combined_info = {}

        for _ in range(self._action_repeat):
            # Gymnasium 的 step 返回 5 个值
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            done = terminated or truncated
            combined_info.update(info)
            if done:
                break

        if self._stack > 1:
            self._frames.append(obs)
            obs = np.stack(self._frames)

        return obs, total_reward, done, combined_info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        self._env.close()

    # [关键修复]：将未找到的属性（如 _max_episode_steps）委托给底层的 gym 环境
    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped