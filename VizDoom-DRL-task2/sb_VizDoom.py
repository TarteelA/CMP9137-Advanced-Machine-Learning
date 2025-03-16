#####################################################
#pip install vizdoom
#pip install stable-baselines3
#pip install opencv-python
#apt-get update && apt-get install -y python3-opencv
#apt-get install libgl1
#pip install swig
#pip install "gymnasium[box2d]"
#pip install readchar
#apt install qt6-base-dev
#Version 1.0, Adapted for Training and Testing Agents with MarioBros Environments
#Version 2.0, Rewritten for Training and Testing Agents with VizDoom Environments
#Version 2.1, Revised for Training and Testing Agents with Non-VizDoom Environments
#such as "LunarLander-v3", which Makes Program More General for DRL
#####################################################

import os
import sys
import cv2
import time
import pickle
import random
import numpy as np
import gymnasium 
import vizdoom.gymnasium_wrapper
from stable_baselines3 import DQN,A2C,PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv


#Returns an Observation Containing a Resised Image and Other Info
#frame_skip=Number of Frames to Skip Between Actions to Speed Up Training
class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shape, frame_skip):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = frame_skip 

        #Create New Observation Space with New Shape
        print(env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        #If Channels Are not 3 (Corresponding to RGB)
        if observation.shape[-1] != 3:  
            #Convert to RGB24
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) 
        return observation

#Converts RGB Observations to Grayscale to Reduce Input Dimensionality
#gymnasium.spaces.Box() Creates a Continuous Space for Observations
class GrayscaleObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            #Only 1 Channel (Grayscale) Instead of 3 (RGB)
        ) 

    def observation(self, observation):
        #Check that Observation is in RGB Format Before Converting to Grayscale
        #If Channels Are not 3 (Corresponding to RGB)
        if observation.shape[-1] != 3:  
            #Convert to RGB24
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) 

        #Convert to Grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) 
        #Expand Dimensions to Match Input Shape (H, W, 1)
        gray = np.expand_dims(gray, axis=-1) 
        return gray


#Class for Creating DRL Agents Through Environment Setup, Model Creation, Training, Evaluation, and Policy Rendering
class DRL_Agent:
    #Initialise DRL Agent with Given Parameters
    def __init__(self, environment_id, learning_alg, train_mode=True, seed=None, n_envs=8, frame_skip=4):
        self.environment_id = environment_id
        self.learning_alg = learning_alg
        self.train_mode = train_mode
        self.seed = seed if seed else random.randint(0, 1000)
        self.policy_filename = f"{learning_alg}-{environment_id}-seed{self.seed}.policy.pkl"
        #Number of Environments for Training or 1 for Testing
        self.n_envs = n_envs if train_mode else 1  
        #Number of Frames to Skip Between Actions in Environment
        self.frame_skip = frame_skip 
        #image res (height, width):e.g., (240, 320); (120, 160); (60, 80);
        self.image_shape = (84, 84)  
        #Total Number of Timesteps for Training Agent
        self.training_timesteps = 10000 
        #Number of Episodes to Run for Testing Trained Agent
        self.num_test_episodes = 20 
        #Learning Rate for Optimiser During Training
        self.l_rate = 0.00083 
        #Discount Factor for Future Rewards (used in RL Algorithms)
        self.gamma = 0.995 
        #Number of Steps/Actions Agent Will Take Before Updating Model
        self.n_steps = 512 
        #if True, Shows Visualisations of Learnt Behaviour
        self.policy_rendering = True 
        #Delay in Rendering
        self.rendering_delay = 0.05 if self.environment_id.find("Vizdoom") > 0 else 0 
        #Directory to Store Logs Containing Agent Performance
        self.log_dir = './logs' 
        #Initialises Model that will Define Agent's Policy & Learning Behavior
        self.model = None 
        #Initialises Policy to "MlpPolicy" or "CnnPolicy" Depending on Environment
        self.policy = None 
        #Initialises Environment as None, to be Set Later (Gym Environment)
        self.environment = None 

        self._check_environment()
        self._create_log_directory()

    #Check if Specified Environment is Available, Stops Execution Otherwise
    def _check_environment(self):
        available_envsA = [env for env in gymnasium.envs.registry.keys() if "LunarLander" in env]
        available_envsB = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
        if self.environment_id in available_envsA :
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsA}")
        elif self.environment_id in available_envsB:
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsB}")
        else:
            print(f"UNKNOWN environment={self.environment_id}")
            print(f"AVAILABLE_ENVS={available_envsA, available_envsB}")
            sys.exit(0)

    #Creates a Log Directory if it Doesn't Exist Already
    def _create_log_directory(self):
        #if self.environment_id.find("Vizdoom") == -1: return
        #Only Logs Info for Vizdoom Environments and Uses a Single Folder
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir) 
            print(f"Log directory created: {self.log_dir}")
        else:
            print(f"Log directory {self.log_dir} already exists!")
    
    #Wrapper Function to Customise Environments for Image-Based Obsertations
    def wrap_env(self, env):
        env = ObservationWrapper(env, shape=self.image_shape, frame_skip=self.frame_skip)
        #Convert to Grayscale
        env = GrayscaleObservationWrapper(env) 
        if self.train_mode: 
            #Scale Rewards for Training Stability, only During Training
            env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01) 
        return env

    #Create Vectorised Environment with Multiple Parallel Environments
	#Report Mean and Standard Deviation of Performance Metrics Across Seeds
    def create_environment(self, use_rendering=False):
        print("self.environment_id="+str(self.environment_id))

        if self.environment_id.find("Vizdoom") == -1:
            if use_rendering: 
                self.environment = gymnasium.make(self.environment_id, render_mode="human")
            else:
                self.environment = gymnasium.make(self.environment_id)
            self.environment = DummyVecEnv([lambda: self.environment])
            self.environment = VecMonitor(self.environment, self.log_dir) 
            self.policy = "MlpPolicy"
        
        else:
            self.environment = make_vec_env(
                self.environment_id,
                n_envs=self.n_envs,
                seed=self.seed,
                monitor_dir=self.log_dir,
                #Applies Wrappers Inside Function
                wrapper_class=self.wrap_env  
            )
            #Stacks Frames for Temporal Context
            self.environment = VecFrameStack(self.environment, n_stack=4)  
            #Transposes Image for Correct Format (Channel First)
            self.environment = VecTransposeImage(self.environment) 
            self.policy = "CnnPolicy"

        print("self.environment.action_space:",self.environment.action_space)

    #Create RL Model Based on Chosen Learning Algorithm
    def create_model(self):
        if self.learning_alg == "DQN":
            self.model = DQN(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma, buffer_size=10000, batch_size=64, exploration_fraction=0.9, verbose=1)
			
        elif self.learning_alg == "A2C":
            self.model = A2C(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma, verbose=1)
			
        elif self.learning_alg == "PPO":
            self.model = PPO(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma, verbose=1)
			
        else:
            print(f"Unknown LEARNING_ALG={self.learning_alg}")
            sys.exit(0)

    #Train Agent's Model or Load a Pre-Trained Model from a File
    def train_or_load_model(self):
        print(self.model)
        if self.train_mode:
            self.model.learn(total_timesteps=self.training_timesteps)
            print(f"Saving policy {self.policy_filename}")
            pickle.dump(self.model.policy, open(self.policy_filename, 'wb'))
        else:
            print("Loading policy...")
            with open(self.policy_filename, "rb") as f:
                policy = pickle.load(f)
            self.model.policy = policy

    #Evaluate Policy on Environment by Running a Number of Test Episodes
    def evaluate_policy(self):
        print("Evaluating policy...")
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=self.num_test_episodes)
        print(f"EVALUATION: mean_reward={mean_reward} std_reward={std_reward}")

    #Render Agent's Behavior in Environment and Track Cumulative Reward
    def render_policy(self):
        steps_per_episode = 0
        reward_per_episode = 0
        total_cummulative_reward = 0
        episode = 1
        self.create_environment(True)
        env = self.environment
        obs = env.reset()

        print("DEMONSTRATION EPISODES:")
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps_per_episode += 1
            reward_per_episode += reward
            if any(done):
                print(f"episode={episode}, steps_per_episode={steps_per_episode}, reward_per_episode={reward_per_episode}")
                total_cummulative_reward += reward_per_episode
                steps_per_episode = 0
                reward_per_episode = 0
                episode += 1
                obs = env.reset()
            if self.policy_rendering:
                env.render("human")
                time.sleep(self.rendering_delay)
            if episode > self.num_test_episodes:
                print(f"total_cummulative_reward={total_cummulative_reward} avg_cummulative_reward={total_cummulative_reward / self.num_test_episodes}")
                break
        env.close()

    #Main Method to Run DRL Agent
    def run(self):
        self.create_environment()
        self.create_model()
        self.train_or_load_model()
        self.evaluate_policy()
        self.render_policy()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("USAGE: sb-VizDoom.py (train|test) (DQN|A2C|PPO) [seed_number]")
        print("EXAMPLE1: sb-VizDoom.py train PPO")
        print("EXAMPLE2: sb-VizDoom.py test PPO 476")
        sys.exit(0)
    
    #Simpler Environment (No Images, Simpler Observations)
    environment_id = "LunarLander-v3"  
    #Default Environment (Image Observations)
    #environment_id = "VizdoomTakeCover-v0" 
    #Boolean Parameter Train/Test
    train_mode = sys.argv[1] == 'train' 
    #Argument to Communicate Algorithm to Use
    learning_alg = sys.argv[2] 
    #Random or Predefined Seed
    seed = random.randint(0, 1000) if train_mode else int(sys.argv[3]) 
    
    agent = DRL_Agent(environment_id, learning_alg, train_mode, seed)
    agent.run()