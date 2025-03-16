############################################################################
#pip install swig
#pip install box2d-py
#pip install gymnasium
#pip install vizdoom
#pip install readchar
############################################################################

import sys
import cv2
import time
import pickle
import random
import readchar
import numpy as np
import torch as th
import torchvision.transforms as transforms
import torch.nn.functional as F  
import bc_SupervisedPolicy as SPT
from sb_VizDoom import DRL_Agent
from PIL import Image  

class VizDoom_LfD():
    def __init__(self, args):
        self.environmentID = "VizdoomTakeCover-v0"
        self.seed = random.randint(0,1000)
        self.randomMode = True if args[1] == 'random' else False
        self.trainMode = True if args[1] == 'train' else False
        self.testMode = True if args[1] == 'test' else False
        self.learningAlg = args[2] if args[1] != 'random' else 'random'
        self.num_test_episodes = 20
        self.image_height_width = (84*1,84*1)
        self.policy_rendering = True
        self.path2DemoData = "VizDoom-DRL-task2/vizdoom-1img"
        self.supervisedModelID = "VizDoom-SPT.pth"
        self.demonstrationData = {}
        self.actionVector = []
        self.last4images = []

        self.transform = transforms.Compose([
            transforms.Resize((self.image_height_width[0], self.image_height_width[1])),
            transforms.ToTensor()
        ])

        self.agent = DRL_Agent(self.environmentID, self.learningAlg, self.trainMode, self.seed, 1)
        self.agent.create_environment()
        self.env = self.agent.environment

        if self.learningAlg != 'random' and not (self.learningAlg == "human" and self.trainMode):
            self.supervised_model, self.class_names = SPT.load_pretrained_classifier(self.path2DemoData, self.supervisedModelID)
        
        self.interact_with_environment()

    def get_actionFromArrowKeys(self):
        action = readchar.readkey()
        action = [ord(c) for c in action]
        #RIGHT Arrow
        if len(action) > 2 and action[2] == 67: return 1 
        #LEFT Arrow
        elif len(action) > 2 and action[2] == 68: return 2 
        else: None
    
    def save_image_in_memory(self, action):
        #print("action="+str(action))
        if action is not None:
            img = self.env.render(mode='rgb_array')
            #Convert NumPy to PIL
            img_pil = Image.fromarray(img)  
            #Resise and Convert to Tensor
            resised_img = self.transform(img_pil) 
            file_name = str(action)+'_'+str(time.time())+'.jpg'
            file_path = self.path2DemoData+'/'+str(action)
            self.demonstrationData[file_path+'/'+file_name] = resised_img
            return True
        else:
            print(f"INVALID input={action}, valid characters are 0,1")
            return False

    def save_images_in_disk(self):
        if self.testMode: return
        if self.randomMode: return

        save_episode = input("Do you want to save the data of this episode [Y,N]?")
        if save_episode.endswith('Y') or save_episode.endswith('y'):
            for file_path_name, img in self.demonstrationData.items():
                print("%s -> " % (file_path_name))
                #Convert From (C, H, W) to (H, W, C)
                img_np = img.numpy().transpose(1, 2, 0)  
                #Check Correct Pixel Range
                success = cv2.imwrite(file_path_name, img_np * 255)  
                #success = cv2.imwrite(file_path_name, img)
                if not success:
                    print(f"Failed to save image: {file_path_name}")

            print("Saved episode in "+str(self.path2DemoData))
        self.demonstrationData = {}

    def get_last4images(self):
        last_img = self.env.render(mode='rgb_array')
        #Resize Image
        last_img = cv2.resize(last_img, self.image_height_width)  

        #Initialize with 4 Identical Images if Empty
        if len(self.last4images) == 0:
            self.last4images = [last_img] * 4
        else:
            #Shift Left
            self.last4images[:-1] = self.last4images[1:]  
            #Append New Image
            self.last4images[-1] = last_img  

        img_seq = np.concatenate(self.last4images, axis=1)
        img_tensor = th.tensor(img_seq, dtype=th.float32)
        #Add Batch Dimension
        img_tensor = img_tensor.unsqueeze(0)  
        return img_tensor

    def get_action_from_agent(self, obs):
        if self.learningAlg == "random":
            action = self.env.action_space.sample()

        elif self.learningAlg == "human" and self.trainMode:
            action = self.get_actionFromArrowKeys()
            while not self.save_image_in_memory(action): 
                action = self.get_actionFromArrowKeys()
                continue
        
        elif self.learningAlg == "human" and self.testMode:
            img_array = self.get_last4images()
            img_tensor = th.tensor(img_array, dtype=th.float32).permute(0, 3, 1, 2)
            self.supervised_model.eval() 
            #No Gradient Calculations at Test Time
            with th.no_grad():  
                predictions = self.supervised_model(img_tensor)

            #print("predictions=",predictions)
            #probabilities = F.softmax(predictions, dim=1).squeeze(0)

            #Softmax is Applied to Last Dimension Regardless of Shape
            probabilities = F.softmax(predictions, dim=-1).squeeze()
            #print("probabilities=",probabilities)
            action = th.argmax(probabilities).item()
            probability_of_best_action = probabilities[action]

            #Selects a Random Action but Only if Uncertain in Learnt Probabilities  
            if probability_of_best_action < 0.90: 
                distribution = th.distributions.Categorical(probabilities)
                action = distribution.sample().item()
                #Due to Prediction-Action Pairs 0=1 and 1=2
            action = action+1  

        else:
            pass

        return [action]
    
    def interact_with_environment(self):
        print("INTERACTING with the environment...")

        steps_per_episode = 0
        reward_per_episode = 0
        total_cummulative_reward = 0
        episode = 1
        obs = self.env.reset()
        self.env.render("human")
        while True and self.policy_rendering:
            action = self.get_action_from_agent(obs)
            obs, reward, done, info = self.env.step(action)
            #print("e=%s t=%s a=%s r=%s, d=%s" % (episode, steps_per_episode, action, reward, done))

            steps_per_episode += 1
            reward_per_episode += reward
            #Or Any(Done):
            if done:
                print("episode=%s, steps_per_episode=%s, reward_per_episode=%s" % \
                       (episode, steps_per_episode, reward_per_episode))
                total_cummulative_reward += reward_per_episode
                steps_per_episode = 0
                reward_per_episode = 0
                episode += 1
                obs = self.env.reset()
                self.save_images_in_disk()
                self.last4images = []

            self.env.render("human")

            if episode > self.num_test_episodes: 
                print("total_cummulative_reward=%s avg_cummulative_reward=%s" % \
                      (total_cummulative_reward, total_cummulative_reward/self.num_test_episodes))
                break
        self.env.close()

if len(sys.argv)<2 or len(sys.argv)>4:
    print("USAGE: bc_VizDoom_FromDemonstration.py (random|train|test) [human|others...not supported]")
    print("EXAMPLE1: python bc_VizDoom_FromDemonstration.py train human")
    print("EXAMPLE2: python bc_VizDoom_FromDemonstration.py test human")
    exit(0)
else:
    VizDoom_LfD(sys.argv)