"""
This script allows the duckiebot to autonamouslly control it self via the MDS based upon visual cues of the enviornment obtained through the bots camera. 
"""
from PIL import Image, ImageDraw
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key
import random
import cv2 as cv
from gym_duckietown.envs import DuckietownEnv
from duckie_state import JamesPond
#from duckie_state import duckiedoos
# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

# Build the Duckiebot World
if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()
    
agent = JamesPond()

def update(dt):
    wheel_distance = 0.102
    min_rad = 0.08
    
    action = np.array([0.0, 0.0])
    
    obs = env.render_obs()
    ry, rr, rw = view_right(obs)
    ly, lr, lw = view_left(obs)
    fy, fr, fw = view_front(obs)
    
    obs_state = [ry, rr, rw, ly, lr, lw, fy, fr, fw]
    agent.understand()
    action = agent.best_action
    
    v1 = action[0]
    v2 = action[1]
    
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2
    reward = env.step(action)
    
    obs, reward, done, info = env.step(action)
    
    obs = env.render_obs()
    ry, rr, rw = view_right(obs)
    ly, lr, lw = view_left(obs)
    fy, fr, fw = view_front(obs)
    
    obs_state = [ry, rr, rw, ly, lr, lw, fy, fr, fw]
    
    agent.update(env.unwrapped.step_count, reward, obs_state)
    
    print(agent.current_state.obs, obs_state)
    
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
   
    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()

def forward(action, feet = 0.25):
    action += np.array([feet, 0])
    return action

def backward(action, feet = 0.25):
    action -= np.array([feet, 0])

    return action

def turn(action, degree = 1):
    action += np.array([0, degree])
    return action
    

# get observation and creat best predictied path based on lines
#        Crop top half of image
#        apply cv threshhold and masking to get mid line
#        canny edge detection,  hough line transformation
#        straighen line segments
#
# Base on paper: Gupta, Archit, and Arvind Easwaran. "A Low-Cost Lane-Following Algorithm for Cyber-Physical Robots." arXiv preprint arXiv:2208.10765 (2022).
# ------------------------------
def see_obj():
    # get observation from camera
    eyes = env.render_obs()
    cam_h = int(eyes.shape[0]/3)
    cam_w = int(eyes.shape[1]/6)
    cam_w2 = int(eyes.shape[1] - eyes.shape[1]/6)
    
    eyes = eyes[cam_h:, cam_w:cam_w2]
    eyes = cv.cvtColor(eyes, cv.COLOR_BGR2HSV)
    
    yellowthresh = [np.array([85,80,100]), np.array([100,255,255])]
    whitethresh = [np.array([0, 0, 100]), np.array([255, 80, 255])]
    redthresh = [np.array([100, 80, 0]), np.array([255, 255, 200])]
    
    yellow, foundy = find_line(eyes, yellowthresh, hthresh = 80, gap = 40)
    white, foundw = find_line(eyes, whitethresh, (0,255,0), 50, 30, 5)
    red, foundr = find_line(eyes, redthresh, (255,0,255), hthresh = 100)
    
    print(foundy, foundr, foundw)
    lines = cv.bitwise_or(red, white)
    lines = cv.bitwise_or(lines, yellow)
    
    print(cam_h)
    cv.imshow('Duck 0 Vision2', lines)
    return 0


def view_front(obs):
    cam_h = int(obs.shape[0]*3/5) + 3
    cam_w = int(obs.shape[1]/4)
    cam_w2 = int(obs.shape[1] - obs.shape[1]/3)
    
    obs = obs[cam_h:, cam_w:cam_w2]
    obs = cv.cvtColor(obs, cv.COLOR_BGR2HSV)
    
    yellowthresh = [np.array([85,80,100]), np.array([100,255,255])]
    whitethresh = [np.array([0, 0, 100]), np.array([255, 70, 255])]
    redthresh = [np.array([100, 80, 0]), np.array([255, 255, 200])]
    
    yellow, foundy = find_line(obs, yellowthresh, hthresh = 80, gap = 40)
    white, foundw = find_line(obs, whitethresh, (0,255,0), 75, 100, 10)
    red, foundr = find_line(obs, redthresh, (255,0,255), hthresh = 100)
    
    print(foundy, foundr, foundw)
    lines = cv.bitwise_or(red, white)
    lines = cv.bitwise_or(lines, yellow)
    cv.imshow('Duck Front', lines)
    
    return foundy, foundr, foundw

def view_left(obs):
    cam_h = int(obs.shape[0]/3)
    cam_w = 0
    cam_w2 = int(obs.shape[1] - (obs.shape[1]*4/5))
    
    obs = obs[cam_h:, cam_w:cam_w2]
    obs = cv.cvtColor(obs, cv.COLOR_BGR2HSV)
    
    yellowthresh = [np.array([85,80,100]), np.array([100,255,255])]
    whitethresh = [np.array([0, 0, 100]), np.array([255, 70, 255])]
    redthresh = [np.array([100, 80, 0]), np.array([255, 255, 200])]
    
    yellow, foundy = find_line(obs, yellowthresh, hthresh = 80, gap = 40)
    white, foundw = find_line(obs, whitethresh, (0,255,0), 75, 100, 10)
    red, foundr = find_line(obs, redthresh, (255,0,255), hthresh = 100)
    
    print(foundy, foundr, foundw)
    lines = cv.bitwise_or(red, white)
    lines = cv.bitwise_or(lines, yellow)
    cv.imshow('Duck Left', lines)
    
    return foundy, foundr, foundw

def view_right(obs):
    cam_h = int(obs.shape[0]/3)
    cam_w = int(obs.shape[1] - (obs.shape[1]/5))
    cam_w2 = int(obs.shape[1])
    
    obs = obs[cam_h:, cam_w:cam_w2]
    obs = cv.cvtColor(obs, cv.COLOR_BGR2HSV)
    
    yellowthresh = [np.array([85,80,100]), np.array([100,255,255])]
    whitethresh = [np.array([0, 0, 100]), np.array([255, 70, 255])]
    redthresh = [np.array([100, 80, 0]), np.array([255, 255, 200])]
    
    yellow, foundy = find_line(obs, yellowthresh, hthresh = 80, gap = 40)
    white, foundw = find_line(obs, whitethresh, (0,255,0), 75, 100, 10)
    red, foundr = find_line(obs, redthresh, (255,0,255), hthresh = 100)
    
    print(foundy, foundr, foundw)
    lines = cv.bitwise_or(red, white)
    lines = cv.bitwise_or(lines, yellow)
    cv.imshow('Duck Right', lines)
    
    return foundy, foundr, foundw

def find_line(img, thresh, linecolor = (0,0,255), hthresh = 50, min = 1, gap = 250):      
    color = cv.inRange(img, thresh[0], thresh[1])
    color = cv.bitwise_and(img, img, mask = color)
    color = cv.blur(color, (5,5)) #cv.bilateralFilter(color, 20, 75, 75)

    color_cany = cv.Canny(color, 50, 100)
    colines = cv.HoughLinesP(color_cany, rho = 1, theta = np.pi/180, threshold = hthresh, minLineLength = min, maxLineGap= gap)
    
    if(colines is not None):
        found = True
        col = 3
        if(len(colines) < 3):
            col = len(colines)-1
        for line in colines[0:col]:
            x1 = line[0][0]
            y1 = line[0][1]
            
            x2 = line[0][2]
            y2 = line[0][3]
        
            cv.line(color, (x1, y1), (x2, y2), linecolor, 2) 
    else:
        found = False
        
    return color, found   
            
pyglet.clock.schedule_interval(update, 2.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
