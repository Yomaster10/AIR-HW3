#!/usr/bin/env python
'''
In order to run the assignment, open 4 terminals, and in each one source ROS Melodic and the workspace
(and set the Turtlebot3 model) in each one:
$ cd ~/AIR_ws/
$ source devel/setup.bash
$ export TURTLEBOT3_MODEL=burger

Then run the following commands in each respective terminal:

Terminal 1:
$ roslaunch MRS_236609 turtlebot3_workstations.launch

Terminal 2:
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/AIR_ws/src/MRS_236609/maps/map3.yaml

Terminal 3:
$ rosrun MRS_236609 assignment3_manager.py

Terminal 4:
$ rosrun MRS_236609 assignment3.py --time 5.0
'''

import rospy
import yaml
import os
import argparse
import numpy as np

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovarianceStamped
import dynamic_reconfigure.client

CUBE_EDGE = 0.5

MAX_LINEAR_VELOCITY_FREE = 0.22 #[m/s]
MAX_LINEAR_VELOCITY_HOLDING = 0.15 #[m/s]

class TurtleBot:
    def __init__(self):
        self.initial_position = None
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, callback=self.set_initial_position)
        self.time = None

        print("Waiting for an initial position...")
        while self.initial_position is None:
            continue
        print("The initial position is {}".format(self.initial_position))

    def set_initial_position(self, msg):
        initial_pose = msg.pose.pose
        self.initial_position = np.array([initial_pose.position.x, initial_pose.position.y])

    def run(self, ws, tasks, time):
        self.time = time

        # ==== You can delete =======

        #print(tasks)
        #for task in tasks:
                
                #if task[0:4] == 
        #dist2aff = {}
        for w, val in ws.items():
            #print(type(val.location))
            #print(type(self.initial_position))
            print(w + ' center is at ' + str(val.location) + ' and its affordance center is at ' + str(
                val.affordance_center))
            #for task in tasks:

                #if task[0:4] == 
            print(val.tasks)
            print(val.update_curr_aff_center_dist(self.initial_position))
            #dist2aff[w] = self.calc_distance(val.affordance_center, self.initial_position)
        #print(dist2aff)
        self.calc_possible_rewards(ws, tasks, time)
    
        # ===========================

    def calc_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
        #return np.linalg.norm(np.array([point1[i] - point2[i] for i in range(len(point1))]))
    
    def calc_possible_rewards(self, ws, tasks, time):
        for idx, task in enumerate(tasks):
            count = len(task)
            acts = {'Task ' + str(idx) : []}
            i = 0
            while i < count:
                curr_task = task[i:i+4]
                locs = []
                for w, val in ws.items():
                    if curr_task in val.tasks:
                        locs.append(w)
                acts['Task ' + str(idx)].append({curr_task:locs})
                i += 6
            print(acts)
            #start_act = task[:3]
            #end_act = task[:-3]

        return


# ======================================================================================================================

def analyse_res(msg):
    result = {}
    for line in msg.split("\n"):
        if line:
            parts = line.split(" ")
            key = parts[0]
            x = float(parts[-2])
            y = float(parts[-1])
            result[key] = [x, y]
    return result


class Workstation:
    def __init__(self, location, tasks):
        self.location = location
        self.tasks = tasks
        self.affordance_center = None        
        self.curr_aff_center_dist = None #custom addition

    def update_affordance_center(self, new_center):
        self.affordance_center = new_center

    def update_curr_aff_center_dist(self, current_pos): #custom addition
        self.curr_aff_center_dist = self.calc_distance(self.affordance_center, current_pos)
        return self.curr_aff_center_dist
    
    def calc_distance(self, point1, point2): #custom addition
        return np.linalg.norm(np.array(point1) - np.array(point2))

# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':
    rospy.init_node('assignment3')

    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--time",
        type=float,
        default=2.0,
    )
    args = CLI.parse_args()
    time = args.time

    gcm_client = dynamic_reconfigure.client.Client("/move_base/global_costmap/inflation_layer")
    gcm_client.update_configuration({"inflation_radius": 0.2})
    lcm_client = dynamic_reconfigure.client.Client("/move_base/local_costmap/inflation_layer")
    lcm_client.update_configuration({"inflation_radius": 0.2})

    ws_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/workstations_config.yaml"
    tasks_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/config/tasks_config.yaml"
    ws = {}
    with open(ws_file, 'r') as f:
        data = yaml.load(f)
        num_ws = data['num_ws']
        for i in range(num_ws):
            ws['ws' + str(i)] = Workstation(data['ws' + str(i)]['location'], data['ws' + str(i)]['tasks'])

    rospy.wait_for_service('/affordance_service', timeout=5.0)
    service_proxy = rospy.ServiceProxy('/affordance_service', Trigger)
    res = service_proxy()
    aff = analyse_res(res.message)
    for key, val in ws.items():
        val.update_affordance_center(aff[key])

    with open(tasks_file, 'r') as f:
        data = yaml.load(f)
        tasks = data['tasks']

    tb3 = TurtleBot()
    tb3.run(ws, tasks, time)
