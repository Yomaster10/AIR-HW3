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

## Original imports
import rospy, yaml, os, argparse
import numpy as np

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovarianceStamped
import dynamic_reconfigure.client
##

## Our imports
import time, actionlib
from nav_msgs.msg import Odometry #, OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from MRS_236609.srv import ActionReq, ActionReqResponse
##

CUBE_EDGE = 0.5
MAX_LINEAR_VELOCITY_FREE = 0.22 #[m/s]
MAX_LINEAR_VELOCITY_HOLDING = 0.15 #[m/s]

"""
def euler_from_quaternion(quaternion):
    q0, q1, q2, q3 = quaternion
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
    pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
    return roll, pitch, yaw

class Pose:
    #x = None; y = None; yaw = None
    def __init__(self):
        self.x = 0
        self.y = 0
        self.yaw = 0

    def update(self, data):
        odom = data.pose.pose
        self.x, self.y = odom.position.x, odom.position.y
        self.yaw = euler_from_quaternion([odom.orientation.x, odom.orientation.z, odom.orientation.z, odom.orientation.w])[2]

# save the turtlebot's place as global
pose = Pose()
"""

class TurtleBot:
    def __init__(self):
        self.start_time = time.time()
        
        self.curr_loc = None
        rospy.Subscriber('/odom', Odometry, callback=self.update_loc)

        self.action_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        self.time = None
        self.initial_position = None
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, callback=self.set_initial_position)
        
        print("Waiting for an initial position...")
        while self.initial_position is None:
            continue
        print("\tThe initial position is {}".format(self.initial_position))

        #self.curr_loc = self.initial_position
        #self.do_action = rospy.ServiceProxy('/do_action', ActionReq)

    def set_initial_position(self, msg):
        initial_pose = msg.pose.pose
        self.initial_position = np.array([initial_pose.position.x, initial_pose.position.y])

    def update_loc(self, msg):
        odom = msg.pose.pose
        self.curr_loc = [odom.position.x, odom.position.y]

    def server_unavailable(self):
        if time.time() - self.start_time > 100:
            rospy.logerr("Action server is unavailable")
            rospy.signal_shutdown("Action server is unavailable")
            return True
        return False

    def add_goal(self, x, y, yaw=1.0):
        # Creates a new goal with the MoveBaseGoal constructor
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = yaw

        self.action_client.send_goal(goal)
        print("New goal command received...")

    def move_to_ws(self, pos, threshold=0.1):
        self.action_client.wait_for_server()
        self.add_goal(pos[0], pos[1])
        if self.curr_loc is None:
            curr_pos = self.initial_position
        else:
            curr_pos = self.curr_loc
        while np.linalg.norm(np.array([pos[0] - curr_pos[0], pos[1] - curr_pos[1]])) > threshold:
            wait = self.action_client.wait_for_result()
            #print(self.curr_loc)
            curr_pos = self.curr_loc

        #print(wait)
            #if self.server_unavailable():
            #    exit(1)
            #else:
        #if wait:
        return self.action_client.get_result()
        #else:
        #    return False

    def run(self, ws, tasks, time_thresh):
        self.time = time_thresh

        # ==== Custom =======
        ws_locs = {}; ws_tasks = {}
        for w, val in ws.items():
            #print(w + ' center is at ' + str(val.location) + ' and its affordance center is at ' + str(
            #    val.affordance_center))
            ws_locs[w] = val.affordance_center
            ws_tasks[w] = val.tasks

        acts, rews = self.calc_possible_rewards(ws, tasks)

        if self.curr_loc is None:
            best_plan = self.make_plan(self.initial_position, acts, rews, ws_tasks, ws_locs)
        else:
            best_plan = self.make_plan(self.curr_loc, acts, rews, ws_tasks, ws_locs)
    
        i = 0
        while time.time() - self.start_time < time_thresh:
            p = best_plan[i]
            i += 1

            #print(best_plan)
            success = True
            #for p in best_plan:
            print(p)
            for t in p[1:]:
                next_ws = list(t.keys())[0]
                ws_pos = ws_locs[next_ws]
                #result = False
                #while not result:
                #    result = self.move_to_ws(ws_pos)
                #    print(result)
                result = self.move_to_ws(ws_pos)
                if not result:
                    print('Failure')

                do_action = rospy.ServiceProxy('/do_action', ActionReq)
                act = t[next_ws]
                    
                for a in act:
                    #res = self.do_action(next_ws, a)
                    res = do_action(next_ws, a)
                    #print(a)
                    #print(res)
                    success = success and res.success
                
            if success:
                acts, rews, ws_tasks = self.clear_activities(p, acts, rews, ws_tasks)
                print("Activity succeeded!")

            if i > len(best_plan)-1:
                best_plan = self.make_plan(self.curr_loc, acts, rews, ws_tasks, ws_locs)
                print("New plan!")
                i = 0

            #if self.curr_loc is None:
            #    best_plan = self.make_plan(self.initial_position, acts, rews, ws_tasks, ws_locs)
                #best_task = self.get_best_activity(self.initial_position, acts, rews, ws_locs, ws_tasks)
            #else:
                
                #best_task = self.get_best_activity(self.curr_loc, acts, rews, ws_locs, ws_tasks)
            
        # ===========================

    def calc_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def calc_possible_rewards(self, ws, tasks):
        activities = {}; rewards = {}
        for idx, task in enumerate(tasks):
            task_name = 'Task ' + str(idx)
            rewards[task_name] = tasks[task]
            i = 0; activities[task_name] = []
            while i < len(task):
                curr_task = task[i:i+4]
                locs = []
                for w, val in ws.items():
                    if curr_task in val.tasks:
                        locs.append(w)
                activities['Task ' + str(idx)].append({curr_task:locs})
                i += 6
        return activities, rewards

    def make_plan(self, current_pos, acts, rews, tasks, locs):
        reward = 0; best_plan = []
        best_task = self.get_best_activity(current_pos, acts, rews, tasks, locs)
        while best_task:
            best_plan.append(best_task)
            reward += rews[best_task[0]]
            final_ws = list(best_task[-1].keys())[0]
            current_pos = locs[final_ws]
            acts, rews, tasks = self.clear_activities(best_task, acts, rews, tasks)
            best_task = self.get_best_activity(current_pos, acts, rews, tasks, locs)
        return best_plan
    
    def get_best_activity(self, current_pos, acts, rews, tasks, locs):
        best_task = None; max_reward = 0
        for a in acts:
            # Step 1: Get valid sequences for each activity
            last = [[b] for b in list(acts[a][0].values())[0]]
            for i in range(1,len(acts[a])):
                new = []
                y = list(acts[a][i].keys())[0]
                for d in acts[a][i][y]:
                    for l in last:
                        if d == l[-1]: # can't remain at the same workstation
                            continue  
                        next_l = list(np.copy(l))
                        next_l.append(d)
                        new.append(next_l)
                last = new

            # Step 2: Get best sequence for each activity
            best_seq = None; min_cost = 10000
            for l in last:
                cost = 0
                prev_loc = current_pos
                for i in l:
                    cost += self.calc_distance(locs[i], prev_loc)
                    prev_loc = locs[i]
                if cost < min_cost:
                    min_cost = cost
                    best_seq = l
            
            # Step 3: Compare best sequence of each activity to the global best
            res = rews[a] - min_cost
            if res > max_reward:
                activities = [list(d.keys())[0] for d in acts[a]]
                pot_best_task = [a]
                for i in range(len(best_seq)):
                    seq = []
                    if i > 0:
                        if 'PL-'+obj in tasks[best_seq[i]]:
                            seq.append('PL-'+obj)
                    seq.append(activities[i])
                    if i < len(best_seq)-1:
                        pickup_found = False
                        for j in tasks[best_seq[i]]:
                            if j[:2] == 'PU':
                                obj = j[-1]
                                if 'PL-'+obj in tasks[best_seq[i+1]]:
                                    seq.append(j)
                                    pickup_found = True
                                    break
                        if not pickup_found:
                            break
                    pot_best_task.append({best_seq[i]:seq})
                if pickup_found:
                    best_task = pot_best_task
                    max_reward = res

        # Step 4: Return the best activity (and sequence) to do among all available options
        return best_task
    
    def clear_activities(self, best_task, acts, rews, tasks):
        task = best_task[0]
        
        new_acts = {}
        for a in acts:
            if a == task:
                continue
            new_acts[a] = acts[a]

        new_rews = {}
        for r in rews:
            if r == task:
                continue
            new_rews[r] = rews[r]
        
        completed_acts = {}
        for i in range(1,len(best_task)):
            w = list(best_task[i].keys())[0]
            if w not in completed_acts:
                completed_acts[w] = best_task[i][w]
            else:
                for j in best_task[i][w]:
                    completed_acts[w].append(j)

        new_tasks = {}
        for t in tasks.keys():
            if t not in completed_acts:
                new_tasks[t] = tasks[t]
                continue
            new_tasks[t] = []
            for x in tasks[t]:
                if x in completed_acts[t]:
                    continue
                new_tasks[t].append(x)
                
        return new_acts, new_rews, new_tasks

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
        #self.curr_aff_center_dist = None #custom addition

    def update_affordance_center(self, new_center):
        self.affordance_center = new_center

    #def update_curr_aff_center_dist(self, current_pos): #custom addition
    #    self.curr_aff_center_dist = self.calc_distance(self.affordance_center, current_pos)
    #    return self.curr_aff_center_dist
    
    #def calc_distance(self, point1, point2): #custom addition
    #    return np.linalg.norm(np.array(point1) - np.array(point2))

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
    time_thresh = args.time

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
    try:
        tb3.run(ws, tasks, time_thresh)
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation Exception.")
