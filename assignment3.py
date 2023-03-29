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
$ rosrun MRS_236609 assignment3.py --time 300.0
'''

## Original imports
import rospy, yaml, os, argparse
import numpy as np

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseWithCovarianceStamped
import dynamic_reconfigure.client
##

## Our imports
import time, actionlib, nav_msgs.srv
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from costmap_listen_and_update import CostmapUpdater
from MRS_236609.srv import GetCostmap, ActionReq
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
##

# Global constants
CUBE_EDGE = 0.5 #[m]
MAX_LINEAR_VELOCITY_FREE = 0.22 #[m/s]
MAX_LINEAR_VELOCITY_HOLDING = 0.15 #[m/s]

class TurtleBot:
    def __init__(self):
        self.start_time = time.time()
        self.latest_ws = None
        self.plan_marker = None
        
        # Costmap service
        self.cmu = CostmapUpdater()
        rospy.Service('/initial_costmap', GetCostmap, self.get_costmap)
        while self.cmu.cost_map is None:
            continue

        # Navigation action client
        self.action_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

        # Pose subscribers
        pose_msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout=5)
        self.curr_loc = [pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y]
        print("Initial Position: {}".format(self.curr_loc))
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, callback=self.update_loc)

        # Plan visualization publisher
        plan_viz_topic = '/plan_viz'
        self.plan_viz_pub = rospy.Publisher(plan_viz_topic, MarkerArray, queue_size = 10)
        
    ### I. Functions for retrieving information from ROS
    def get_costmap(self):
        """ This function retrieves the costmap from the corresponding service """
        return self.cmu.initial_msg

    def update_loc(self, msg):
        """ This function serves as the callback for the odometry subscriber, getting the current XY position of the robot """
        odom = msg.pose.pose
        self.curr_loc = [odom.position.x, odom.position.y]

        if self.plan_marker is not None:
            self.plan_viz_pub.publish(self.plan_marker)
    
    def get_current_reward(self):
        """ Retrieves the current accumulated reward from the current_cost topic """
        msg = rospy.wait_for_message('/current_cost', String, timeout=10)
        return msg.data.split("reward: ")[1]

    ### II. Main solver function
    def run(self, ws, tasks, time_thresh):
        """ This function is the main bulk of our implementation, it acts as our solver """
        
        ## Step 1: Acquire workstation and task info.
        ws_locs = {}; ws_tasks = {}
        for w, val in ws.items():
            ws_locs[w] = val.affordance_center
            ws_tasks[w] = val.tasks

        ## Step 2: Construct the lookup table
        self.ws_plans = self.calc_ws_plans(ws_locs)
        acts, rews = self.calc_possible_rewards(ws, tasks)
        max_rew = sum(rews.values())
        print("\nTotal Reward Available: {}".format(max_rew))

        Res = self.calc_base(acts, rews, ws_tasks, acts_time=0) # assume action costs are uniform and fixed, here we take them to be zero (instantaneous)
        print("\nLookup Table:")
        for k in range(len(Res.keys())):
            key = "Task {}".format(k)
            print("\t{}:".format(key))
            for item in Res[key]:
                item_copy = item.copy()
                item_copy['Cost'] = int(round(item_copy['Cost']))
                item_copy['Time'] = int(round(item_copy['Time']))
                print("{}".format(item_copy))

        ## Step 3: Choose the next best task and execute it in a loop, until time runs out
        print("\nBeginning Run...")
        while time.time() - self.start_time < time_thresh and len(Res) > 0:
            
            ## Step 3.1: Choose the best task and activity sequence to tackle
            best_task, best_seq = self.get_best_task(self.curr_loc, ws_locs, time_thresh - (time.time() - self.start_time), Res)
            if best_task == None: # there are no tasks left that can be completed in the remaining time
                break
            print('\tAttempting ' + best_task + '...')

            # Activity sequence visualization
            markerArray = self.create_plan_marker_array(self.curr_loc, best_seq, ws_locs)
            self.plan_marker = markerArray

            ## Step 3.2: Get the current reward accumulated 
            curr_reward = self.get_current_reward()
            print("\t\tCurrent Reward: " + str(curr_reward))
            if int(curr_reward) == max_rew:
                print("\t\tMax. Reward Already Achieved\n")
                break

            ## Step 3.3: Attempt to execute the desired activity sequence (task)
            for b in best_seq:
                if time.time() - self.start_time > time_thresh:
                    break
                
                ## Step 3.3.1: Determine what workstation we need to go to
                next_ws = list(b.values())[0]
                ws_pos = ws_locs[next_ws]
        
                ## Step 3.3.2: If not at the desired workstation already, attempt navigation
                if self.latest_ws != next_ws:
                    print("\t\tCommencing Navigation...")
                    result = self.move_to_ws(ws_pos)
                    if not result:
                        print("\t\t\t...Navigation Failure (already located at the desired position)")

                ## Step 3.3.3: Execute the desired activity
                do_action = rospy.ServiceProxy('/do_action', ActionReq)
                act = list(b.keys())[0]
                print("\t\tCommencing Action...")
                res = do_action(next_ws, act)
                if not res.success:
                    print("\t\t\t...Action Failure (conditions aren't met)")
                self.latest_ws = next_ws

            ## Step 3.4: Get the current reward (after the activity sequence)
            new_reward = self.get_current_reward()
            print("\t\tNew Reward: " + str(new_reward))

            ## Step 3.5: If the current reward is higher than the previous reward, the task succeeded and we remove it from the lookup table
            if curr_reward != new_reward:
                print('\t' + best_task + ' Succeeded!\n')
                Res.pop(best_task)
            else:
                print('\t' + best_task + ' Failed\n')

        print("...Run Completed")
        print("Total Runtime: {} [sec]".format(time.time()-self.start_time))
        return
    
    ### III. Navigation functions
    def add_goal(self, x, y, yaw=1.0):
        """ Creates a new goal with the MoveBaseGoal constructor """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = yaw
        self.action_client.send_goal(goal)
        
    def move_to_ws(self, pos, threshold=CUBE_EDGE/4):
        """ Moves the robot to the desired location using MoveBase """
        self.action_client.wait_for_server()
        self.add_goal(pos[0], pos[1])
        curr_pos = self.curr_loc
        while np.linalg.norm(np.array([pos[0] - curr_pos[0], pos[1] - curr_pos[1]])) >= threshold:
            wait = self.action_client.wait_for_result()
            curr_pos = self.curr_loc
            if not wait:
                continue
        return self.action_client.get_result()

    ### IV. Calculations needed for determining the next best task
    def calc_euclidean_distance(self, point1, point2):
        """ Calculates Euclidean distance between a pair of 2D points """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def calc_ws_plans(self, locs):
        """ Retrieves motion plans between each pair of workstations """
        ws_plans = {}
        for ws1 in locs:
            ws_plans[ws1] = {}
            for ws2 in locs:
                if ws1 != ws2:
                    ws_plans[ws1][ws2] = self.calc_motion_plan(locs[ws1], locs[ws2])
                else:
                    ws_plans[ws1][ws2] = None
        return ws_plans

    def calc_motion_plan(self, point1, point2):
        """ Creates motion plans between a pair of points (without executing it) """
        start = PoseStamped()
        start.header.seq = 0
        start.header.frame_id = "map"
        start.header.stamp = rospy.Time(0)
        start.pose.position.x = point1[0]
        start.pose.position.y = point1[1]

        goal = PoseStamped()
        goal.header.seq = 0
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time(0)
        goal.pose.position.x = point2[0]
        goal.pose.position.y = point2[1]

        get_plan = rospy.ServiceProxy('/move_base/make_plan', nav_msgs.srv.GetPlan)
        req = nav_msgs.srv.GetPlan()
        req.start = start
        req.goal = goal
        req.tolerance = 1.0 #.5
        resp = get_plan(req.start, req.goal, req.tolerance)

        return resp.plan.poses
    
    def calc_cost(self, plan, velocity):
        """ Calculates the cost of a motion plan, given the (max.) velocity along the path """
        M = 0; T= 0; i = 0
        for p in range(len(plan)-1):
            x1 = plan[p].pose.position.x
            y1 = plan[p].pose.position.y
            pos1 = np.array([x1,y1])

            x2 = plan[p+1].pose.position.x
            y2 = plan[p+1].pose.position.y
            pos2 = np.array([x2,y2])
            
            distance = self.calc_euclidean_distance(pos1, pos2)
            T += distance / velocity

            if p == 0:
                i += np.floor(T)
            if T > i:
                idx_x, idx_y = self.cmu.position_to_map(pos2)
                M += self.cmu.cost_map[int(idx_x)][int(idx_y)]
                i += 1
            pos1 = pos2
        M = 0 # this should be the case, but for some reason we get weird results from the costmap
        return T + M, T
    
    def calc_possible_rewards(self, ws, tasks):
        """ Creates the data structures containing the reward and activity sequence information, for use in other custom-defined methods """
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

    def calc_cost_between_ws(self, ws1, ws2, velocity):
        """ Calculates the cost of travelling between two workstations, given the robot's (max.) velocity along the path """
        plan = self.ws_plans[ws1][ws2]
        if plan is not None:
            return self.calc_cost(plan, velocity)
        else:
            return 0, 0

    def calc_sequence_cost(self, seq, acts_time):
        """ Calculates the total cost of a given activity sequence """
        tot_cost = acts_time
        tot_time = 0
        for i in range(len(seq)-1):
            action = list(seq[i+1].keys())[0]
            if action[:2] == 'PL':
                velocity = MAX_LINEAR_VELOCITY_HOLDING
            else:
                velocity = MAX_LINEAR_VELOCITY_FREE
            res = self.calc_cost_between_ws(list(seq[i].values())[0], list(seq[i+1].values())[0], velocity)
            tot_cost += res[0] + acts_time
            tot_time += res[1] + acts_time
        return tot_cost, tot_time
        
    def calc_base(self, acts, rews, tasks, acts_time):
        """ This function creates a lookup table characterizing every legal sequence of activities possible for each given task """
        Res = {}
        for a in acts:
            Res[a] = []
            activities = [list(d.keys())[0] for d in acts[a]]

            ## Step 1: Get valid sequences for each activity
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
            
            ## Step 2: Get cost for each activity sequence
            for l in last:
                full = []; rev = []
                for i in range(len(l)):
                    seq = []
                    if i > 0:
                        if 'PL-'+obj in tasks[l[i]]:
                            seq.append('PL-'+obj)
                    seq.append(activities[i])
                    if i < len(l)-1:
                        pickup_found = False
                        for j in tasks[l[i]]:
                            if j[:2] == 'PU':
                                obj = j[-1]
                                if 'PL-'+obj in tasks[l[i+1]]:
                                    seq.append(j)
                                    pickup_found = True
                                    break
                    if not pickup_found:
                        break
                    else:
                        full.append({l[i]:seq})
                        for j in range(len(seq)):
                            rev.append({seq[j]:l[i]})
                if pickup_found:
                    res = self.calc_sequence_cost(rev, acts_time)
                    Res[a].append({'Sequence':rev, 'Cost':res[0], 'Reward':rews[a], 'Time':res[1]})
        return Res
    
    ### V. Determination of the next best task
    def get_best_task(self, current_pos, locs, time_thresh, Res):
        """ Update the lookup table given the robot's current position, then determine which activity sequence is optimal """
        best_M = -np.inf # M = R - C
        best_seq = None; best_task = None
        for w in locs:
            plan = self.calc_motion_plan(current_pos, locs[w])
            cost, time = self.calc_cost(plan, velocity=MAX_LINEAR_VELOCITY_FREE)
            for t in Res:
                for s in range(len(Res[t])):
                    if w == list(Res[t][s]['Sequence'][0].values())[0]:
                        Res[t][s]['Cost'] += cost
                        Res[t][s]['Time'] += time
                        # Check if this activity sequence is better than the best one so far
                        if (Res[t][s]['Reward'] - Res[t][s]['Cost'] > best_M) and (Res[t][s]['Time'] <= time_thresh):
                            best_M = Res[t][s]['Reward'] - Res[t][s]['Cost']
                            best_seq = Res[t][s]['Sequence']
                            best_task = t
        return best_task, best_seq

    ### VI. Visualization of the task
    def create_plan_marker(self, loc1, loc2):
        """ Given two points, creates a Marker message containing a yellow arrow pointing from loc1 to loc2 """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = 0 # arrow
        marker.action = 0 # add the marker
        marker.pose.position.z = 0.5
        marker.scale.x = 0.1; marker.scale.y = 0.25; marker.scale.z = 0.25
        marker.color.a = 1.0; marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0

        pt1 = Point()
        pt1.x = loc1[0]; pt1.y = loc1[1]
        marker.points.append(pt1)

        pt2 = Point()
        pt2.x = loc2[0]; pt2.y = loc2[1]
        marker.points.append(pt2)
        return marker

    def create_plan_marker_array(self, curr_pos, seq, locs):
        """ Given a plan for an activity sequence, creates a Marker arrow message for every pair of consecutive (different) workstations in the plan,
        then stores them all in a MarkerArray message for RViZ visualization """
        markerArray = MarkerArray()
        loc1 = curr_pos; id = 0
        for a in range(len(seq)):
            loc2 = locs[list(seq[a].values())[0]]
            if self.calc_euclidean_distance(loc1, loc2) > 10**-3:
                marker = self.create_plan_marker(loc1, loc2)
                marker.id = id
                markerArray.markers.append(marker)
                id += 1
            loc1 = loc2
        return markerArray
    
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
        
    def update_affordance_center(self, new_center):
        self.affordance_center = new_center

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