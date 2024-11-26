import numpy as np
import math
import rospy
# Gym
import gymnasium as gym
import gymnasium.spaces as spaces
# from gym.envs.registration import register
# ROS msgs
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Twist, Pose
# Gazebo interfaces
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import PointCloud2, Image, JointState, CompressedImage
import cv2
import torchvision.models as models
# ROS packages required
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import sys
from gymnasium.envs.registration import register
import time
from squaternion import Quaternion
from std_srvs.srv import Empty
from openai_ros import robot_gazebo_env
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_quaternion, quaternion_from_euler, euler_matrix, quaternion_matrix
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Twist, Pose
from squaternion import Quaternion
from shapely.geometry import Point, Polygon, LineString
from utils import linear_to_body, angular_to_body,transformed_nose_position,imgmsg_to_cv2
import torchvision.transforms as transforms
from std_msgs.msg import Int32

min_linear_x = -1
max_linear_x = 1
max_angular_z = -1
min_angular_z = 1

try:
    register(
        id="auto_docking_v0",
        entry_point="task_env:AutoDockingEnv",
        max_episode_steps=200
    )
    print("Environment auto_docking_v0 registered successfully.")
except gym.error.Error as e:
    print(f"Failed to register environment auto_docking_v0: {e}")



class VrxAutoDockingEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):

        self.controllers_list = []
        self.nose_in_body = np.array([2.65, 0, 0]) # coordinates of auv nose tip in body frame

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(VrxAutoDockingEnv, self).__init__(controllers_list=self.controllers_list,
                                          robot_name_space=self.robot_name_space,
                                          reset_controls=False,
                                          start_init_physics_parameters=False,
                                          reset_world_or_sim="WORLD")
        '''setting this to "WORLD" instead of "SIM" causes the init pose of the AUV to 
        get overwritten by it's spawning pose hence we then cannot set random initial position
        of the AUV'''

        self.gazebo.unpauseSim()
        self._check_all_systems_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._usv_data_callback)

        self.set_usv_init_pose = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        self._cmd_drive_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Checking that the publishers are connected to their respective topics
        self._check_pub_connection()

        self.gazebo.pauseSim()



    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("checking all systems ready")
        self._check_all_sensors_ready()
        rospy.logdebug("all systems ready")
        return True

    # DeeplengEnv virtual methods
    # ----------------------------
    def _check_all_sensors_ready(self):
        rospy.logdebug("Checking sensors ")
        self._check_usv_ready()
        rospy.logdebug("Sensors ready")

    def _check_usv_ready(self):
        self.usv_data = None
        self.img_data = None
        rospy.logdebug("Waiting for /gazebo/model_states to be ready")
        while self.usv_data is None and not rospy.is_shutdown():
            try:
                self.usv_data = rospy.wait_for_message("/gazebo/model_states", ModelStates)
                rospy.logdebug("/gazebo/model_states ready")

            except:
                rospy.logerr("/gazebo/model_states not ready yet, retrying for getting auv pose")
        return self.usv_data

    def _usv_data_callback(self, data):

        self.usv_data = data


    def _check_pub_connection(self):

        rate = rospy.Rate(10)  # 10hz

        while self.set_usv_init_pose.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to set_usv_init_pose yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass

        while self._cmd_drive_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no susbribers to cmd_vel yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("All Publishers READY")


    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # def _is_truncated(self, observations):
    #     raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def set_usv_pose(self, x, y, z, yaw, time_sleep):
        """
         It will set the initial pose the deepleng.
        """
        # get orientation
        state_msg = ModelState()
        state_msg.model_name = "wamv"
        quaternion = Quaternion.from_euler(0, 0, yaw)
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = quaternion.x
        state_msg.pose.orientation.y = quaternion.y
        state_msg.pose.orientation.z = quaternion.z
        state_msg.pose.orientation.w = quaternion.w

        # print("Setting auv pose to {}".format((x, y, z)))
        # publish pose
        self.set_usv_init_pose.publish(state_msg)
        self.wait_time_for_execute_movement(time_sleep)


    def wait_time_for_execute_movement(self, time_sleep):
        """
        Because this Wamv position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need
        to evaluate the diference in position and speed on the local reference.
        """
        time.sleep(time_sleep)

    def modelstate2numpy(self, data, mode='pose'):
        # todo: update to return quaternion orientation instead of euler
        if mode.lower() == 'pose':
            auv_pose = data.pose[17]

            auv_pose = np.array([auv_pose.position.x,
                                 auv_pose.position.y,
                                 auv_pose.position.z,
                                 auv_pose.orientation.x,
                                 auv_pose.orientation.y,
                                 auv_pose.orientation.z,
                                 auv_pose.orientation.w])
            return auv_pose

        if mode.lower() == 'vel':
            auv_vel = data.twist[17]
            auv_vel = np.array([auv_vel.linear.x,
                                auv_vel.linear.y,
                                auv_vel.linear.z,
                                auv_vel.angular.x,
                                auv_vel.angular.y,
                                auv_vel.angular.z])
            return auv_vel
    def get_usv_pose(self, data):
        """
        returns the auv_pose at the nose tip of the auv
        """
        usv_pose = self.modelstate2numpy(data, 'pose')

        roll, pitch, yaw = euler_from_quaternion(usv_pose[3:])
        # print(np.degrees([yaw, pitch, roll]))


        nose_position = usv_pose[:3] + transformed_nose_position(usv_pose[3:],
                                                                 self.nose_in_body)


        pose_out = np.hstack((nose_position, np.array([roll, pitch, yaw])))
        # this should be moved as a parameter to the function
        indices_to_remove = [2, 3, 4]
        pose_out = np.delete(pose_out, indices_to_remove)


        # return x,y,raw
        return pose_out


    def get_usv_velocity(self, data, frame="body"):
        """
        returns the linear and angular auv_velocity in either world or body frame
        """
        auv_vel_world = self.modelstate2numpy(data, 'vel')
        auv_pose = self.modelstate2numpy(data, 'pose')
        # print("RobotEnv::auv vel world:", auv_vel_world)

        if frame == 'world':
            return auv_vel_world

        if frame == 'body':
            orientation = auv_pose[3:]
            #roll, pitch, yaw = euler_from_quaternion(auv_pose[3:])

            linear_vel = linear_to_body(auv_vel_world[:3],
                                        orientation)

            angular_vel = angular_to_body(auv_vel_world[3:],
                                          orientation)


            return np.hstack((linear_vel, angular_vel))

class AutoDockingEnv(VrxAutoDockingEnv):
    def __init__(self):
        """
        Make Deepleng learn how to dock to the docking station from a
        starting point within a given range around the docking station.
        """


        self.action_space = spaces.Box(low = np.array([-1, -1]),
                                       high=np.array([1, 1]),dtype=np.float64)

        self.reward_range = (-np.inf, np.inf)
        self.num_episodes = 0
        self.data_dict = dict()

        # low_img = np.zeros(12288)
        # low_ = np.array([0, -1, -30, -30, -3.14])
        # low = np.concatenate([low_, low_img])
        # high_img = np.full(12288, 255)
        # high_ = np.array([1, 1, 30, 30, 3.14])
        # high = np.concatenate([high_, high_img])

        low_img = np.zeros(12288)
        low_ = np.array([0, -1, -30, -30, -3.14])
        low_enc = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        low = np.concatenate([low_enc, low_, low_img])
        high_img = np.full(12288, 255)
        high_ = np.array([1, 1, 30, 30, 3.14])
        high_enc = -low_enc
        high = np.concatenate([high_enc, high_, high_img])

        self.observation_space = spaces.Box(low, high)

        self.target_pose = self.set_target_pose()
        self.num = 0
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(AutoDockingEnv, self).__init__()


    def set_target_pose(self):
        # Professor: Provide the desired goal pose. yes = using a local frame: either robot or camera
        target = Pose()
        target.position.x = -555
        target.position.y = 217.85
        target.position.z = 0

        quaternion = quaternion_from_euler(
                        0,
                        0,
                        3.14,
                        )

        target.orientation.x = quaternion[0]
        target.orientation.y = quaternion[1]
        target.orientation.z = quaternion[2]
        target.orientation.w = quaternion[3]

        return target


    def _set_init_pose(self):
        """

        """
        # x = -540
        # y = 215
        x = np.random.uniform(-530, -525)
        y = np.random.uniform(210,225)
        print(x,y)
        z = 0
        yaw = -np.pi
        self.set_usv_pose(x, y, z, yaw, time_sleep=0.5)
        rospy.loginfo("Using provided initial poses for respawning")

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        We get the initial pose to measure the distance from the desired point.
        All the data from the simulator is pre-processed into numpy arrays
        """
        self.ep_reward = list()
        self.num_episodes += 1
        self.he=0
        #init_pose = self.get_usv_pose(self.usv_data)



    def _set_action(self, action):
        # action comes from the DRL algo we are using
        """
        """

        cmd = Twist()

        cmd.linear.x = (action[0] + 1)/3
        cmd.linear.y = 0
        cmd.angular.z = action[1]/4
        #print(lin_vel*0.3, ang_vel)
        usv_vel = self.get_usv_velocity(self.usv_data)
        self.previous_action_1 = [usv_vel[0], usv_vel[5]]

        self._cmd_drive_pub.publish(cmd)

        self.wait_time_for_execute_movement(0.2)



    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        Currently it includes only the vehicle pose, we need to check how
        to include the camera data also.
        :return: observation
        """
        # print("Getting observation")

        # todo: could modify to get both the pose and the camera data(in deepleng_env)
        # getting pose, velocity and thrust observations at the same time

        img_raw = rospy.wait_for_message("/wamv/sensors/cameras/front_left_camera/image_raw", Image)
        img_data = imgmsg_to_cv2(img_raw)
        img_data = img_data[185:, :]
        # cv2.imwrite(f"pic/{self.num}.jpg",img_data)
        # self.num+=1
        img_data = cv2.resize(img_data, (64, 64), interpolation=cv2.INTER_AREA)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        img_data = img_data.flatten()

        usv_position = self.get_usv_pose(self.usv_data)

        usv_vel = self.get_usv_velocity(self.usv_data)

        self.last_action = [usv_vel[0], usv_vel[5]]

        position_error = [self.target_pose.position.x - usv_position[0],
                          self.target_pose.position.y - usv_position[1],

                          ]

        current_yaw = usv_position[2]



        heading_error = self.calculate_heading_error(current_yaw, position_error)  # obs[8]

        none = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # 0 1 + 2 3 + 4
        observation = np.round(np.hstack((self.last_action, position_error, [heading_error],none, img_data)), 2)
        #print(position_error, heading_error, self.last_action)
        return observation

    def _is_collision(self):
        if self.previous_action_1[0] - self.last_action[0] > 0.5:
            return True
        else:
            return False

    def _is_done(self, observations):
        """
        """

        # here we use the actual pose of the AUV to check if it is within workspace limits
        is_within_limit = self.is_inside_workspace(observations)
        is_collision = self._is_collision()

        position_error = np.array(observations[2:4])
        distance = self.distance_to_target(position_error)



        done = not is_within_limit or is_collision or distance< 1.2


        return done


    def distance_to_target(self, error):
        return np.linalg.norm(error)




    def is_inside_workspace(self, obs):
        """
        Check if the deepleng is inside the defined workspace limits,
        returns true if it is inside the workspace, and false otherwise
        """
        position_error = np.array(obs[2:4])
        distance = self.distance_to_target(position_error)

        if distance < 40:
            return True
        return False



    def calculate_heading_error(self, yaw, position_error):

        dx = position_error[0]
        dy = position_error[1]

        # Calculate the angle to the target in the global frame
        target_angle_global = math.atan2(dy, dx)

        # Calculate the relative angle from the robot's current yaw angle to the target
        relative_yaw_to_target = target_angle_global - yaw

        # Normalize the angle to be between -pi and pi
        heading_error = math.atan2(math.sin(relative_yaw_to_target), math.cos(relative_yaw_to_target))

        return heading_error


    def _compute_reward(self, observations, done):
        """
        We Base the rewards in if its done or not and we base it on
        if the distance to the desired point has increased or not
        :return:
        """
        heading_error = np.abs(observations[4])


        position_error = np.array(observations[2:4])
        euclidean_distance = np.linalg.norm(position_error)

        distance_reward = -euclidean_distance

        path_reward = -heading_error



        reward = distance_reward*1.2 + path_reward
        if not done:
            if euclidean_distance < 20 and euclidean_distance > 1.2:
                return reward
        if done:
            if euclidean_distance < 1.2:
                reward = 2000 + reward
            elif self._is_collision():
                reward = -100 + reward
            elif not self.is_inside_workspace(observations):
                reward = -2000 + reward
            else:
                reward = -1000 + reward
        return float(reward)



if __name__ == '__main__':
    rospy.init_node('auto_docking', anonymous=True, log_level=rospy.DEBUG)
    env = gym.make("auto_docking_v0")
    rospy.spin()
