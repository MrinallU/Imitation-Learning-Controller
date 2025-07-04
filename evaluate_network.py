#!/usr/bin/env python3
WEIGHTS_PATH = 'imit_policy.pth'   
GOAL_X, GOAL_Y = 6.5,-1       
EPISODES       = 3                 
GOAL_TOL       = 0.7            
MAX_STEPS      = 300              
RATE_HZ        = 2.0              

import math, time, rclpy, torch
from rclpy.node         import Node
from std_srvs.srv       import Empty
from nav_msgs.msg       import Odometry
from geometry_msgs.msg  import Twist
from torch import nn, tensor

class ImitationPolicy(nn.Module):
    def __init__(self, n_actions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,n_actions)
        )
    def forward(self,x):
        return self.net(x)

ACTIONS = [
    (0.0,0.0), (0.22,0.0), (0.0,0.5),
    (0.0,-0.5), (0.22,0.5), (0.22,-0.5)
]

class EvalNode(Node):
    def __init__(self):
        super().__init__('imit_eval_node')
        self.gx, self.gy   = GOAL_X, GOAL_Y
        self.goal_tol      = GOAL_TOL
        self.max_steps     = MAX_STEPS
        self.episodes_left = EPISODES
        self.rate_hz       = RATE_HZ

        self.policy = ImitationPolicy()
        self.policy.load_state_dict(torch.load(WEIGHTS_PATH,
                                               map_location='cpu'))
        self.policy.eval()

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.cx = self.cy = None
        self.step = 0
        self.create_timer(1.0 / self.rate_hz, self.loop)

    def odom_cb(self,msg):
        pos = msg.pose.pose.position
        self.cx, self.cy = pos.x, pos.y

    def loop(self):
        if self.cx is None:                           
            return

        dist = math.hypot(self.gx-self.cx, self.gy-self.cy)
        if dist < self.goal_tol or self.step >= self.max_steps:
            if dist < self.goal_tol: 
                print("GOAL REACHED!!")
            self.stop_and_reset(dist)
            return

        # greedy action
        with torch.no_grad():
            inp = tensor([[self.cx, self.cy, self.gx, self.gy]],
                         dtype=torch.float32)
            a   = self.policy(inp).argmax(1).item()
        lin, ang = ACTIONS[a]
        tw = Twist(); tw.linear.x, tw.angular.z = lin, ang
        self.cmd_pub.publish(tw)
        self.step += 1

    def stop_and_reset(self, dist):
        self.cmd_pub.publish(Twist())
        time.sleep(0.1)
        self.get_logger().info(f"Episode done  d={dist:.2f}  steps={self.step}")

        self.episodes_left -= 1
        if self.episodes_left <= 0:
            rclpy.shutdown(); return

        # reset Gazebo
        reset_node = rclpy.create_node('reset_client')
        client = reset_node.create_client(Empty, '/reset_simulation')
        if client.wait_for_service(timeout_sec=3.0):
            client.call_async(Empty.Request())
            rclpy.spin_once(reset_node, timeout_sec=1.5)
        reset_node.destroy_node()

        self.step = 0
        time.sleep(1.0)           

def main():
    rclpy.init()
    node = EvalNode()         
    try:
        rclpy.spin(node)      
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
