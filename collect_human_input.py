#!/usr/bin/env python3
import sys, termios, tty, select, csv, os, time, argparse, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

ACTIONS = [
    (0.00, 0.00), (0.22, 0.00), (0.00, 0.50),
    (0.00,-0.50), (0.22, 0.50), (0.22,-0.50)
]
ACTION_KEYS = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5}
QUIT_KEY      = 'q'
STEP_DURATION = 0.4      

def read_key():
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r,_,_ = select.select([sys.stdin],[],[], None)
        return sys.stdin.read(1) if r else ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

class StepTeleop(Node):
    def __init__(self, csv_path, gx, gy):
        super().__init__('step_teleop_recorder')
        self.gx, self.gy = gx, gy
        self.cx = self.cy = None
        self.create_subscription(Odometry,'/odom',self.odom_cb,10)
        self.cmd_pub = self.create_publisher(Twist,'/cmd_vel',10)

        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        self.csv = open(csv_path,'a'); self.wr = csv.writer(self.csv)
        self.rows = 0
        self.print_help()

        # wait until first odom
        while rclpy.ok() and self.cx is None:
            rclpy.spin_once(self, timeout_sec=0.1)

    def odom_cb(self,msg):
        self.cx = msg.pose.pose.position.x
        self.cy = msg.pose.pose.position.y

    def run(self):
        while rclpy.ok():
            key = read_key()
            if key == QUIT_KEY: break
            if key not in ACTION_KEYS:
                self.get_logger().info('invalid key'); continue
            self.execute_action(ACTION_KEYS[key])
        rclpy.shutdown()

    def execute_action(self, a:int):
        lin, ang = ACTIONS[a]
        self.cmd_pub.publish(Twist(linear=Twist().linear.__class__(x=lin),
                                   angular=Twist().angular.__class__(z=ang)))

        start = time.time()
        while time.time() - start < STEP_DURATION and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)   

        self.cmd_pub.publish(Twist())                
        rclpy.spin_once(self, timeout_sec=0.0)      
        self.wr.writerow([self.cx, self.cy, self.gx, self.gy, a])
        self.rows += 1
        self.get_logger().info(f"logged {self.rows} samples  action {a}")

    def destroy_node(self):
        super().destroy_node(); self.csv.close()
        print(f"\nSaved {self.rows} samples – bye!")

    def print_help(self):
        print("""\nStep-wise tele-op: press a key to move, robot stops automatically.
 0 idle | 1 fwd | 2 left | 3 right | 4 fwd-left | 5 fwd-right | q quit\n""")


def main():
    csv_path = 'demos.csv'
    goal_x, goal_y = 5, 1
    rclpy.init()
    node = StepTeleop(csv_path, goal_x, goal_y)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
