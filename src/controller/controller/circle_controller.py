import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import OffboardControlMode

import numpy as np
import math
import csv
import os
import datetime
import sys

class Controller(Node):

    def __init__(self):
        super().__init__('controller_node')
        
        # --- QoS Profile Setup ---
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Offboard Control Mode ---
        self.offboard_control_mode_params = {
            "position": False, "velocity": False, "acceleration": False,
            "attitude": False, "body_rate": False, "thrust_and_torque": False,
            "direct_actuator": True, 
        }

        # --- Messages Initialization ---
        self.vehicle_status = VehicleStatus()
        self.offboard_control_mode = OffboardControlMode()
        self.actuator_motors = ActuatorMotors()

        # --- State Variables ---
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.euler = np.zeros(3)
        self.body_rate = np.zeros(3)
        self.acc_prev = np.zeros(3)
        self.jerk = np.zeros(3)

        # --- UAV Parameters ---
        self.mass = 2.0
        self.arm_length = 0.174 
        self.moment_constant = 0.016 
        self.thrust_constant = 8.54858e-06 
        self.max_rotor_speed = 1000.0
        self.gravity = 9.81
        
        self.Ixx = 0.0216667; self.Iyy = 0.0216667; self.Izz = 0.040

        # --- Control Allocation ---
        self.torques_and_thrust_to_rotor_velocities = np.zeros((4, 4))
        self.compute_control_allocation_matrix()

        # --- FBL Controller States ---
        self.u0 = self.mass * self.gravity 
        self.u0_dot = 0.0

        # --- Trajectory Parameters ---
        self.dt = 0.01 # 100Hz
        self.mission_complete = False
        self.flight_duration = 20.0 # 20 Seconds duration
        
        self.pos_des = np.zeros(3)
        self.vel_des = np.zeros(3)
        self.acc_des = np.zeros(3)
        
        self.is_trajectory_initialized = False
        self.traj_start_time = 0.0
        
        # --- Logging Setup (CIRCLE) ---
        self.log_dir = 'flight_logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Filename distinguishes CIRCLE
        self.csv_filename = os.path.join(self.log_dir, f'log_circle_{timestamp}.csv')
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        header = [
            'Timestamp', 
            'Pos_X', 'Pos_Y', 'Pos_Z', 
            'Vel_X', 'Vel_Y', 'Vel_Z',
            'Roll', 'Pitch', 'Yaw',
            'BodyRate_X', 'BodyRate_Y', 'BodyRate_Z',
            'Motor_1', 'Motor_2', 'Motor_3', 'Motor_4',
            'Des_Pos_X', 'Des_Pos_Y', 'Des_Pos_Z',
            'Des_Vel_X', 'Des_Vel_Y', 'Des_Vel_Z',
            'Des_Acc_X', 'Des_Acc_Y', 'Des_Acc_Z'
        ]
        self.csv_writer.writerow(header)
        self.get_logger().info(f"Logging started: {self.csv_filename}")

        # --- Timer & Subscribers/Publishers ---
        self.timer = self.create_timer(self.dt, self.cmdloop_callback)

        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._vehicle_status_callback, self.qos_profile)
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._vehicle_local_position_callback, self.qos_profile)
        self.vehicle_attitude_subscriber = self.create_subscription(
            VehicleAttitude, "/fmu/out/vehicle_attitude", self._vehicle_attitude_callback, self.qos_profile)
        self.vehicle_angular_velocity_subscriber = self.create_subscription(
            VehicleAngularVelocity, "/fmu/out/vehicle_angular_velocity", self._vehicle_angular_velocity_callback, self.qos_profile)

        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", self.qos_profile)
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", self.qos_profile)
        self.actuator_motors_publisher = self.create_publisher(
            ActuatorMotors, "/fmu/in/actuator_motors", self.qos_profile)

    def compute_control_allocation_matrix(self):
        geometry_matrix = np.array([
            [-1,  1,  1, -1], 
            [-1,  1, -1,  1], 
            [-1,   -1,    1,    1],   
            [ 1,    1,    1,    1]    
        ])
        scaling = np.diag([
            self.thrust_constant * self.arm_length, 
            self.thrust_constant * self.arm_length, 
            self.moment_constant,                   
            self.thrust_constant                    
        ])
        rotor_vel_to_wrench = scaling @ geometry_matrix
        self.wrench_to_rotor_velocities = np.linalg.pinv(rotor_vel_to_wrench)
        self.get_logger().info("Control Allocation Matrix Computed.")

    def cmdloop_callback(self):
        self.publish_offboard_control_mode()

        if self.vehicle_status.timestamp == 0:
            self.get_logger().info("Waiting for vehicle status...", throttle_duration_sec=1.0)
            return

        if self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info("Waiting for Offboard mode...", throttle_duration_sec=1.0)
            self.pos_des = self.pos.copy(); self.vel_des = np.zeros(3); self.acc_des = np.zeros(3)
            self.is_trajectory_initialized = False; self.u0 = self.mass * self.gravity; self.u0_dot = 0.0
            return

        if self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.get_logger().info("Waiting for Arming...", throttle_duration_sec=1.0)
            return

        self.update_trajectory()

        if self.mission_complete:
            self.get_logger().info("20 Seconds Reached. Closing log and shutting down.")
            self.csv_file.close()
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            sys.exit(0) 

        u_physical = self.fbl_controller()
        throttles = self.px4_inverse(u_physical)
        
        self.publish_actuator_motors(throttles)
        self.log_data(throttles)
        self.print_debug(throttles)

    def log_data(self, throttles):
        now = self.get_clock().now().nanoseconds / 1e9
        row = [
            now,
            self.pos[0], self.pos[1], self.pos[2],
            self.vel[0], self.vel[1], self.vel[2],
            self.euler[0], self.euler[1], self.euler[2],
            self.body_rate[0], self.body_rate[1], self.body_rate[2],
            throttles[0], throttles[1], throttles[2], throttles[3],
            self.pos_des[0], self.pos_des[1], self.pos_des[2],
            self.vel_des[0], self.vel_des[1], self.vel_des[2],
            self.acc_des[0], self.acc_des[1], self.acc_des[2]
        ]
        self.csv_writer.writerow(row)

    def fbl_controller(self):
        x, y, z = self.pos; vx, vy, vz = self.vel; ax, ay, az = self.acc
        phi, theta, psi = self.euler
        self.jerk = (self.acc - self.acc_prev) / self.dt; jx, jy, jz = self.jerk
        self.acc_prev = self.acc.copy()
        p, q, r = self.body_rate
        cos_theta = math.cos(theta) if abs(math.cos(theta)) > 0.01 else 0.01
        tt = math.tan(theta)
        phi_dot = p + q * math.sin(phi) * tt + r * math.cos(phi) * tt
        theta_dot = q * math.cos(phi) - r * math.sin(phi)
        psi_dot = (q * math.sin(phi) + r * math.cos(phi)) / cos_theta

        k0, k1, k2, k3 = 10.0, 42.49, 40.27, 13.43; k_psi0, k_psi1 = 10.0, 10.0
        v1 = 0.0 - k3*jx - k2*(ax - self.acc_des[0]) - k1*(vx - self.vel_des[0]) - k0*(x - self.pos_des[0])
        v2 = 0.0 - k3*jy - k2*(ay - self.acc_des[1]) - k1*(vy - self.vel_des[1]) - k0*(y - self.pos_des[1])
        v3 = 0.0 - k3*jz - k2*(az - self.acc_des[2]) - k1*(vz - self.vel_des[2]) - k0*(z - self.pos_des[2])

        u0 = self.u0
        if u0 < 1.0: u0 = 1.0
        c_phi, s_phi = math.cos(phi), math.sin(phi); c_theta, s_theta = math.cos(theta), math.sin(theta)
        A = np.array([
            [c_phi*s_theta,  -(1/self.Ixx)*s_phi*s_theta*u0,   (1/self.Iyy)*c_phi*c_theta*u0],
            [-s_phi,         -(1/self.Ixx)*c_phi*u0,            0.0],
            [c_phi*c_theta,  -(1/self.Ixx)*s_phi*c_theta*u0,  -(1/self.Iyy)*c_phi*s_theta*u0]
        ])
        term1 = (phi_dot**2 + theta_dot**2)*c_phi*s_theta*u0 + 2*phi_dot*theta_dot*s_phi*c_theta*u0 + 2*phi_dot*s_phi*s_theta*self.u0_dot + 2*theta_dot*c_phi*c_theta*self.u0_dot
        term2 = 2*phi_dot*c_phi*self.u0_dot - (phi_dot**2)*s_phi*u0
        term3 = (phi_dot**2 + theta_dot**2)*c_phi*c_theta*u0 - 2*phi_dot*theta_dot*s_phi*s_theta*u0 + 2*phi_dot*s_phi*c_theta*self.u0_dot + 2*theta_dot*c_phi*s_theta*self.u0_dot
        f_vec = np.array([term1, term2, term3]); mv_vec = np.array([self.mass * v1, self.mass * v2, self.mass * v3]); v_star = mv_vec + f_vec
        
        try: u_sol = np.linalg.solve(A, v_star)
        except np.linalg.LinAlgError: u_sol = np.zeros(3)
        u0_ddot_cmd, u1_cmd, u2_cmd = u_sol
        u3_cmd = self.Izz * (0.0 - k_psi1*(0.0 - psi_dot) - k_psi0*(0.0 - psi))
        
        max_thrust = 40.0 
        next_u0 = self.u0 + (self.u0_dot + u0_ddot_cmd * self.dt) * self.dt
        if next_u0.any() > max_thrust and u0_ddot_cmd > 0: self.u0_dot = 0.0
        elif next_u0.any() < 1.0 and u0_ddot_cmd < 0: self.u0_dot = 0.0
        else: self.u0_dot += u0_ddot_cmd * self.dt; self.u0 += self.u0_dot * self.dt
        self.u0 = np.clip(self.u0, 1.0, max_thrust)
        return np.array([self.u0, u1_cmd, u2_cmd, u3_cmd])

    def px4_inverse(self, wrench):
        target_wrench = np.array([wrench[1], wrench[2], wrench[3], wrench[0]])
        omega_sq = self.wrench_to_rotor_velocities @ target_wrench
        omega_sq = np.maximum(omega_sq, 0.0); omega = np.sqrt(omega_sq)
        throttles = omega / self.max_rotor_speed
        return np.clip(throttles, 0.0, 1.0)

    def publish_actuator_motors(self, throttles):
        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.reversible_flags = 0
        control_signal = [float('nan')] * 12
        control_signal[0] = float(throttles[0]); control_signal[1] = float(throttles[1])
        control_signal[2] = float(throttles[2]); control_signal[3] = float(throttles[3])
        msg.control = control_signal
        self.actuator_motors_publisher.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False; msg.velocity = False; msg.acceleration = False
        msg.attitude = False; msg.body_rate = False; msg.thrust_and_torque = False; msg.direct_actuator = True 
        self.offboard_control_mode_publisher.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = param1; msg.param2 = param2; msg.command = command
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1; msg.from_external = True
        self.vehicle_command_publisher.publish(msg)

    def update_trajectory(self):
        # --- Circle Trajectory (Radius 1.5m, Speed 0.5 rad/s) ---
        radius = 1.5
        omega = 0.5
        altitude = 2.0
        
        if not self.is_trajectory_initialized:
            self.traj_start_time = self.get_clock().now().nanoseconds / 1e9
            self.is_trajectory_initialized = True

        now = self.get_clock().now().nanoseconds / 1e9
        elapsed = now - self.traj_start_time

        # Stop after 20 seconds
        if elapsed >= self.flight_duration:
            self.mission_complete = True
            # Hold last position
            return

        # Parametric Equations for Circle
        self.pos_des = np.array([
            radius * math.cos(omega * elapsed),
            radius * math.sin(omega * elapsed),
            altitude
        ])
        
        self.vel_des = np.array([
            -radius * omega * math.sin(omega * elapsed),
            radius * omega * math.cos(omega * elapsed),
            0.0
        ])
        
        self.acc_des = np.array([
            -radius * (omega**2) * math.cos(omega * elapsed),
            -radius * (omega**2) * math.sin(omega * elapsed),
            0.0
        ])

    def print_debug(self, m):
        err = self.pos_des - self.pos
        self.get_logger().info(
            f"\nCurrently tracking: CIRCLE Trajectory ({self.flight_duration}s limit)\n"
            f"Current position     : [{self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f}]\n"
            f"Desired target pos   : [{self.pos_des[0]:.3f}, {self.pos_des[1]:.3f}, {self.pos_des[2]:.3f}]\n"
            f"Position error       : [{err[0]:.3f}, {err[1]:.3f}, {err[2]:.3f}]\n"
            f"Motor input          : [{m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}, {m[3]:.3f}]",
            throttle_duration_sec=0.5
        )

    def _vehicle_status_callback(self, msg): self.vehicle_status = msg
    def _vehicle_local_position_callback(self, msg):
        self.vehicle_local_position = msg
        self.pos = np.array([msg.y, msg.x, -msg.z])
        self.vel = np.array([msg.vy, msg.vx, -msg.vz])
        self.acc = np.array([msg.ay, msg.ax, -msg.az])
    def _vehicle_attitude_callback(self, msg):
        self.vehicle_attitude = msg
        self.euler = euler_from_quaternion(msg.q)
    def _vehicle_angular_velocity_callback(self, msg):
        self.vehicle_angular_velocity = msg
        self.body_rate = np.array([msg.xyz[0], -msg.xyz[1], -msg.xyz[2]])

def euler_from_quaternion(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    t0 = +2.0 * (w * x + y * z); t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x); t2 = +1.0 if t2 > +1.0 else t2; t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y); t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    yaw_enu = -yaw_z + (math.pi / 2.0)
    if yaw_enu > math.pi: yaw_enu -= 2*math.pi
    if yaw_enu < -math.pi: yaw_enu += 2*math.pi
    return np.array([roll_x, -pitch_y, yaw_enu])

def main(args=None):
    rclpy.init(args=args)
    try: controller = Controller(); rclpy.spin(controller)
    except KeyboardInterrupt: pass
    except SystemExit: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()