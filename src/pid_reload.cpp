#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/PoseStamped.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

const std::string POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer/pose";
const double ERR_EPSILON = 0.1;
const double GOAL_X = -3.0;
const double GOAL_Y = -3.0;

class PIDController {
public:
    PIDController(double kp_, double ki_, double kd_)
        : kp(kp_), ki(ki_), kd(kd_), integral(0), last_error(0), last_time(ros::Time::now()) {}

    void reset() {
        integral = 0;
        last_error = 0;
        last_time = ros::Time::now();
    }

    double update(double error) {
        ros::Time current_time = ros::Time::now();
        double dt = (current_time - last_time).toSec();
        double de = error - last_error;

        if (dt <= 0) dt = 1e-6;

        integral += error * dt;
        double derivative = de / dt;

        double output = kp * error + ki * integral + kd * derivative;

        last_error = error;
        last_time = current_time;

        return output;
    }

private:
    double kp, ki, kd;
    double integral, last_error;
    ros::Time last_time;
};

class JetRacerController {
    private:
        PIDController heading_pid;
        PIDController distance_pid;
        ros::Publisher steering_pub;
        ros::Publisher throttle_pub;
        ros::Subscriber pose_sub;

        double angleWrap(double angle) {
            while (angle > M_PI) angle -= 2 * M_PI;
            while (angle < -M_PI) angle += 2 * M_PI;
            return angle;
        }

        void eulerFromQuaternion(double x, double y, double z, double w, double& roll, double& pitch, double& yaw) {
            // roll (x-axis rotation)
            double t0 = +2.0 * (w * x + y * z);
            double t1 = +1.0 - 2.0 * (x * x + y * y);
            roll = atan2(t0, t1);

            // pitch (y-axis rotation)
            double t2 = +2.0 * (w * y - z * x);
            t2 = t2 > +1.0 ? +1.0 : t2;
            t2 = t2 < -1.0 ? -1.0 : t2;
            pitch = asin(t2);

            // yaw (z-axis rotation)
            double t3 = +2.0 * (w * z + x * y);
            double t4 = +1.0 - 2.0 * (y * y + z * z);
            yaw = atan2(t3, t4);
        }
    public:
        JetRacerController(ros::NodeHandle& nh)
            : heading_pid(1.0, 0.0, 0.65), distance_pid(0.5, 0.0, 0.70) {
            steering_pub = nh.advertise<std_msgs::Float32>("/jetracer/steering", 1);
            throttle_pub = nh.advertise<std_msgs::Float32>("/jetracer/throttle", 1);
            pose_sub = nh.subscribe(POSE_TOPIC, 1, &JetRacerController::poseCallback, this);
        }

        void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
            double x = msg->pose.position.x;
            double y = msg->pose.position.y;
            ROS_INFO("Position: %f, %f", x, y);

            double qx = msg->pose.orientation.x;
            double qy = msg->pose.orientation.y;
            double qz = msg->pose.orientation.z;
            double qw = msg->pose.orientation.w;

            double roll, pitch, yaw;
            eulerFromQuaternion(qx, qy, qz, qw, roll, pitch, yaw);

            double dx = GOAL_X - x;
            double dy = GOAL_Y - y;
            double goal_dist = hypot(dx, dy);
            double goal_heading = atan2(dy, dx);
            double heading_error = angleWrap(goal_heading - yaw);

            // PID control
            double steering = heading_pid.update(heading_error);
            double throttle = distance_pid.update(goal_dist);

            // Limit values
            steering = std::max(std::min(steering, 1.0), -1.0);
            throttle = (goal_dist > ERR_EPSILON) ? std::max(std::min(throttle, 0.2), 0.0) : 0.0;

            std_msgs::Float32 steering_msg;
            std_msgs::Float32 throttle_msg;
            steering_msg.data = steering;
            throttle_msg.data = throttle;

            steering_pub.publish(steering_msg);
            throttle_pub.publish(throttle_msg);
        }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "jetracer_pid_controller");
    ros::NodeHandle nh;
    JetRacerController controller(nh);
    ros::spin();
    return 0;
}
