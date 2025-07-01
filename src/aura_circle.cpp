#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>

const std::string POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer/pose";
const double ERR_EPSILON = 0.1;
const double RADIUS = 0.75;
const double CENTER[] = {-1.5, 1.5};
const double CHASING_DIST = 0.25;

// gets time in seconds since epoch
double get_current_time() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now().time_since_epoch()).count();
}

class PIDController
{
    private:
        double kp, ki, kd;
        double integral, last_error;
        double last_time;
    public:
        PIDController() {
            this->kp = 0;
            this->ki = 0;
            this->kd = 0;
            integral = 0;
            last_error = 0;
            last_time = get_current_time();
        }

        PIDController(double kp, double ki, double kd) {
            this->kp = kp;
            this->ki = ki;
            this->kd = kd;
            integral = 0;
            last_error = 0;
            last_time = get_current_time();
        }

        void reset() {
            integral = 0;
            last_error = 0;
            last_time = get_current_time();
        }

        double update(double error) {
            double current_time = get_current_time();
            double dt = current_time - last_time;
            double de = error - last_error;

            integral += error * dt;
            double derivative = dt > 0 ? de / dt : 0;

            double output = kp * error + ki * integral + kd * derivative;

            last_error = error;
            last_time = current_time;

            return output;
        }
};

class JetRacerController
{
    private:
        ros::Publisher steering_pub;
        ros::Publisher throttle_pub;
        ros::Subscriber pose_sub;
        double center[2];
        double radius, lookahead_distance;
        PIDController heading_pid;
        PIDController distance_pid;

        double angle_wrap(double angle)
        {
            while (angle > M_PI)
            {
                angle -= 2 * M_PI;
            }
            while (angle < -M_PI)
            {
                angle += 2 * M_PI;
            }
            return angle;
        }

        double yaw_from_quaternion(double x, double y, double z, double w)
        {
            /*
                Convert a quaternion into euler angles (roll, pitch, yaw)
                roll is rotation around x in radians (counterclockwise)
                pitch is rotation around y in radians (counterclockwise)
                yaw is rotation around z in radians (counterclockwise)
            */

            double t0 = 2.0 * (w * x + y * z);
            double t1 = 1.0 - 2.0 * (x * x + y * y);
            double roll_x = std::atan2(t0, t1);
        
            double t2 = 2.0 * (w * y - z * x);
            t2 = t2 > 1.0 ? 1.0 : t2;
            t2 = t2 < -1.0 ? -1.0 : t2;
            double pitch_y = std::asin(t2);
        
            double t3 = 2.0 * (w * z + x * y);
            double t4 = 1.0 - 2.0 * (y * y + z * z);
            double yaw_z = std::atan2(t3, t4);
        
            return /*roll_x, pitch_y,*/ yaw_z; // in radians
        }

        void pose_callback(geometry_msgs::PoseStamped::ConstPtr& msg)
        {
            auto x = msg->pose.position.x;
            auto y = msg->pose.position.y;
            auto q = msg->pose.orientation;
            // std::cout << "Postion: (" << x << ", " << y << ")\n";
            double yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w);

            // Vector from center to car
            double dx = x - center[0];
            double dy = y - center[1];
            double r = std::hypot(dx, dy);
            double theta = std::atan2(dy, dx);
            double radial_error = r - radius;
            double target_x, target_y;
        
            if (std::abs(radial_error) > ERR_EPSILON)  // Phase 1: Converge to circle
            {    
                // Drive toward the closest point on the circle
                target_x = center[0] + radius * std::cos(theta);
                target_y = center[1] + radius * std::sin(theta);
            }
            else  // Phase 2: Follow the circle
            {
                double theta_target = theta + lookahead_distance / radius;
                target_x = center[0] + radius * std::cos(theta_target);
                target_y = center[1] + radius * std::sin(theta_target);
            }
        
            // Steer toward the target
            double goal_heading = std::atan2(target_y - y, target_x - x);
            double heading_error = angle_wrap(goal_heading - yaw);
        
            double steering = heading_pid.update(heading_error);
            double throttle = abs(radial_error) < 0.5 ? 0.15 : 0.16; // Slightly faster until you get to the orbit
            steering = std::max(std::min(steering, 1.0), -1.0);

	    std_msgs::Float32 steering_msg;
	    std_msgs::Float32 throttle_msg;
	    steering_msg.data = steering;
	    throttle_msg.data = throttle;
            steering_pub.publish(steering_msg);
            throttle_pub.publish(throttle_msg);
        }
    public:        
        JetRacerController(ros::NodeHandle& nh)
            : heading_pid(0.3, 0.0, 0.02), distance_pid(0.25, 0.0, 0.4)
        {
            steering_pub = nh.advertise<std_msgs::Float32>("/jetracer/steering", 1);
            throttle_pub = nh.advertise<std_msgs::Float32>("/jetracer/throttle", 1);
            pose_sub = nh.subscribe(POSE_TOPIC, 1000, &JetRacerController::pose_callback, this);
            center[0] = CENTER[0];
            center[1] = CENTER[1];
            radius = RADIUS;
            lookahead_distance = CHASING_DIST;
        }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "jetracer_pid_controller");
    ros::NodeHandle nh;
    JetRacerController controller = JetRacerController(nh);
    ros::spin();
}
