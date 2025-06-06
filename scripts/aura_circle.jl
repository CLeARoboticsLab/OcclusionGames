#!/usr/bin/env julia

#=
Authors: Jai Nagaraj, Ali Chimoun
The purpose of this script is to have an NVIDIA JetRacer move in a circle using PID control.
This is essentially a translated version of the existing Python script using the RobotOS.jl package.
=#

using RobotOS
@rosimport std_msgs.msg: Float32
@rosimport geometry_msgs.msg: PoseStamped
rostypegen()

using .std_msgs.msg
using .geometry_msgs.msg
using Dates
using LinearAlgebra

const POSE_TOPIC = "vrpn_client_node/JaiAliJetRacer/pose"
const ERR_EPSILON = 0.1
const RADIUS = 0.75
const CENTER = [-1.5, 1.5]
const CHASING_DIST = 0.25

#=
**** PIDCONTROLLER CLASS STUFF ****
Since Julia does not have full OOP, code is manually organized into classes.
=#

# Julia version of the Python PIDController class.
mutable struct PIDController
    kp::Float64
    ki::Float64
    kd::Float64
    integral::Float64
    last_error::Float64
    last_time::DateTime
end

function PIDController(kp, ki, kd)
    now = nowtime()
    PIDController(kp, ki, kd, 0.0, 0.0, now)
end

function reset!(pid::PIDController)
    pid.integral = 0.0
    pid.last_error = 0.0
    pid.last_time = nowtime()
end

function update!(pid::PIDController, error::Float64)
    current_time = nowtime()
    dt = max(1e-6, Dates.value(current_time - pid.last_time) / 1e3)  # seconds
    de = error - pid.last_error

    pid.integral += error * dt
    derivative = de / dt

    output = pid.kp * error + pid.ki * pid.integral + pid.kd * derivative

    pid.last_error = error
    pid.last_time = current_time

    return output
end

#=
**** JETRACERCONTROLLER CLASS STUFF ****
Since Julia does not have full OOP, code is manually organized into classes.
=#

struct JetRacerController
    steering_pub # Publisher to jetracer steering
    throttle_pub # Publisher to jetracer throttle
    heading_pid::PIDController # controls values published to steering
    distance_pid::PIDController # controls values published to throttle
    center::Vector{Float64}
    radius::Float64
    lookahead_distance::Float64
end

function JetRacerController()
    init_node("jetracer_pid_controller")
    steering_pub = Publisher{Float32}("/jetracer/steering", queue_size=1)
    throttle_pub = Publisher{Float32}("/jetracer/throttle", queue_size=1)
    controller = JetRacerController(
        steering_pub,
        throttle_pub,
        PIDController(1.0, 0.0, 0.65),
        PIDController(0.5, 0.0, 0.70),
        CENTER,
        RADIUS,
        CHASING_DIST
    )
    Subscriber(POSE_TOPIC, PoseStamped, msg -> pose_callback(controller, msg))
    return controller
end

function pose_callback(controller::JetRacerController, msg::PoseStamped)
    x = msg.pose.position.x
    y = msg.pose.position.y
    q = msg.pose.orientation
    _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

    dx = x - controller.center[1]
    dy = y - controller.center[2]
    r = hypot(dx, dy)
    theta = atan(dy, dx)
    radial_error = r - controller.radius

    target_x = 0
    target_y = 0

    if abs(radial_error) > ERR_EPSILON  # Phase 1: Converge to circle
        # Drive toward the closest point on the circle
        target_x = controller.center[1] + controller.radius * cos(theta)
        target_y = controller.center[2] + controller.radius * sin(theta)
    else  # Phase 2: Follow the circle
        theta_target = theta + controller.lookahead_distance / controller.radius
        target_x = controller.center[1] + controller.radius * cos(theta_target)
        target_y = controller.center[2] + controller.radius * sin(theta_target)
    end
    
    goal_heading = atan(target_y - y, target_x - x)
    heading_error = angle_wrap(goal_heading - yaw)

    steering = update!(controller.heading_pid, heading_error)
    throttle = update!(controller.distance_pid, goal_dist)

    steering = clamp(steering, -1.0, 1.0)
    throttle = abs(radial_error) < 0.5 ? 0.15 : 0.16

    publish(controller.steering_pub, Float32(steering))
    publish(controller.throttle_pub, Float32(throttle))
end


#=
**** UTILITY FUNCTIONS ****
=#

function angle_wrap(angle::Float64)
    while angle > π
        angle -= 2π
    end
    while angle < -π
        angle += 2π
    end
    return angle
end

function euler_from_quaternion(x, y, z, w)
    #=
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
    =#
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = clamp(t2, -1.0, 1.0)
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan(t3, t4)

    return roll_x, pitch_y, yaw_z
end

function nowtime()
    return now(UTC)
end

# --- Main ---
if abspath(PROGRAM_FILE) == @__FILE__
    controller = JetRacerController()
    spin()
end
