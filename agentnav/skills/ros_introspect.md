# ROS2 Introspection Skill

Use these tools to autonomously explore any ROS2 robot's nodes, topics, and services without prior knowledge of the robot's configuration.

## Discovery Workflow

When connecting to an unknown robot, explore in this order:

1. `ros_list_nodes()` — see which nodes are running
2. `ros_list_topics(show_types=True)` — list all topics with message types
3. `ros_topic_info('/topic_name')` — see publishers/subscribers for a topic
4. `ros_topic_echo('/topic_name', timeout_s=5)` — sample actual data

## Common Patterns

### Check robot velocity interface
```
ros_topic_info('/cmd_vel')
→ {type: 'geometry_msgs/msg/Twist', subscriber_count: 1, publishers: [], ...}
```
`publisher_count=0` means no other controller is publishing velocity — safe to take control.

### Read sensor data
```
ros_topic_echo('/scan', timeout_s=5)          # lidar
ros_topic_echo('/odom', timeout_s=3)          # odometry
ros_topic_echo('/camera/image_raw', timeout_s=5)  # camera (returns metadata, not image)
ros_topic_echo('/battery_state', timeout_s=30)    # low-frequency topics need longer timeout
```

### Publish a one-shot message
```
ros_topic_pub('/cmd_vel', 'geometry_msgs/msg/Twist',
              {'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
               'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}})
```

### Call a service
```
ros_service_list()   # find available services first
ros_service_call('/clear_costmaps', 'std_srvs/srv/Empty', {})
ros_service_call('/reinitialize_global_localization', 'std_srvs/srv/Empty', {})
```

## Safety Rules

- **Motion topics carry a warning**: `ros_topic_pub` always returns a `warning` field for `/cmd_vel` and similar topics.
  Use `robot_status()` to confirm the robot is in `IDLE` state and the area is clear before publishing.
- **Use `robot_stop()` for emergency stop**, not `ros_topic_pub`.
  `robot_stop()` guarantees < 50ms response via a dedicated stop_flag — more reliable.
- **Cancel tasks before resetting localization**: `/reinitialize_global_localization` affects ongoing navigation;
  use `task_cancel(task_id)` to cancel the current task first.

## Timeout Guidelines

| Topic type | Recommended timeout_s |
|------------|----------------------|
| High-frequency (> 1 Hz) | 3–5 |
| Medium-frequency (0.1–1 Hz) | 10–15 |
| Low-frequency (< 0.1 Hz) | 20–30 |
| Service call | 10 (default) |

## Diagnosing ROS_DOMAIN_ID Issues

If `ros_list_nodes()` returns an empty list but the robot is confirmed running:
- Check the `ros_domain_id` field in the response
- The robot may be on a different ROS_DOMAIN_ID
- Ask the user to confirm, or set the `ROS_DOMAIN_ID` environment variable in the startup script and restart the bridge

## Notes

- All tools return `{"error": "ROS2 not available: ..."}` instead of crashing when ROS2 is unavailable
- `ros_topic_echo` returns `{"error": "timeout"}` on timeout — use `ros_topic_info` first to confirm the topic has publishers
- The `type` field returned by `ros_service_list()` is the `srv_type` parameter needed by `ros_service_call`
