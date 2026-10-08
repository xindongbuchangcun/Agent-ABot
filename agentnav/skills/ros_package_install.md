# ROS2 Package Install Skill

当用户提供一个 GitHub 链接或功能包名称，按以下流程自主完成：下载 → 安装依赖 → 编译 → 学会使用。

---

## Step 1：定位 ROS2 工作空间

```bash
# 按优先级检查常见工作空间位置
ls ~/wheeltec_ros2/src 2>/dev/null && echo "ws=~/wheeltec_ros2"
ls ~/ros2_ws/src       2>/dev/null && echo "ws=~/ros2_ws"
ls ~/colcon_ws/src     2>/dev/null && echo "ws=~/colcon_ws"
```

选择存在 `src/` 目录的那个。若都不存在，创建标准工作空间：
```bash
mkdir -p ~/ros2_ws/src && echo "ws=~/ros2_ws"
```

以下步骤中 `$WS` 代表确定的工作空间路径。

---

## Step 2：获取功能包

### 情况 A：用户提供 GitHub 链接

```bash
cd $WS/src
git clone <github_url>
# 若有子模块：
git clone --recurse-submodules <github_url>
```

克隆完成后读取 README：
```bash
cat $WS/src/<package_name>/README.md 2>/dev/null | head -100
```

### 情况 B：用户提供功能包名（无链接）

先尝试 apt 安装（最简单，无需编译）：
```bash
sudo apt install ros-humble-<package-name-with-dashes> -y
```
- 若 apt 成功：跳到 **Step 4（验证）**
- 若 apt 找不到：在 GitHub 搜索 `ros2 <package_name>`，找到仓库后按情况 A 处理

### 情况 C：包名转连字符规则（apt）

ROS2 apt 包名规则：`<package_name>` 中的下划线 `_` 全部换成 `-`
```
slam_toolbox  →  ros-humble-slam-toolbox
nav2_bringup  →  ros-humble-nav2-bringup
```

---

## Step 3：安装依赖

```bash
cd $WS
# 更新 rosdep（首次使用需初始化）
rosdep update
# 自动安装所有依赖
rosdep install --from-paths src --ignore-src -r -y
```

若 rosdep 未初始化：
```bash
sudo rosdep init && rosdep update
```

---

## Step 4：编译

```bash
cd $WS
source /opt/ros/humble/setup.bash
# 只编译新包（速度快）
colcon build --packages-select <package_name> --symlink-install
# 若不确定包名，或有依赖其他包：
colcon build --symlink-install
```

编译成功标志：`Summary: X packages finished`

常见编译错误处理：
- `CMake Error: could not find package` → rosdep 漏了依赖，手动 `sudo apt install ros-humble-<dep>`
- `Python import error` → `pip install <dep>` 或检查 package.xml 中的 `<exec_depend>`
- `colcon: command not found` → `sudo apt install python3-colcon-common-extensions`

---

## Step 5：激活环境

```bash
source $WS/install/setup.bash
```

验证包已安装：
```bash
ros2 pkg list | grep <package_name>
```

若每次启动都需要此包，将 source 命令加入 bridge 启动脚本，或提示用户添加到 `~/.bashrc`。

---

## Step 6：学习使用这个包

### 6.1 查看包提供了什么

```bash
# 可执行文件（节点）
ros2 pkg executables <package_name>

# 消息类型
ros2 interface list | grep <package_name>

# launch 文件
find $WS/install/<package_name> -name "*.launch.py" 2>/dev/null
```

### 6.2 读取文档

```bash
# README（最重要）
cat $WS/src/<package_name>/README.md

# launch 文件参数
cat $WS/src/<package_name>/launch/*.launch.py 2>/dev/null | head -80
```

### 6.3 试运行，观察节点和话题

```bash
# 后台启动（或用 launch 文件）
ros2 run <package_name> <executable> &
sleep 3

# 用内省工具观察它发布/订阅了什么
ros_list_nodes()
ros_list_topics(show_types=True)
ros_topic_info('/<new_topic>')
```

运行后通过 `ros_topic_echo` 采样数据，理解消息格式。

### 6.4 查看节点参数

```bash
ros2 node list
ros2 param list /<node_name>
ros2 param describe /<node_name> <param_name>
```

---

## 完整示例

```
User: "帮我安装 slam_toolbox，GitHub: https://github.com/SteveMacenski/slam_toolbox"

Agent:
  1. ls ~/wheeltec_ros2/src → 工作空间 = ~/wheeltec_ros2

  2. cd ~/wheeltec_ros2/src
     git clone https://github.com/SteveMacenski/slam_toolbox

  3. cat ~/wheeltec_ros2/src/slam_toolbox/README.md | head -80
     → 了解：这是一个 2D SLAM 包，提供 online_async_launch.py

  4. cd ~/wheeltec_ros2
     rosdep install --from-paths src --ignore-src -r -y
     colcon build --packages-select slam_toolbox --symlink-install

  5. source ~/wheeltec_ros2/install/setup.bash
     ros2 pkg list | grep slam_toolbox  → slam_toolbox ✓

  6. ros2 pkg executables slam_toolbox
     → sync_slam_toolbox_node, async_slam_toolbox_node, ...

     find .../install/slam_toolbox -name "*.launch.py"
     → online_async_launch.py, online_sync_launch.py, ...

     ros2 launch slam_toolbox online_async_launch.py &
     sleep 5
     ros_list_topics()
     → /map, /slam_toolbox/scan_visualization, /pose, ...

  回复用户：
  "slam_toolbox 已编译安装完毕。
   启动命令：ros2 launch slam_toolbox online_async_launch.py
   主要话题：/map（地图输出）、/scan（激光输入，需连接 lidar）
   可用服务：/slam_toolbox/save_map、/slam_toolbox/serialize_map"
```

---

## 注意事项

- **不要在 /opt/ros/humble 中修改文件**，始终安装到工作空间 src/
- `--symlink-install` 允许直接编辑 Python 文件而无需重新编译
- 编译后必须重新 `source install/setup.bash`，旧终端不会自动更新
- 若包依赖特定硬件（如 GPU），在没有该硬件的机器上编译可能失败，提示用户
- 大型包（如 navigation2）编译时间较长（5–15 分钟），告知用户耐心等待
