import os
from random import choice
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from stable_baselines3.common.env_checker import check_env

# Constants
ACTION_NONE = 0
ACTION_ACCELERATE = 1
ACTION_BRAKE = 2
ACTION_SWITCH_LANE_0 = 3
ACTION_SWITCH_LANE_1 = 4
AGENT_VEHICLE_ID = 'agent'
ABSOLUTE_MAX_TRAFFIC = 20 # Maximum number of surrounding vehicles in the observation space, the actual max_traffic limit could be smaller
ACTION_LENGTH = 5 # 0.5 seconds for each agent action
STEP_LENGTH = 0.1 # Length of a step in seconds, in this case 1 second = 10 steps
LATERAL_RESOLUTION = 0.2 # Accuracy of side-to-side vehicle movement for lane changes
VISUAL_DELAY = 200 # Sets the speed of the visualization for the user
DIAGONAL = np.sqrt(2) * 100.0
DANGER_THRESHOLD = 0.15 # Distance at which vehicle is close enough to agent for penalties
MAX_SPEED = 25.0

class VehicleEnv(gym.Env):
  """Gymnasium environment using the SUMO traffic simulator to control a vehicle at a 4-way signalized intersection"""
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

  def __init__(self, steps_limit, max_traffic, collision_coef, timeout_coef, speed_coef, success_coef, proximity_coef, sumo_config_file, render_mode=None, render_resolution=(1920, 1080)):
    super().__init__()
    self.steps_limit = steps_limit
    self.max_traffic = max_traffic
    self.collision_coef = collision_coef
    self.timeout_coef = timeout_coef
    self.speed_coef = speed_coef
    self.success_coef = success_coef
    self.proximity_coef = proximity_coef
    self.sumo_config_file = sumo_config_file
    self.render_mode = render_mode
    self.render_resolution = render_resolution

    # 5 actions: do nothing, accelerate, brake
    self.action_space = spaces.Discrete(3)
    # One-hot encoded destinations (3 possible), ego and traffic vehicle positions, orientations and speeds
    self.observation_space = spaces.Box(low=-1, high=1, shape=(4 + 5 * (ABSOLUTE_MAX_TRAFFIC),), dtype=np.float32)
    self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    self.sumo = None
    self.label = os.getpid()
    self._cur_step = 0
    self._episode_ended = False
    self._success = False
    self._fail_collision = False
    self.episode_mean_speed = 0

    if self.render_mode is not None:
      self._sumo_binary = 'sumo-gui'
    else:
      self._sumo_binary = 'sumo'

  def close(self):
    if self.sumo is not None:
      try:
        self.sumo.close()
      except traci.exceptions.FatalTraCIError:
        pass # Already closed
      self.sumo = None

  def __del__(self):
    self.close()

  def render(self):
    if self.render_mode == 'human':
      return  # sumo-gui will already be rendering the frame
    elif self.render_mode == 'rgb_array':
      img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
                                     f'temp/img{self._cur_step}.jpg',
                                     width=self.render_resolution[0],
                                     height=self.render_resolution[1])
      return np.array(img)

  def reset(self, seed=None, options=None):
    super().reset(seed=seed, options=options)
    self._cur_step = 0
    self._episode_ended = False
    self._success = False
    self._fail_collision = False
    self.episode_mean_speed = 0

    # Set up SUMO command
    sumo_cmd = [
      self._sumo_binary,
      '-c', self.sumo_config_file,
      '--step-length', str(STEP_LENGTH),
      '--lateral-resolution', str(LATERAL_RESOLUTION),
      '--no-step-log',
      '--no-warnings',
      '--random',
      '--error-log', '/dev/null',
      '--max-num-vehicles', str(self.max_traffic + 1),
      #'--collision.mingap-factor', '0', # Only detect direct physical collisions
      '--collision.check-junctions',
      '--collision.action', 'remove' # Detect collisions
    ]
    if self.render_mode is not None:
      sumo_cmd.extend(['--delay', str(VISUAL_DELAY)])
      sumo_cmd.extend(['--start', '--quit-on-end'])
      if self.render_mode == 'rgb_array':
        sumo_cmd.extend(['--window-size', f'{self.render_resolution[0]},{self.render_resolution[1]}'])

    if self.sumo is None:
      # Start SUMO sim only the first time
      traci.start(sumo_cmd, port=self.label % 65536, label=self.label)
      self.sumo = traci.getConnection(self.label)
    else:
      # Just reload the simulation state
      self.sumo.load(sumo_cmd[1:])

    if self.render_mode is not None:
      self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, 'real world')

    # Choose random destination
    self.dest = choice(['-E0', 'E1', '-E2'])
    self.sumo.vehicle.setRoute(AGENT_VEHICLE_ID, ['-E3', self.dest])
    self.sumo.vehicle.setSpeedMode(AGENT_VEHICLE_ID, 96) # Disable safety checks for vehicle control
    self.sumo.simulationStep()

    info = self.get_info()
    observations = self.get_normalized_observation(info)

    return (
      observations,
      info
    )

  def step(self, action):
    self.process_action(action)

    for _ in range(ACTION_LENGTH):
      self.sumo.simulationStep()
      self._cur_step += 1
      if self._cur_step > self.steps_limit:
        self._episode_ended = True
        break
      if AGENT_VEHICLE_ID in self.sumo.simulation.getCollidingVehiclesIDList():
        self._fail_collision = True
        break
      elif AGENT_VEHICLE_ID in self.sumo.simulation.getArrivedIDList():
        self._success = True
        break
      new_vehicles = self.sumo.simulation.getDepartedIDList()
      for new_vehicle_id in new_vehicles:
        self.sumo.vehicle.setSpeedMode(new_vehicle_id, 117) # Disable safety checks for vehicle control

    terminated = self._fail_collision or self._episode_ended or self._success
    truncated = False # No truncate condition, timeout = fail
    info = self.get_info(terminated)
    observations = self.get_normalized_observation(info, terminated)
    reward = self.get_reward(observations, info)

    return (
      observations,
      reward,
      terminated,
      truncated,
      info,
    )

  def process_action(self, action):
    if action == ACTION_NONE:
      self.sumo.vehicle.setAcceleration(AGENT_VEHICLE_ID, 0.0, 0.5)
    elif action == ACTION_ACCELERATE:
      self.sumo.vehicle.setAcceleration(AGENT_VEHICLE_ID, 1.5, 0.5)
    elif action == ACTION_BRAKE:
      self.sumo.vehicle.setAcceleration(AGENT_VEHICLE_ID, -4.0, 0.5)
    elif action == ACTION_SWITCH_LANE_0:
      self.sumo.vehicle.changeLane(AGENT_VEHICLE_ID, 0, 0.5)
    elif action == ACTION_SWITCH_LANE_1:
      self.sumo.vehicle.changeLane(AGENT_VEHICLE_ID, 1, 0.5)

  def get_reward(self, observations, info):
    ego_speed = info['speed'] / MAX_SPEED
    collision_penalty = -self.collision_coef * self._fail_collision
    timeout_penalty = -self.timeout_coef * self._episode_ended
    speed_reward = self.speed_coef * ego_speed
    success_reward = self.success_coef * self._success
    proximity_penalty = 0.0

    # Skip the first 4 elements (ego data)
    traffic_obs = observations[4:]
    # Loop through each vehicle's observations
    for i in range(0, len(traffic_obs), 5):
      rel_x = traffic_obs[i]
      rel_y = traffic_obs[i+1]
      speed = traffic_obs[i+2]
      rel_angle = traffic_obs[i+3]
      dist = traffic_obs[i+4]

      # Skip vehicles that are too far
      if dist >= DANGER_THRESHOLD:
        continue

      # Check if the vehicle is aiming roughly at the agent (within 45 degrees)
      # Find vector pointing from the target vehicle TO the agent
      # Agent is at (0.5, 0.5)
      v_to_agent_x = 0.5 - rel_x
      v_to_agent_y = 0.5 - rel_y
      # Find the angle of this "line of sight" vector in radians
      bearing_to_agent = np.atan2(v_to_agent_y, v_to_agent_x)
      # Convert the target's relative angle back to radian
      target_heading_rad = (rel_angle - 0.5) * 2 * np.pi
      # Find the difference between where the car is looking vs where the agent is
      heading_to_bearing_diff = target_heading_rad - bearing_to_agent
      # Normalize to [-pi, pi]
      heading_to_bearing_diff = (heading_to_bearing_diff + np.pi) % (2 * np.pi) - np.pi
      # If the angle difference is within 45 deg, it's aimed at the agent
      is_heading_towards_agent = np.abs(heading_to_bearing_diff) < (np.pi / 4)

      # Ego is at (0.5, 0.5). If rel_y > 0.5, the car is in front of the agent
      is_in_front = rel_y > 0.5 and np.abs(rel_x - 0.5) < 0.02
      # print(f'v_to_agent_x: {v_to_agent_x}, v_to_agent_y: {v_to_agent_y}')
      # print(f'Rel X: {rel_x}, Rel Y: {rel_y}')
      # print(f'bearing_to_agent: {bearing_to_agent}, target_heading_rad: {target_heading_rad}, heading_to_bearing_diff: {heading_to_bearing_diff}')

      if is_heading_towards_agent or (is_in_front and ego_speed > 0.01):
        proximity_penalty += -self.proximity_coef * (1.0 - (dist / DANGER_THRESHOLD))**2
        # print(f'Heading to agent: {is_heading_towards_agent}, In front: {is_in_front}, proximity_penalty: {proximity_penalty}')

      # Remove speed reward if we're heading for a collision
      if is_in_front:
        speed_reward = 0.0

    # print(f'Collision Penalty: {collision_penalty}, Timeout Penalty: {timeout_penalty}, Speed Reward: {speed_reward}, Success Reward: {success_reward}, Proximity Penalty: {proximity_penalty}')
    return collision_penalty + timeout_penalty + speed_reward + success_reward + proximity_penalty

  def get_info(self, done=False):
    info = {
      'speed': 0.0,
    }

    if not done:
      speed = self.sumo.vehicle.getSpeed(AGENT_VEHICLE_ID)
      speed = speed * (speed > 0)
      info['speed'] = speed
      self.episode_mean_speed += speed

    else:
      episode_steps = self._cur_step // ACTION_LENGTH
      info['episode_mean_speed'] = self.episode_mean_speed / episode_steps
      info['success'] = self._success

    return info

  def get_normalized_observation(self, info, done=False):
    if done:
      return np.array([], dtype=np.float32)

    # Ego vehicle
    ego_x, ego_y = self.sumo.vehicle.getPosition(AGENT_VEHICLE_ID)
    ego_angle = self.sumo.vehicle.getAngle(AGENT_VEHICLE_ID)
    ego_angle_rad = np.radians(ego_angle)
    observations = np.array(
      # First one-hot encoded destination, then own speed
      [self.dest == '-E0', self.dest == 'E1', self.dest == '-E2', info['speed'] / MAX_SPEED],
      dtype=np.float32
    )

    # Other traffic
    vehicle_ids = sorted(self.sumo.vehicle.getIDList())
    for vehicle_id in vehicle_ids:
      if vehicle_id == AGENT_VEHICLE_ID:
        continue
      x, y = self.sumo.vehicle.getPosition(vehicle_id)

      # Delta position
      dx = x - ego_x
      dy = y - ego_y

      # Distance
      dist = np.sqrt(dx ** 2 + dy ** 2) / DIAGONAL

      # Rotate coordinates so they are relative to the Agent's heading
      rel_x = (dx * np.cos(ego_angle_rad) - dy * np.sin(ego_angle_rad)) / (2*DIAGONAL) + 0.5
      rel_y = (dx * np.sin(ego_angle_rad) + dy * np.cos(ego_angle_rad)) / (2*DIAGONAL) + 0.5

      # Relative angle to agent
      angle = self.sumo.vehicle.getAngle(vehicle_id)
      angle_diff = angle - ego_angle
      # Normalize difference to [-180, 180] degrees
      angle_diff = (angle_diff + 180) % 360 - 180
      # Scale to [0, 1]
      rel_angle = (angle_diff / 360.0) + 0.5

      speed = np.abs(self.sumo.vehicle.getSpeed(vehicle_id)) / MAX_SPEED
      speed = speed * (speed > 0)
      observations = np.append(observations, np.array([rel_x, rel_y, speed, rel_angle, dist], dtype=np.float32))

    # Pad empty slots with 1s (maximum distance away)
    observations = np.append(observations, np.ones(shape=(5 * (ABSOLUTE_MAX_TRAFFIC - len(vehicle_ids) + 1),), dtype=np.float32))

    return observations

if __name__ == '__main__':
  steps_limit = 128 * 5

  register(
    id='Vehicle-Sumo-v1',
    entry_point=VehicleEnv,
  )

  # test_env = gym.make(
  #   'Vehicle-Sumo-v1',
  #   steps_limit=steps_limit,
  #   max_traffic=10,
  #   collision_coef=100.0,
  #   timeout_coef=100.0,
  #   speed_coef=0.1,
  #   success_coef=10.0,
  #   proximity_coef=10.0,
  #   sumo_config_file='sumo_files/intersection_vehicle.sumocfg',
  #   render_mode='human'
  # )

  # print('Checking env')
  # check_env(test_env, warn=True)
  # print('Closing env')
  # test_env.close()

  env = gym.make(
    'Vehicle-Sumo-v1',
    steps_limit=steps_limit,
    max_traffic=1,
    collision_coef=100.0,
    timeout_coef=100.0,
    speed_coef=0.1,
    success_coef=100.0,
    proximity_coef=5.0,
    sumo_config_file='sumo_files/intersection_vehicle.sumocfg',
    render_mode='human'
  )
  obs, _ = env.reset()

  print('Running env')
  for step in range(steps_limit // 5 + 1):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    # print('obs=', obs, 'reward=', reward, 'done=', done, 'info=', info)
    if done:
      print('info=', info, 'reward=', reward)
      break
  print('Closing env')
  env.close()
