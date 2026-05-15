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

class VehicleEnv(gym.Env):
  """Gymnasium environment using the SUMO traffic simulator to control a vehicle at a 4-way signalized intersection"""
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

  def __init__(self, steps_limit, max_traffic, collision_coef, timeout_coef, speed_coef, success_coef, sumo_config_file, render_mode=None, render_resolution=(1920, 1080)):
    super().__init__()
    self.steps_limit = steps_limit
    self.max_traffic = max_traffic
    self.collision_coef = collision_coef
    self.timeout_coef = timeout_coef
    self.speed_coef = speed_coef
    self.success_coef = success_coef
    self.sumo_config_file = sumo_config_file
    self.render_mode = render_mode
    self.render_resolution = render_resolution

    # 5 actions: do nothing, accelerate, brake, go to left lane, go to right lane
    self.action_space = spaces.Discrete(5)
    # One-hot encoded destinations (3 possible), ego and traffic vehicle positions, orientations and speeds
    self.observation_space = spaces.Box(low=0, high=1, shape=(3 + 4 * (1 + ABSOLUTE_MAX_TRAFFIC),), dtype=np.float32)
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
    reward = self.get_reward(info)

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
      self.sumo.vehicle.setAcceleration(AGENT_VEHICLE_ID, 1.0, 0.5)
    elif action == ACTION_BRAKE:
      self.sumo.vehicle.setAcceleration(AGENT_VEHICLE_ID, -0.5, 0.5)
    elif action == ACTION_SWITCH_LANE_0:
      self.sumo.vehicle.changeLane(AGENT_VEHICLE_ID, 0, 0.5)
    elif action == ACTION_SWITCH_LANE_1:
      self.sumo.vehicle.changeLane(AGENT_VEHICLE_ID, 1, 0.5)

  def get_reward(self, info):
    collision_reward = -self.collision_coef * self._fail_collision
    timeout_reward = -self.timeout_coef * self._episode_ended
    speed_reward = self.speed_coef * info['speed']
    success_reward = self.success_coef * self._success
    # print(f'Collision Reward: {collision_reward}, Timeout Reward: {timeout_reward}, Speed Reward: {speed_reward}, Success Reward: {success_reward}')
    return collision_reward + timeout_reward + speed_reward + success_reward

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

    return info

  def get_normalized_observation(self, info, done=False):
    if done:
      return np.array([], dtype=np.float32)

    # Ego vehicle
    x, y = self.sumo.vehicle.getPosition(AGENT_VEHICLE_ID)
    x, y = x / 200.0 + 0.5, y / 200.0 + 0.5
    angle = self.sumo.vehicle.getAngle(AGENT_VEHICLE_ID) / 360.0
    observations = np.array(
      [self.dest == '-E0', self.dest == 'E1', self.dest == '-E2', x, y, info['speed'] / 20.0, angle],
      dtype=np.float32
    )

    # Other traffic
    vehicle_ids = sorted(self.sumo.vehicle.getIDList())
    for vehicle_id in vehicle_ids:
      if vehicle_id == AGENT_VEHICLE_ID:
        continue
      x, y = self.sumo.vehicle.getPosition(vehicle_id)
      x, y = x / 200.0 + 0.5, y / 200.0 + 0.5
      angle = self.sumo.vehicle.getAngle(vehicle_id) / 360.0
      speed = np.abs(self.sumo.vehicle.getSpeed(vehicle_id)) / 20.0
      speed = speed * (speed > 0)
      observations = np.append(observations, np.array([x, y, speed, angle], dtype=np.float32))

    observations = np.append(observations, np.zeros(shape=(4 * (ABSOLUTE_MAX_TRAFFIC - len(vehicle_ids) + 1),), dtype=np.float32))

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
    max_traffic=10,
    collision_coef=100.0,
    timeout_coef=100.0,
    speed_coef=0.1,
    success_coef=10.0,
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
