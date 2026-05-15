import os
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from stable_baselines3.common.env_checker import check_env

# Constants
ACTION_RESUME_PHASE = 0
ACTION_SWITCH_PHASE = 1
TRAFFIC_LIGHT_ID = 'J3'
NUM_PHASES = 4 # How many phases the traffic light has
NUM_LANES = 8 # How many lanes around the traffic light
MIN_PHASE_LENGTH = 30 # 3 seconds is the default length of a yellow signal
CAR_LENGTH = 5 # Default length of a car
MIN_GAP = 2.5 # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html)
STEP_LENGTH = 0.1 # Length of a step in seconds, in this case 1 second = 10 steps
LATERAL_RESOLUTION = 0.5 # Accuracy of side-to-side vehicle movement for lane changes
VISUAL_DELAY = 200 # Sets the speed of the visualization for the user
SUMO_FILE = 'sumo_env/sumo_files/intersection.sumocfg'

class LightsEnv(gym.Env):
  """Gymnasium environment using the SUMO traffic simulator to control a traffic light at a 4-way intersection"""
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

  def __init__(self, steps_limit, awt_coef, aql_coef, speed_coef, success_coef, render_mode=None, render_resolution=(1920, 1080)):
    super().__init__()
    self.steps_limit = steps_limit
    self.awt_coef = awt_coef
    self.aql_coef = aql_coef
    self.speed_coef = speed_coef
    self.success_coef = success_coef
    self.render_mode = render_mode
    self.render_resolution = render_resolution

    self.action_space = spaces.Discrete(2) # 2 actions - 0 = don't change signal, 1 = change signal
    self.observation_space = spaces.Box(low=0, high=1, shape=(2 + NUM_LANES,), dtype=np.float32) # One-hot encoded phases followed by lane queue density
    self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    self.sumo = None
    self.label = os.getpid()
    self._cur_step = 0
    self._cur_episode = 0
    self._episode_ended = False
    self.num_departed_vehicles = 0
    self.episode_mean_speed = 0
    self.episode_mean_waiting_time = 0
    self.episode_mean_queue_length = 0

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
    if self._cur_episode != 0:
      self.close()
    self._cur_episode += 1
    self.num_departed_vehicles = 0
    self.episode_mean_speed = 0
    self.episode_mean_waiting_time = 0
    self.episode_mean_queue_length = 0

    # Set up SUMO command
    sumo_cmd = [
      self._sumo_binary,
      '-c', SUMO_FILE,
      '--step-length', str(STEP_LENGTH),
      '--lateral-resolution', str(LATERAL_RESOLUTION),
      '--no-step-log',
      '--no-warnings',
      '--random'
    ]
    if self.render_mode is not None:
      sumo_cmd.extend(['--delay', str(VISUAL_DELAY)])
      sumo_cmd.extend(['--start', '--quit-on-end'])
      if self.render_mode == 'rgb_array':
        sumo_cmd.extend(['--window-size', f'{self.render_resolution[0]},{self.render_resolution[1]}'])

    # Start SUMO sim
    traci.start(sumo_cmd, port=self.label % 65536, label=self.label)
    self.sumo = traci.getConnection(self.label)
    if self.render_mode is not None:
      self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, 'real world')

    self.cur_phase = self.sumo.trafficlight.getPhase(TRAFFIC_LIGHT_ID)
    self.lanes = list(
      dict.fromkeys(self.sumo.trafficlight.getControlledLanes(TRAFFIC_LIGHT_ID))
    )
    self.lanes_length = self.sumo.lane.getLength(self.lanes[0])
    self.max_lane_occupancy = self.lanes_length / (MIN_GAP + CAR_LENGTH)

    observations = self.get_normalized_observation()
    info = self.get_info()

    return (
      observations,
      info
    )

  def step(self, action):
    # If we don't change phase, just run for 2 min phases
    # If we do, run a yellow phase, then change and run another min phase length
    for _ in range(2):
      # Switch to the next phase (first to yellow, then swap)
      if action == ACTION_SWITCH_PHASE:
        self.switch_lights()

      # Then run for a min phase length
      for _ in range(MIN_PHASE_LENGTH):
        self.sumo.simulationStep()

        self.num_departed_vehicles += self.sumo.simulation.getDepartedNumber()

        self._cur_step += 1
        if self._cur_step > self.steps_limit:
          self._episode_ended = True
          break
      if self._episode_ended:
        break

    terminated = False # No termination condition
    truncated = self._episode_ended
    observations = self.get_normalized_observation()
    info = self.get_info(truncated)
    reward = self.get_reward(info)

    return (
      observations,
      reward,
      terminated,
      truncated,
      info,
    )

  def switch_lights(self):
    self.cur_phase = (self.cur_phase + 1) % NUM_PHASES
    self.sumo.trafficlight.setPhase(TRAFFIC_LIGHT_ID, self.cur_phase)

  def get_reward(self, info):
    awt_reward = -self.awt_coef * info['mean_waiting_time']
    aql_reward = -self.aql_coef * info['total_queued']
    speed_reward = self.speed_coef * info['mean_speed']
    success_reward = self.success_coef * info['total_departed']
    # print(f'AWT Reward: {awt_reward}, AQL Reward: {aql_reward}, Speed Reward: {speed_reward}, Success Reward: {success_reward}')
    return awt_reward + aql_reward + speed_reward + success_reward

  def get_info(self, done=False):
    vehicles = self.sumo.vehicle.getIDList()
    mean_speed = 100.0 # High mean speed for a good reward, since there's no vehicles waiting
    queue_length = 0.0
    mean_waiting_time = 0.0

    if len(vehicles):
      mean_speed = np.mean([self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles])
      queue_length = sum([self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])
      mean_waiting_time = np.mean([self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles])

    self.episode_mean_speed += mean_speed
    self.episode_mean_queue_length += queue_length
    self.episode_mean_waiting_time += mean_waiting_time

    info = {
      'mean_speed': mean_speed,
      'total_queued': queue_length,
      'mean_waiting_time': mean_waiting_time,
      'total_departed': self.num_departed_vehicles,
    }

    if done:
      episode_steps = self._cur_step // (MIN_PHASE_LENGTH * 2)
      info['episode_mean_speed'] = self.episode_mean_speed / episode_steps
      info['episode_mean_queue_length'] = self.episode_mean_queue_length / episode_steps
      info['episode_mean_waiting_time'] = self.episode_mean_waiting_time / episode_steps

    return info

  def get_normalized_observation(self):
    phase_id_ohe = [self.cur_phase == 0, self.cur_phase != 0] # One-hot encoding
    lanes_density = [
      self.sumo.lane.getLastStepVehicleNumber(lane) / self.max_lane_occupancy
      for lane in self.lanes
    ]
    observations = np.array(
      phase_id_ohe + lanes_density,
      dtype=np.float32
    )
    return observations

if __name__ == '__main__':
  steps_limit = 9_000

  register(
    id='Lights-Sumo-v1',
    entry_point=LightsEnv,
  )

  # test_env = gym.make(
  #   'Lights-Sumo-v1',
  #   steps_limit=steps_limit,
  #   awt_coef=0.25,
  #   aql_coef=0.25,
  #   speed_coef=0.25,
  #   success_coef=0.25,
  #   render_mode='human'
  # )

  # print('Checking env')
  # check_env(test_env, warn=True)
  # print('Closing env')
  # test_env.close()

  env = gym.make(
    'Lights-Sumo-v1',
    steps_limit=steps_limit,
    awt_coef=0.05,
    aql_coef=0.05,
    speed_coef=0.15,
    success_coef=0.005,
    render_mode='human'
  )
  obs, _ = env.reset()

  print('Running env')
  for step in range(steps_limit // 60):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print('obs=', obs, 'reward=', reward, 'done=', done, 'info=', info)
    if done:
      break
  print('Closing env')
  env.close()
