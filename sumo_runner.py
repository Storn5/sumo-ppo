import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

sumo_config = [
    'sumo-gui',
    '-c', 'intersection.sumocfg',
    '--step-length', '0.1',
    '--delay', '200',
    '--lateral-resolution', '0.1'
]

traci.start(sumo_config)

EPISODE_LENGH = 1000 # 100 second limit
MIN_PHASE_LENGTH = 30 # 3 seconds is the default length of a yellow signal

def main():    
    while steps < EPISODE_LENGH:
        steps += 1
        traci.simulationStep()
        if 'agent' in traci.vehicle.getIDList():
            vehicle_speed = traci.vehicle.getSpeed('agent')
            total_speed += vehicle_speed
            print(f'Speed: {vehicle_speed}')

    print(f'Average speed: {total_speed / steps}')

    traci.close()

if __name__ == '__main__':
    main()
