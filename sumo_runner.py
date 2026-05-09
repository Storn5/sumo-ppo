import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

sumo_config = [
    'sumo-gui',
    '-c', 'tjunction.sumocfg',
    '--step-length', '0.1',
    '--delay', '200',
    '--lateral-resolution', '0.1'
]

traci.start(sumo_config)

total_speed = 0
steps = 0

while steps < 100:
    steps += 1
    traci.simulationStep()
    if 'agent' in traci.vehicle.getIDList():
        vehicle_speed = traci.vehicle.getSpeed('agent')
        total_speed += vehicle_speed
        print(f'Speed: {vehicle_speed}')

print(f'Average speed: {total_speed / steps}')

traci.close()
