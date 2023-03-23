import sys
import json
import time
import numpy as np
import MalmoPython
import xml.etree.ElementTree

__chunks__ = None


class Environment:
    def __init__(self, speed: float = 1.0, size: tuple = (512, 288), chunk_num: int = 20, time_limit: float = 300.0, start_pos: tuple = (0, 64, 0), block_type: str = 'redstone_ore'):
        self.speed = speed
        self.start_pos = np.array(start_pos)
        self.target_pos = None
        self.chunk_num = chunk_num
        self.block_type = block_type
        self.time_limit = time_limit
        self.width, self.height = size

        self.prev_timestamp = None
        self.action_cost = None
        self.agent_host = None
        self.prev_pos = None

        self.stuck_counter = 0
        self.traveled = 0

        self.restart()

    def restart(self):
        self.wait_mission_end()

        global __chunks__
        if __chunks__ is None:
            with open('chunks.json') as f:
                __chunks__ = json.load(f)

        blocks = {tuple(self.start_pos + np.array([x, 0, 0])) for x in range(8)}
        pos = self.start_pos + np.array([8, 0, 0])

        weights = np.array([c['weight'] for c in __chunks__])
        weights /= np.sum(weights)
        for _ in range(self.chunk_num):
            chunk = np.random.choice(__chunks__, p=weights)
            assert [0, 0, 0] in chunk['blocks'], 'missing block at (0, 0, 0)'
            for block in chunk['blocks']:
                assert block[1] >= 0
                blocks.add(tuple(pos + np.array(block)))
            pos += np.array(chunk['offset'])
        self.target_pos = pos

        max_radio = np.max(np.abs(np.array([b for c in __chunks__ for b in c['blocks']])), axis=0) + 2
        x1, y1, z1 = tuple(self.start_pos - (self.chunk_num + 1) * max_radio)
        x2, y2, z2 = tuple(self.start_pos + (self.chunk_num + 1) * max_radio)
        map_xml = f'<DrawCuboid x1="{x1}" x2="{x2}" y1="{max(y1, 0)}" y2="{min(y2, 255)}" z1="{z1}" z2="{z2}" type="air"/>\n'
        map_xml += '\n'.join(f'<DrawBlock x="{x}" y="{y}" z="{z}" type="{self.block_type}"/>' for x, y, z in blocks)

        mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                            <About>
                                <Summary>Minecraft Parkour</Summary>
                            </About>
                            <ServerSection>
                                <ServerInitialConditions>
                                    <Time>
                                        <StartTime>1000</StartTime>
                                        <AllowPassageOfTime>false</AllowPassageOfTime>
                                    </Time>
                                    <Weather>clear</Weather>
                                </ServerInitialConditions>
                                <ServerHandlers>
                                    <FlatWorldGenerator generatorString="3;1*minecraft:air;0;"/>
                                    <DrawingDecorator>
                                        ''' + map_xml + '''
                                    </DrawingDecorator>
                                    ''' + f'<ServerQuitFromTimeUp timeLimitMs="{self.time_limit * 1000:3.1f}"/>' + '''
                                    <ServerQuitWhenAnyAgentFinishes/>
                                </ServerHandlers>
                            </ServerSection>
                            
                            <AgentSection mode="Survival">
                                <Name>ParkourBot</Name>
                                <AgentStart>
                                    ''' + f'<Placement x="{self.start_pos[0] + 0.5}" y="{self.start_pos[1] + 1}" z="{self.start_pos[2] + 0.5}" yaw="-90" pitch="43"/>' + '''
                                    <Inventory>
                                        <InventoryItem slot="0" type="redstone_ore" quantity="64"/>
                                    </Inventory>
                                </AgentStart>
                                <AgentHandlers>
                                    <ObservationFromFullStats/>
                                    <ContinuousMovementCommands/>
                                    <MissionQuitCommands/>
                                    <AgentQuitFromReachingPosition>
                                        ''' + f'<Marker x="{self.target_pos[0] + 0.5}" y="{self.target_pos[1] + 1}" z="{self.target_pos[2] + 0.5}" tolerance="1.0" description="Goal_Reached"/>' + '''
                                    </AgentQuitFromReachingPosition>
                                    <VideoProducer>
                                        <Width>''' + str(self.width) + '''</Width>
                                        <Height>''' + str(self.height) + '''</Height>
                                    </VideoProducer>
                                </AgentHandlers>
                            </AgentSection>
                        </Mission>'''

        # Create default Malmo objects:
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)
        if self.agent_host.receivedArgument("help"):
            print(self.agent_host.getUsage())
            exit(0)

        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission_record = MalmoPython.MissionRecordSpec()

        # Attempt to start a mission:
        max_retries = 100
        for retry in range(max_retries):
            try:
                self.agent_host.startMission(my_mission, my_mission_record)
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(0.01)

        self.prev_pos = self.start_pos + np.array([0.5, 1, 0.5])
        self.prev_timestamp = None
        self.stuck_counter = 0
        self.action_cost = 0
        self.traveled = 0

        self.wait_mission_start()
        self.agent_host.sendCommand(f'move {self.speed}')
        time.sleep(0.3)
        self.get_state()

    def get_state(self):
        while True:
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                for error in world_state.errors:
                    print("Error:", error.text)

            if world_state.has_mission_begun:
                if world_state.is_mission_running:
                    if world_state.number_of_video_frames_since_last_state > 0:
                        break
                else:
                    break
            time.sleep(0.001)

        termination_speed = np.array([0, 0, 0])
        died_reward = -5
        if world_state.is_mission_running:
            frame = world_state.video_frames[0]
            image, pos = np.frombuffer(frame.pixels, np.uint8).reshape(self.height, self.width, 3), np.array([frame.xPos, frame.yPos, frame.zPos])
            moved, dt = pos - self.prev_pos, (frame.timestamp - self.prev_timestamp).total_seconds() if self.prev_timestamp is not None else 0.1
            speed = moved / dt
            self.prev_pos, self.prev_timestamp = pos, frame.timestamp
            self.traveled = max(self.traveled, pos[0])

            if speed[0] < 0.5:
                self.stuck_counter += 1
            else:
                self.stuck_counter = max(self.stuck_counter - 1, 0)

            if pos[1] < self.start_pos[1] + 1:
                self.agent_host.sendCommand('quit')
                return None, died_reward, termination_speed
            elif self.stuck_counter >= 3:
                self.agent_host.sendCommand('quit')
                return None, -2, termination_speed
            else:
                return image, 0.5 + self.action_cost - 0.02*speed[2]**2 - 0.02*speed[1]**2 - 0.5*self.stuck_counter, speed
        else:
            if len(world_state.mission_control_messages) > 0:
                for message in world_state.mission_control_messages:
                    message = xml.etree.ElementTree.fromstring(message.text)
                    state = message.find('{http://ProjectMalmo.microsoft.com}HumanReadableStatus')
                    if state is None:
                        continue
                    state = state.text
                    if state is None:
                        continue
                    if state == 'MALMO_AGENT_DIED':
                        return None, died_reward, termination_speed
                    if state == 'Goal_Reached':
                        return None, 10, termination_speed
            return None, 0, termination_speed

    def take_action(self, action: int):
        # action = placed * 7 * 2 + jump * 7 + dir * 1
        placed, jump, direct = action // 7 // 2, (action // 7) % 2, action % 7
        self.agent_host.sendCommand(f'use {placed}')
        self.agent_host.sendCommand(f'jump {jump}')
        self.agent_host.sendCommand(f'move {self.speed}')
        self.agent_host.sendCommand(f'strafe {self.speed * (direct - 3) / 3}')
        self.agent_host.sendCommand('use 0')
        self.action_cost = - (jump * 0.1 + placed * 0.4)

    def wait_mission_end(self):
        if self.agent_host is None:
            return
        while True:
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                for error in world_state.errors:
                    print("Error:", error.text)

            if not (world_state.has_mission_begun and world_state.is_mission_running):
                break
            if world_state.has_mission_begun and world_state.is_mission_running:
                self.agent_host.sendCommand('quit')
            time.sleep(0.01)

    def wait_mission_start(self):
        if self.agent_host is None:
            return
        while True:
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                for error in world_state.errors:
                    print("Error:", error.text)

            if world_state.has_mission_begun:
                break
            time.sleep(0.01)
