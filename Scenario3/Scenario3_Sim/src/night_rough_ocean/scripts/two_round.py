#!/usr/bin/env python3
import os
import time
import asyncio
import platform
import base64
from typing import Any, List

import fire
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.team import Team

# 清除代理环境变量
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('socks_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)

class AssessPlatformFromDrone(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for observing the robotic arm platform during a drone and robotic arm platform docking mission.

    Additional Information:
    - The robotic arm platform is located on a rough ocean surface.
    - The drone's battery level is sufficient and no need to land for charging.

    Analyze the image from the drone camera and additional information, provide your initial judgment on whether the platform and environment allow for a safe docking.

    Provide a concise response in no more than 80 words.
    """

    name: str = "AssessPlatformFromDrone"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_drone", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)
        
        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"AssessPlatformFromDrone Prompt: {prompt}")
        rsp = await self._aask_image(prompt, encoded_image)
        return rsp

    async def _aask_image(self, prompt: str, image_data: str) -> str:
        message = self.llm._user_msg_with_imgs(prompt, image_data)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class AssessDroneFromPlatform(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for observing the drone during a drone and robotic arm platform docking mission.

    Additional Information:
    - The operation is taking place at night.
    - The docking platform has not been serviced recently, and its condition is uncertain.

    Analyze the image from the platform camera and additional information, provide your initial judgment on whether the drone and environment allow for a safe docking.

    Provide a concise response in no more than 80 words.
    """

    name: str = "AssessDroneFromPlatform"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_robot_arm", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)
        
        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"AssessDroneFromPlatform Prompt: {prompt}")
        rsp = await self._aask_image(prompt, encoded_image)
        return rsp

    async def _aask_image(self, prompt: str, image_data: str) -> str:
        message = self.llm._user_msg_with_imgs(prompt, image_data)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class FinalDecision(Action):
    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Topic: {topic}
    You are {name}, collaborating with {teammate_name} to achieve a successful docking mission.
    ## IMAGE ANALYSIS RESULT
    The latest image analysis indicates: {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Based on the image analysis result and previous communication, make a final decision: Is docking advisable? 
    Respond with either "yes" or "no".
    """
    name: str = "FinalDecision"

    async def run(self, context: str, assessment_result: str, name: str, teammate_name: str, topic: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, assessment_result=assessment_result, name=name, teammate_name=teammate_name, topic=topic)
        print(f"FinalDecision Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp

class DroneRobot(Role):
    name: str = "DroneRobot"
    profile: str = "Drone Observer"
    teammate_name: str = "PlatformRobot"
    aspect: str = "platform"
    peer_aspect: str = "drone"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessPlatformFromDrone(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessPlatformFromDrone, FinalDecision])
        self.assessment_result = ""
        self.topic = "drone docking mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessPlatformFromDrone):
            rsp = await todo.run()
            self.assessment_result = rsp
        else:  # FinalDecision
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammate_name, topic=self.topic)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammate_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammate_name}: {rsp}")
        return msg
    
class PlatformRobot(Role):
    name: str = "PlatformRobot"
    profile: str = "Platform Observer"
    teammate_name: str = "DroneRobot"
    aspect: str = "drone"
    peer_aspect: str = "platform"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessDroneFromPlatform(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessDroneFromPlatform, FinalDecision])
        self.assessment_result = ""
        self.topic = "drone docking mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessDroneFromPlatform):
            rsp = await todo.run()
            self.assessment_result = rsp
        else:  # FinalDecision
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammate_name, topic=self.topic)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammate_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammate_name}: {rsp}")
        return msg

async def cooperate(idea: str, investment: float = 3.0, n_round: int = 2):
    rospy.init_node('drone_platform_docking', anonymous=True)

    start_time = time.time()
    
    drone_robot = DroneRobot()
    platform_robot = PlatformRobot()

    team = Team()
    team.hire([drone_robot, platform_robot])
    team.invest(investment)
    team.run_project(idea, send_to="DroneRobot")

    message_pool = []
    for _ in range(n_round):
        message_pool.append(await drone_robot._act(AssessPlatformFromDrone(), message_pool))
        message_pool.append(await platform_robot._act(AssessDroneFromPlatform(), message_pool))
        message_pool.append(await drone_robot._act(FinalDecision(), message_pool))
        message_pool.append(await platform_robot._act(FinalDecision(), message_pool))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

def main(idea: str, investment: float = 3.0, n_round: int = 1):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cooperate(idea, investment, n_round))

if __name__ == "__main__":
    fire.Fire(main)
# rosrun night_rough_ocean two_round.py --idea "drone docking mission"