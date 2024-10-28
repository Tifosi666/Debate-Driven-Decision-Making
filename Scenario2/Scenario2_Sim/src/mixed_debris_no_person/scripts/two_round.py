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

class AssessDebris(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for assessing the debris during a joint rescue mission.

    Additional Information:
    - The current search environment contains both small ruins and some medium-sized collapsed structures.
    - There is no risk of aftershocks at the site.

    Analyze the image from the humanoid robot's camera and additional information.
    Provide your assessment in no more than 50 words.
    """

    name: str = "AssessDebris"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_medium_ruin_robot", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)
        
        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"AssessDebris Prompt: {prompt}")
        rsp = await self._aask_image(prompt, encoded_image)
        return rsp

    async def _aask_image(self, prompt: str, image_data: str) -> str:
        message = self.llm._user_msg_with_imgs(prompt, image_data)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class AssessSurvivor(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for searching for survivors during a joint rescue mission.

    Additional Information:
    - No signs of life have been detected in the ruins.
    - No immediate need for emergency assistance has been detected.

    Analyze the image from the robot dog's camera and additional information.
    Provide your assessment in no more than 50 words.
    """

    name: str = "AssessSurvivor"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_medium_ruin_dog", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)
        
        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"AssessSurvivor Prompt: {prompt}")
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
    You are {name}, collaborating with {teammate_name} to achieve a successful rescue mission.
    ## IMAGE ANALYSIS RESULT
    The latest image analysis indicates: {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Based on the image analysis result and previous communication, make a final decision: Should the team continue the rescue operation? 
    Respond with either "yes" or "no".
    """
    name: str = "FinalDecision"

    async def run(self, context: str, assessment_result: str, name: str, teammate_name: str, topic: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, assessment_result=assessment_result, name=name, teammate_name=teammate_name, topic=topic)
        print(f"FinalDecision Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp

class HumanoidRobot(Role):
    name: str = "HumanoidRobot"
    profile: str = "Debris Cleaner"
    teammate_name: str = "RobotDog"
    aspect: str = "debris"
    peer_aspect: str = "survivor"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessDebris(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessDebris, FinalDecision])
        self.assessment_result = ""
        self.topic = "rescue mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessDebris):
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
    
class RobotDog(Role):
    name: str = "RobotDog"
    profile: str = "Survivor Finder"
    teammate_name: str = "HumanoidRobot"
    aspect: str = "survivor"
    peer_aspect: str = "debris"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessSurvivor(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessSurvivor, FinalDecision])
        self.assessment_result = ""
        self.topic = "rescue mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessSurvivor):
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
    rospy.init_node('rescue_mission', anonymous=True)

    start_time = time.time()
    
    humanoid_robot = HumanoidRobot()
    robot_dog = RobotDog()

    team = Team()
    team.hire([humanoid_robot, robot_dog])
    team.invest(investment)
    team.run_project(idea, send_to="HumanoidRobot")

    message_pool = []
    for _ in range(n_round):
        message_pool.append(await humanoid_robot._act(AssessDebris(), message_pool))
        message_pool.append(await robot_dog._act(AssessSurvivor(), message_pool))
        message_pool.append(await humanoid_robot._act(FinalDecision(), message_pool))
        message_pool.append(await robot_dog._act(FinalDecision(), message_pool))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

def main(idea: str, investment: float = 3.0, n_round: int = 1):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cooperate(idea, investment, n_round))

if __name__ == "__main__":
    fire.Fire(main)

# rosrun mixed_debris_no_person two_round.py --idea "joint rescue mission"
