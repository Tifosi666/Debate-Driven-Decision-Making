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

class AssessEnvironment(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for observing the surrounding environment during a liquid transport mission.

    Additional Information:
    - You are moving on a slope.
    - The robot's battery level is sufficient.
    - There are no other urgent transport tasks after this mission.
    - You and your teammate in this mission are physically identical.

    Analyze the image and provide your initial judgment on whether the environment requires a speed reduction.

    Provide a concise response in no more than 50 words.
    """

    name: str = "AssessEnvironment"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_front", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)
        
        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"AssessEnvironment Prompt: {prompt}")
        rsp = await self._aask_image(prompt, encoded_image)
        return rsp

    async def _aask_image(self, prompt: str, image_data: str) -> str:
        message = self.llm._user_msg_with_imgs(prompt, image_data)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class AssessLiquidStatus(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for observing the liquid status during a transport mission.

    Additional Information:
    - The liquid being transported is water, which is not a viscous substance.
    - The robot's battery level is low.
    - There are no other urgent transport tasks after this mission.
    - You and your teammate in this mission are physically identical.

    Analyze the image and provide your initial judgment on whether the liquid status requires a speed reduction.

    Provide a concise response in no more than 50 words.
    """

    name: str = "AssessLiquidStatus"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_behind", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)
        
        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"AssessLiquidStatus Prompt: {prompt}")
        rsp = await self._aask_image(prompt, encoded_image)
        return rsp

    async def _aask_image(self, prompt: str, image_data: str) -> str:
        message = self.llm._user_msg_with_imgs(prompt, image_data)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class SpeakAloud(Action):
    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Topic: {topic}
    You are {name}, collaborating with {teammate_name} to achieve a successful liquid transport mission.
    ## IMAGE ANALYSIS RESULT
    The latest image analysis indicates: {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Based on the image analysis result and previous communication, provide feedback and suggest whether a speed reduction is necessary.
    Provide a concise response in no more than 80 words.
    """
    name: str = "SpeakAloud"

    async def run(self, context: str, assessment_result: str, name: str, teammate_name: str, topic: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, assessment_result=assessment_result, name=name, teammate_name=teammate_name, topic=topic)
        print(f"SpeakAloud Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp
    
class FinalDecision(Action):
    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Topic: {topic}
    You are {name}, collaborating with {teammate_name} to achieve a successful liquid transport mission.
    ## IMAGE ANALYSIS RESULT
    The latest image analysis indicates: {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Based on the image analysis result and previous communication, make a final decision: Is a speed reduction necessary? 
    Respond with either "yes" or "no".
    """
    name: str = "FinalDecision"

    async def run(self, context: str, assessment_result: str, name: str, teammate_name: str, topic: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, assessment_result=assessment_result, name=name, teammate_name=teammate_name, topic=topic)
        print(f"FinalDecision Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp

class EnvironmentRobot(Role):
    name: str = "EnvironmentRobot"
    profile: str = "Environment Observer"
    teammate_name: str = "LiquidRobot"
    aspect: str = "environment"
    peer_aspect: str = "liquid status"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessEnvironment(), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessEnvironment, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "liquid transport mission"  # 初始化topic

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessEnvironment):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, SpeakAloud):
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammate_name, topic=self.topic)
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
    
class LiquidRobot(Role):
    name: str = "LiquidRobot"
    profile: str = "Liquid Observer"
    teammate_name: str = "EnvironmentRobot"
    aspect: str = "liquid status"
    peer_aspect: str = "environment"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessLiquidStatus(), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessLiquidStatus, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "liquid transport mission"  # 初始化topic

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessLiquidStatus):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, SpeakAloud):
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammate_name, topic=self.topic)
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
    rospy.init_node('obstacle_honey_room', anonymous=True)

    # 记录开始时间
    start_time = time.time()
    
    env_robot = EnvironmentRobot()
    liquid_robot = LiquidRobot()

    team = Team()
    team.hire([env_robot, liquid_robot])
    team.invest(investment)
    team.run_project(idea, send_to="EnvironmentRobot")

    message_pool = []
    for _ in range(n_round):
        message_pool.append(await env_robot._act(AssessEnvironment(), message_pool))
        message_pool.append(await liquid_robot._act(AssessLiquidStatus(), message_pool))
        message_pool.append(await env_robot._act(SpeakAloud(), message_pool))
        message_pool.append(await liquid_robot._act(SpeakAloud(), message_pool))
        message_pool.append(await env_robot._act(SpeakAloud(), message_pool)) 
        message_pool.append(await liquid_robot._act(SpeakAloud(), message_pool)) 
        message_pool.append(await env_robot._act(FinalDecision(), message_pool)) 
        message_pool.append(await liquid_robot._act(FinalDecision(), message_pool)) 

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

def main(idea: str, investment: float = 3.0, n_round: int = 1):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cooperate(idea, investment, n_round))

if __name__ == "__main__":
    fire.Fire(main)
# rosrun slope_water_room four_round.py --idea "liquid transport mission"
