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

class AssessGroundRobots(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for observing the cleaning robots from a ground-level perspective during a navigation task.

    Additional Information:
    - There are three cleaning robots in front of you.
    - The cleaning robots are moving fast.
    - Humanoid robots take up twice as much space as cleaning robots.

    Analyze the image and additional information. Provide your judgment on whether the humanoid robot can pass through without being obstructed.
    Provide a concise response in no more than 50 words.
    """

    name: str = "AssessGroundRobots"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_h1", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)
        
        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"AssessGroundRobots Prompt: {prompt}")  # 打印生成的Prompt内容
        rsp = await self._aask_image(prompt, encoded_image)
        return rsp

    async def _aask_image(self, prompt: str, image_data: str) -> str:
        message = self.llm._user_msg_with_imgs(prompt, image_data)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp


class AssessAerialView(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for observing the cleaning robots from an aerial perspective during a navigation task.

    Additional Information:
    - You have a clear view of three cleaning robots from above.
    - The cleaning robots are moving fast.
    - Humanoid robots take up twice as much space as cleaning robots.
    - The humanoid robot moves from the left side of the image to the right side.
    
    Analyze the image and provide your judgment on whether there is enough space for the humanoid robot to pass through.
    Provide a concise response in no more than 50 words.
    """

    name: str = "AssessAerialView"

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
        print(f"AssessAerialView Prompt: {prompt}")  # 打印生成的Prompt内容
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
    You are {name}, collaborating with {teammate_name} to achieve a successful navigation mission.
    ## IMAGE ANALYSIS RESULT
    The latest image analysis indicates: {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Based on the image analysis result and previous communication, provide feedback and suggest whether it is safe for humanoid robot to proceed.
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
    You are {name}, collaborating with {teammate_name} to perform a joint rescue mission.
    ## IMAGE ANALYSIS RESULT
    The latest image analysis indicates: {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Based on the image analysis result and previous communication, make a final decision: Is it safe to proceed?
    Respond with either "yes" or "no".
    """
    name: str = "FinalDecision"

    async def run(self, context: str, assessment_result: str, name: str, teammate_name: str, topic: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, assessment_result=assessment_result, name=name, teammate_name=teammate_name, topic=topic)
        print(f"FinalDecision Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp


class GroundRobot(Role):
    name: str = "GroundRobot"
    profile: str = "Ground-Level Observer"
    teammate_name: str = "AerialRobot"
    aspect: str = "ground view"
    peer_aspect: str = "aerial view"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessGroundRobots(), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessGroundRobots, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "navigate through cleaning robots"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessGroundRobots):
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


class AerialRobot(Role):
    name: str = "AerialRobot"
    profile: str = "Aerial Observer"
    teammate_name: str = "GroundRobot"
    aspect: str = "aerial view"
    peer_aspect: str = "ground view"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessAerialView(), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessAerialView, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "navigate through cleaning robots"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessAerialView):
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
    rospy.init_node('navigate_through_cleaning_robots', anonymous=True)

    start_time = time.time()

    ground_robot = GroundRobot()
    aerial_robot = AerialRobot()

    team = Team()
    team.hire([ground_robot, aerial_robot])
    team.invest(investment)
    team.run_project(idea, send_to="GroundRobot")

    message_pool = []
    for _ in range(n_round):
        message_pool.append(await ground_robot._act(AssessGroundRobots(), message_pool))
        message_pool.append(await aerial_robot._act(AssessAerialView(), message_pool))
        message_pool.append(await ground_robot._act(SpeakAloud(), message_pool))
        message_pool.append(await aerial_robot._act(SpeakAloud(), message_pool))
        message_pool.append(await ground_robot._act(SpeakAloud(), message_pool))
        message_pool.append(await aerial_robot._act(SpeakAloud(), message_pool))
        message_pool.append(await ground_robot._act(FinalDecision(), message_pool))
        message_pool.append(await aerial_robot._act(FinalDecision(), message_pool))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")


def main(idea: str, investment: float = 3.0, n_round: int = 1):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cooperate(idea, investment, n_round))


if __name__ == "__main__":
    fire.Fire(main)

# rosrun three_robots_fast four_round.py --idea "navigate through cleaning robots"
