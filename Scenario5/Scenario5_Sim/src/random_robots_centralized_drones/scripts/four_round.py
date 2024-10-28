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

class MonitorTargets(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for monitoring the humanoid robots in {scene}.

    Additional Information:
    - The humanoid robots are black.
    - Scene1 is sub-region1, scene2 is sub-region2, scene3 is sub-region3.
    - Your goal is to assess the number of humanoid robots in the scene.

    Analyze the image and provide your judgment on the number of humanoid robots in the scene.
    Provide a response in no more than 20 words.
    """

    name: str = "MonitorTargets"

    def __init__(self, scene: str):
        super().__init__()
        self.scene = scene
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self, topic):
        rospy.Subscriber(topic, Image, self.image_callback)

    async def run(self):
        topic = f"/rgb_{self.scene}"
        self.subscribe_to_camera(topic)
        while self.cv_image is None:
            await asyncio.sleep(0.1)

        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE.format(scene=self.scene)
        print(f"MonitorTargets Prompt: {prompt}")
        rsp = await self._aask_image(prompt, encoded_image)
        return rsp

    async def _aask_image(self, prompt: str, image_data: str) -> str:
        message = self.llm._user_msg_with_imgs(prompt, image_data)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class CommandDrones(Action):
    PROMPT_TEMPLATE: str = """
    You are responsible for commanding the drones in scene1 (left side of the image), scene2 (middle of the image), and scene3 (right side of the image).

    Additional Information:
    - The drones are white.
    - The drones in the image are divided into three clusters, and total number is 6.
    - Scene1 is sub-region1, scene2 is sub-region2, scene3 is sub-region3.
    - Your goal is to assess the number of drones in each scene.

    Analyze the image and provide your judgment on the drone distribution.
    Provide a concise response in no more than 20 words.
    """

    name: str = "CommandDrones"

    def __init__(self):
        super().__init__()
        self.bridge = CvBridge()
        self.cv_image = None

    def image_callback(self, data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def subscribe_to_camera(self):
        rospy.Subscriber("/rgb_master", Image, self.image_callback)

    async def run(self):
        self.subscribe_to_camera()
        while self.cv_image is None:
            await asyncio.sleep(0.1)

        _, buffer = cv2.imencode('.jpg', self.cv_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        prompt = self.PROMPT_TEMPLATE
        print(f"CommandDrones Prompt: {prompt}")
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
    You are {name}, collaborating with {teammate_name} to achieve a successful humanoid robots monitoring mission.
    ## YOUR ANALYSIS RESULT
    {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## REQUIRMENT
    1. At least one drone is required in each sub-region and the total number of drones cannot be changed.
    2. Under the premise of ensuring Condition 1, where there are more humanoid robots, there should be more drones
    ## YOUR TURN
    Based on the your analysis result, previous communication and requirements, provide feedback and suggest whether the drone distribution needs to be adjusted based on humanoid robots distribution.
    Provide a concise response in no more than 30 words.
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
    You are {name}, collaborating with {teammate_name} to monitor humanoid robots.
    ## YOUR ANALYSIS RESULT
    {assessment_result}
    ## COMMUNICATION HISTORY
    Previous rounds:
    {context}
    ## REQUIRMENT
    1. At least one drone is required in each sub-region and the total number of drones cannot be changed.
    2. Under the premise of ensuring Condition 1, where there are more humanoid robots, there should be more drones
    ## YOUR TURN
    Based on your analysis result, previous communication and requirements, make a final decision: Should the drone distribution needs to be adjusted? 
    Respond with either "yes" or "no".
    """
    name: str = "FinalDecision"

    async def run(self, context: str, assessment_result: str, name: str, teammate_name: str, topic: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, assessment_result=assessment_result, name=name, teammate_name=teammate_name, topic=topic)
        print(f"FinalDecision Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp


class SubRegion1Drone(Role):
    name: str = "SubRegion1Drone"
    profile: str = "Sub-Region Monitor"
    teammates_name: list = ["SubRegion2Drone", "SubRegion3Drone", "CommandDrone"]
    aspect: str = "humanoid robots in sub-region1"
    peer_aspect: list = ["humanoid robots in sub-region2", "humanoid robots in sub-region3", "drones' distribution"]

    def __init__(self, scene: str, **data: Any):
        super().__init__(**data)
        self.scene = scene
        self.set_actions([MonitorTargets(scene), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, MonitorTargets, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "humanoid robots monitor mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, MonitorTargets):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, SpeakAloud):
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)
        else:  # FinalDecision
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammates_name}: {rsp}")
        return msg

class SubRegion2Drone(Role):
    name: str = "SubRegion2Drone"
    profile: str = "Sub-Region Monitor"
    teammates_name: list = ["SubRegion1Drone", "SubRegion3Drone", "CommandDrone"]
    aspect: str = "humanoid robots in sub-region1"
    peer_aspect: list = ["humanoid robots in sub-region1", "humanoid robots in sub-region3", "drones' distribution"]

    def __init__(self, scene: str, **data: Any):
        super().__init__(**data)
        self.scene = scene
        self.set_actions([MonitorTargets(scene), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, MonitorTargets, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "humanoid robots monitor mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, MonitorTargets):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, SpeakAloud):
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)
        else:  # FinalDecision
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammates_name}: {rsp}")
        return msg

class SubRegion3Drone(Role):
    name: str = "SubReg3on1Drone"
    profile: str = "Sub-Region Monitor"
    teammates_name: list = ["SubRegion1Drone", "SubRegion2Drone", "CommandDrone"]
    aspect: str = "humanoid robots in sub-region3"
    peer_aspect: list = ["humanoid robots in sub-region1", "humanoid robots in sub-region2", "drones' distribution"]

    def __init__(self, scene: str, **data: Any):
        super().__init__(**data)
        self.scene = scene
        self.set_actions([MonitorTargets(scene), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, MonitorTargets, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "humanoid robots monitor mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, MonitorTargets):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, SpeakAloud):
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)
        else:  # FinalDecision
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammates_name}: {rsp}")
        return msg

class CommandDrone(Role):
    name: str = "CommandDrone"
    profile: str = "Drone Commander"
    teammates_name: list = ["SubRegion1Drone", "SubRegion2Drone", "SubRegion3Drone"]
    aspect: str = "drones' distribution"
    peer_aspect: list = ["humanoid robots in sub-region1", "humanoid robots in sub-region2", "humanoid robots in sub-region3"]

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([CommandDrones(), SpeakAloud(), FinalDecision()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, CommandDrones, SpeakAloud, FinalDecision])
        self.assessment_result = ""
        self.topic = "humanoid robots monitor mission"

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, CommandDrones):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, SpeakAloud):
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)
        else:  # FinalDecision
            context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in message_pool)
            rsp = await todo.run(context=context, assessment_result=self.assessment_result, name=self.name, teammate_name=self.teammates_name, topic=self.topic)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammates_name}: {rsp}")
        return msg


async def cooperate(idea: str, investment: float = 3.0, n_round: int = 2):
    rospy.init_node('humanoid_robot_monitor_mission', anonymous=True)

    start_time = time.time()

    subregion1_drone = SubRegion1Drone("scene1")
    subregion2_drone = SubRegion2Drone("scene2")
    subregion3_drone = SubRegion3Drone("scene3")
    command_drone = CommandDrone()

    team = Team()
    team.hire([subregion1_drone, subregion2_drone, subregion3_drone, command_drone])
    team.invest(investment)
    team.run_project(idea, send_to="SubRegion1Drone")

    message_pool = []
    for _ in range(n_round):
        message_pool.append(await subregion1_drone._act(MonitorTargets("scene1"), message_pool))
        message_pool.append(await subregion2_drone._act(MonitorTargets("scene2"), message_pool))
        message_pool.append(await subregion3_drone._act(MonitorTargets("scene3"), message_pool))
        message_pool.append(await command_drone._act(CommandDrones(), message_pool))
        message_pool.append(await subregion1_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await subregion2_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await subregion3_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await command_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await subregion1_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await subregion2_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await subregion3_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await command_drone._act(SpeakAloud(), message_pool))
        message_pool.append(await subregion1_drone._act(FinalDecision(), message_pool))
        message_pool.append(await subregion2_drone._act(FinalDecision(), message_pool))
        message_pool.append(await subregion3_drone._act(FinalDecision(), message_pool))
        message_pool.append(await command_drone._act(FinalDecision(), message_pool))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")


def main(idea: str, investment: float = 3.0, n_round: int = 1):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cooperate(idea, investment, n_round))


if __name__ == "__main__":
    fire.Fire(main)

# rosrun random_robots_centralized_drones four_round.py --idea 'humanoid robots monitor mission'
