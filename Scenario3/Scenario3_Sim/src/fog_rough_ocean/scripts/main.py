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
    - The operation is taking place in the fog.
    - The docking platform has just been serviced and confirmed to be in good working condition.

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

class EvaluatePeer(Action):
    PROMPT_TEMPLATE: str = """
    ## CONTEXT
    You have analyzed the {aspect} and your peer has analyzed {peer_aspect}.
    ## YOUR ANALYSIS RESULT
    {self_result}
    ## PEER ANALYSIS RESULT
    {peer_result}
    ## TASK
    Evaluate your peer's analysis considering your own. Are there any inconsistencies or agreements? Provide feedback and suggest whether docking is advisable.
    Provide a concise response in no more than 80 words.
    """
    name: str = "EvaluatePeer"

    async def run(self, self_result: str, peer_result: str, aspect: str, peer_aspect: str):
        prompt = self.PROMPT_TEMPLATE.format(self_result=self_result, peer_result=peer_result, aspect=aspect, peer_aspect=peer_aspect)
        print(f"EvaluatePeer Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp

    async def _aask(self, prompt: str) -> str:
        message = self.llm._user_msg(prompt)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class ReflectAndDecide(Action):
    PROMPT_TEMPLATE: str = """
    ## CONTEXT
    You have received feedback from your peer based on your initial analysis of the {aspect}.
    ## PEER FEEDBACK
    {peer_feedback}
    ## TASK
    Reflect on the feedback and make a final decision: Is docking advisable? Respond with either "yes" or "no".
    """
    name: str = "ReflectAndDecide"

    async def run(self, peer_feedback: str, aspect: str):
        prompt = self.PROMPT_TEMPLATE.format(peer_feedback=peer_feedback, aspect=aspect)
        print(f"ReflectAndDecide Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp

    async def _aask(self, prompt: str) -> str:
        message = self.llm._user_msg(prompt)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class DroneRobot(Role):
    name: str = "DroneRobot"
    profile: str = "Drone Observer"
    teammate_name: str = "PlatformRobot"
    aspect: str = "platform"
    peer_aspect: str = "drone"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessPlatformFromDrone(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessPlatformFromDrone, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self._setting}: to do {todo.name}")

        if isinstance(todo, AssessPlatformFromDrone):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessDroneFromPlatform"][0]
            rsp = await todo.run(self_result=self.assessment_result, peer_result=peer_result, aspect=self.aspect, peer_aspect=self.peer_aspect)
        else:  # ReflectAndDecide
            peer_feedback = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.EvaluatePeer"][0]
            rsp = await todo.run(peer_feedback=peer_feedback, aspect=self.aspect)

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
        self.set_actions([AssessDroneFromPlatform(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessDroneFromPlatform, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self._setting}: to do {todo.name}")

        if isinstance(todo, AssessDroneFromPlatform):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessPlatformFromDrone"][0]
            rsp = await todo.run(self_result=self.assessment_result, peer_result=peer_result, aspect=self.aspect, peer_aspect=self.peer_aspect)
        else:  # ReflectAndDecide
            peer_feedback = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.EvaluatePeer"][0]
            rsp = await todo.run(peer_feedback=peer_feedback, aspect=self.aspect)

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

class Arbiter:
    def __init__(self):
        self.weights = {"DroneRobot": 0.5, "PlatformRobot": 0.5}

    def arbitrate(self, drone_decision: str, platform_decision: str) -> str:
        drone_score = 1 if "yes" in drone_decision.lower() else 0
        platform_score = 1 if "yes" in platform_decision.lower() else 0

        final_score = (drone_score * self.weights["DroneRobot"]) + (platform_score * self.weights["PlatformRobot"])
        print(f"Current Weights: DroneRobot={self.weights['DroneRobot']}, PlatformRobot={self.weights['PlatformRobot']}")
        print(f"Final Score: {final_score}")

        if final_score > 0.5:
            return "yes"  # 表示可以对接
        else:
            return "no"  # 表示不可以对接

async def cooperate(idea: str, investment: float = 3.0, n_round: int = 2):
    rospy.init_node('drone_platform_docking', anonymous=True)

    start_time = time.time()
    
    drone_robot = DroneRobot()
    platform_robot = PlatformRobot()
    arbiter = Arbiter()

    team = Team()
    team.hire([drone_robot, platform_robot])
    team.invest(investment)
    team.run_project(idea, send_to="DroneRobot")

    message_pool = []
    for _ in range(n_round):
        assess_tasks = [
            drone_robot._act(AssessPlatformFromDrone(), message_pool),
            platform_robot._act(AssessDroneFromPlatform(), message_pool)
        ]
        drone_assess_result, platform_assess_result = await asyncio.gather(*assess_tasks)
        message_pool.extend([drone_assess_result, platform_assess_result])

        evaluate_tasks = [
            drone_robot._act(EvaluatePeer(), message_pool),
            platform_robot._act(EvaluatePeer(), message_pool)
        ]
        drone_evaluate_result, platform_evaluate_result = await asyncio.gather(*evaluate_tasks)
        message_pool.extend([drone_evaluate_result, platform_evaluate_result])

        decide_tasks = [
            drone_robot._act(ReflectAndDecide(), message_pool),
            platform_robot._act(ReflectAndDecide(), message_pool)
        ]
        drone_decide_result, platform_decide_result = await asyncio.gather(*decide_tasks)
        message_pool.extend([drone_decide_result, platform_decide_result])

        drone_decision = message_pool[-2].content.strip()
        platform_decision = message_pool[-1].content.strip()

        final_decision = arbiter.arbitrate(drone_decision, platform_decision)
        print(f"Final Decision: {final_decision}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

def main(idea: str, investment: float = 3.0, n_round: int = 1):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cooperate(idea, investment, n_round))

if __name__ == "__main__":
    fire.Fire(main)
# rosrun fog_rough_ocean main.py --idea "drone docking mission"
