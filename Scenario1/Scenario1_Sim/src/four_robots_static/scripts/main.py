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
    - There are four cleaning robots in front of you.
    - The cleaning robots are static.
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
    - You have a clear view of four cleaning robots from above.
    - The cleaning robots are static.
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

class EvaluatePeer(Action):
    PROMPT_TEMPLATE: str = """
    ## CONTEXT
    You have analyzed the {aspect} and your peer has analyzed {peer_aspect}.
    ## YOUR ANALYSIS RESULT
    {self_result}
    ## PEER ANALYSIS RESULT
    {peer_result}
    ## TASK
    Evaluate your peer's analysis considering your own. Are there any inconsistencies or agreements? Provide feedback and suggest whether it is safe for humanoid robot to proceed.

    Provide a concise response in no more than 80 words.
    """
    name: str = "EvaluatePeer"

    async def run(self, self_result: str, peer_result: str, aspect: str, peer_aspect: str):
        prompt = self.PROMPT_TEMPLATE.format(self_result=self_result, peer_result=peer_result, aspect=aspect, peer_aspect=peer_aspect)
        print(f"EvaluatePeer Prompt: {prompt}")  # 打印生成的Prompt内容
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
    ## INITIAL JUDGEMENT
    {self_result}
    ## PEER FEEDBACK
    {peer_feedback}
    ## TASK
    Based on your initial juedgement and the feedback. Make a final decision: Is it safe to proceed?
    Respond with either "yes" or "no".
    """
    name: str = "ReflectAndDecide"

    async def run(self, self_result: str, peer_feedback: str, aspect: str):
        prompt = self.PROMPT_TEMPLATE.format(self_result=self_result, peer_feedback=peer_feedback, aspect=aspect)
        print(f"ReflectAndDecide Prompt: {prompt}")  # 打印生成的Prompt内容
        rsp = await self._aask(prompt)
        return rsp

    async def _aask(self, prompt: str) -> str:
        message = self.llm._user_msg(prompt)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class GroundRobot(Role):
    name: str = "GroundRobot"
    profile: str = "Ground-Level Observer"
    teammate_name: str = "AerialRobot"
    aspect: str = "ground view"
    peer_aspect: str = "aerial view"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessGroundRobots(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessGroundRobots, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self._setting}: to do {todo.name}")

        if isinstance(todo, AssessGroundRobots):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessAerialView"][0]
            rsp = await todo.run(self_result=self.assessment_result, peer_result=peer_result, aspect=self.aspect, peer_aspect=self.peer_aspect)
        else:  # ReflectAndDecide
            peer_feedback = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.EvaluatePeer"][0]
            rsp = await todo.run(self_result=self.assessment_result, peer_feedback=peer_feedback, aspect=self.aspect)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammate_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammate_name}: {rsp}")  # 打印对话内容到终端
        return msg

class AerialRobot(Role):
    name: str = "AerialRobot"
    profile: str = "Aerial Observer"
    teammate_name: str = "GroundRobot"
    aspect: str = "aerial view"
    peer_aspect: str = "ground view"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessAerialView(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessAerialView, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self._setting}: to do {todo.name}")

        if isinstance(todo, AssessAerialView):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessGroundRobots"][0]
            rsp = await todo.run(self_result=self.assessment_result, peer_result=peer_result, aspect=self.aspect, peer_aspect=self.peer_aspect)
        else:  # ReflectAndDecide
            peer_feedback = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.EvaluatePeer"][0]
            rsp = await todo.run(self_result=self.assessment_result, peer_feedback=peer_feedback, aspect=self.aspect)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammate_name,
        )
        self.rc.memory.add(msg)

        print(f"{self.name} to {self.teammate_name}: {rsp}")  # 打印对话内容到终端
        return msg
    
class Arbiter:
    def __init__(self):
        self.weights = {"GroundRobot": 0.5, "AerialRobot": 0.5}

    def arbitrate(self, ground_decision: str, aerial_decision: str) -> str:
        ground_score = 1 if "yes" in ground_decision.lower() else 0
        aerial_score = 1 if "yes" in aerial_decision.lower() else 0

        final_score = (ground_score * self.weights["GroundRobot"]) + (aerial_score * self.weights["AerialRobot"])
        print(f"Current Weights: GroundRobot={self.weights['GroundRobot']}, AerialRobot={self.weights['AerialRobot']}")
        print(f"Final Score: {final_score}")

        if final_score > 0.5:
            return "yes"  # 表示可以通过
        else:
            return "no"  # 表示不能通过

async def cooperate(idea: str, investment: float = 3.0, n_round: int = 2):
    rospy.init_node('navigate_through_cleaning_robots', anonymous=True)

    start_time = time.time()
    
    ground_robot = GroundRobot()
    aerial_robot = AerialRobot()
    arbiter = Arbiter()

    team = Team()
    team.hire([ground_robot, aerial_robot])
    team.invest(investment)
    team.run_project(idea, send_to="GroundRobot")

    message_pool = []
    for _ in range(n_round):
        # 并行执行AssessGroundRobots和AssessAerialView
        assess_tasks = [
            ground_robot._act(AssessGroundRobots(), message_pool),
            aerial_robot._act(AssessAerialView(), message_pool)
        ]
        ground_assess_result, aerial_assess_result = await asyncio.gather(*assess_tasks)
        message_pool.extend([ground_assess_result, aerial_assess_result])

        # 并行执行EvaluatePeer
        evaluate_tasks = [
            ground_robot._act(EvaluatePeer(), message_pool),
            aerial_robot._act(EvaluatePeer(), message_pool)
        ]
        ground_evaluate_result, aerial_evaluate_result = await asyncio.gather(*evaluate_tasks)
        message_pool.extend([ground_evaluate_result, aerial_evaluate_result])

        # 并行执行ReflectAndDecide
        decide_tasks = [
            ground_robot._act(ReflectAndDecide(), message_pool),
            aerial_robot._act(ReflectAndDecide(), message_pool)
        ]
        ground_decide_result, aerial_decide_result = await asyncio.gather(*decide_tasks)
        message_pool.extend([ground_decide_result, aerial_decide_result])

        # 仲裁机制
        ground_decision = message_pool[-2].content.strip()
        aerial_decision = message_pool[-1].content.strip()

        final_decision = arbiter.arbitrate(ground_decision, aerial_decision)
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

# rosrun four_robots_static main.py --idea "navigate through cleaning robots"