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
        print(f"AssessEnvironment Prompt: {prompt}")  # 打印生成的Prompt内容
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
    - The liquid being transported is oil, which is a relatively viscous substance.
    - The robot's battery level is low.
    - There are other urgent transport tasks after this mission.
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
        print(f"AssessLiquidStatus Prompt: {prompt}")  # 打印生成的Prompt内容
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
    Evaluate your peer's analysis considering your own. Are there any inconsistencies or agreements? Provide feedback and suggest whether a speed reduction is necessary.
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
    ## PEER FEEDBACK
    {peer_feedback}
    ## TASK
    Reflect on the feedback and make a final decision: Is a speed reduction necessary? Respond with either "yes" or "no".
    """
    name: str = "ReflectAndDecide"

    async def run(self, peer_feedback: str, aspect: str):
        prompt = self.PROMPT_TEMPLATE.format(peer_feedback=peer_feedback, aspect=aspect)
        print(f"ReflectAndDecide Prompt: {prompt}")  # 打印生成的Prompt内容
        rsp = await self._aask(prompt)
        return rsp

    async def _aask(self, prompt: str) -> str:
        message = self.llm._user_msg(prompt)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp

class EnvironmentRobot(Role):
    name: str = "EnvironmentRobot"
    profile: str = "Environment Observer"
    teammate_name: str = "LiquidRobot"
    aspect: str = "environment"
    peer_aspect: str = "liquid status"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessEnvironment(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessEnvironment, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self._setting}: to do {todo.name}")

        if isinstance(todo, AssessEnvironment):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessLiquidStatus"][0]
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

        print(f"{self.name} to {self.teammate_name}: {rsp}")  # 打印对话内容到终端
        return msg

class LiquidRobot(Role):
    name: str = "LiquidRobot"
    profile: str = "Liquid Observer"
    teammate_name: str = "EnvironmentRobot"
    aspect: str = "liquid status"
    peer_aspect: str = "environment"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessLiquidStatus(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessLiquidStatus, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self._setting}: to do {todo.name}")

        if isinstance(todo, AssessLiquidStatus):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessEnvironment"][0]
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

        print(f"{self.name} to {self.teammate_name}: {rsp}")  # 打印对话内容到终端
        return msg
    
class Arbiter:
    def __init__(self):
        self.weights = {"EnvironmentRobot": 0.5, "LiquidRobot": 0.5}

    def arbitrate(self, env_decision: str, liq_decision: str) -> str:
        # 判断字符串中是否包含 "yes" 或 "no"
        env_score = 1 if "yes" in env_decision.lower() else 0
        liq_score = 1 if "yes" in liq_decision.lower() else 0

        # 计算最终的加权分数
        final_score = (env_score * self.weights["EnvironmentRobot"]) + (liq_score * self.weights["LiquidRobot"])
        print(f"Current Weights: EnvironmentRobot={self.weights['EnvironmentRobot']}, LiquidRobot={self.weights['LiquidRobot']}")
        print(f"Final Score: {final_score}")

        # 根据最终得分决定是否减速
        if final_score >= 0.5:
            return "yes"  # 表示需要减速
        else:
            return "no"  # 表示不需要减速


async def cooperate(idea: str, investment: float = 3.0, n_round: int = 2):
    rospy.init_node('obstacle_honey_room', anonymous=True)

    # 记录开始时间
    start_time = time.time()
    
    env_robot = EnvironmentRobot()
    liquid_robot = LiquidRobot()
    arbiter = Arbiter()

    team = Team()
    team.hire([env_robot, liquid_robot])
    team.invest(investment)
    team.run_project(idea, send_to="EnvironmentRobot")

    message_pool = []
    for _ in range(n_round):
        # 并行执行AssessEnvironment和AssessLiquidStatus
        assess_tasks = [
            env_robot._act(AssessEnvironment(), message_pool),  # EnvironmentRobot AssessEnvironment
            liquid_robot._act(AssessLiquidStatus(), message_pool)  # LiquidRobot AssessLiquidStatus
        ]
        env_assess_result, liq_assess_result = await asyncio.gather(*assess_tasks)
        message_pool.extend([env_assess_result, liq_assess_result])

        # 并行执行EvaluatePeer
        evaluate_tasks = [
            env_robot._act(EvaluatePeer(), message_pool),  # EnvironmentRobot EvaluatePeer
            liquid_robot._act(EvaluatePeer(), message_pool)  # LiquidRobot EvaluatePeer
        ]
        env_evaluate_result, liq_evaluate_result = await asyncio.gather(*evaluate_tasks)
        message_pool.extend([env_evaluate_result, liq_evaluate_result])

        # 并行执行ReflectAndDecide
        decide_tasks = [
            env_robot._act(ReflectAndDecide(), message_pool),  # EnvironmentRobot ReflectAndDecide
            liquid_robot._act(ReflectAndDecide(), message_pool)  # LiquidRobot ReflectAndDecide
        ]
        env_decide_result, liq_decide_result = await asyncio.gather(*decide_tasks)
        message_pool.extend([env_decide_result, liq_decide_result])

        # 仲裁机制
        env_decision = message_pool[-2].content.strip()
        liq_decision = message_pool[-1].content.strip()

        final_decision = arbiter.arbitrate(env_decision, liq_decision)
        print(f"Final Decision: {final_decision}")

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
# rosrun slope_oil_room main.py --idea "liquid transport mission"