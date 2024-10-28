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
    - Weak signs of life have been detected in the ruins.
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


class EvaluatePeer(Action):
    PROMPT_TEMPLATE: str = """
    ## CONTEXT
    You have analyzed the {aspect} and your peer has analyzed {peer_aspect}.
    ## YOUR ANALYSIS RESULT
    {self_result}
    ## PEER ANALYSIS RESULT
    {peer_result}
    ## TASK
    Evaluate your peer's analysis considering your own. Are there any inconsistencies or agreements? Provide feedback and suggest whether to proceed deeper into the ruins.
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
    ## YOUR INITIAL ANALYSIS
    {self_result}
    ## PEER FEEDBACK
    {peer_feedback}
    ## TASK
    Reflect on the feedback and make a final decision: Should the team proceed deeper into the ruins? Respond with either "yes" or "no".
    """
    name: str = "ReflectAndDecide"

    async def run(self, self_result: str, peer_feedback: str, aspect: str):
        prompt = self.PROMPT_TEMPLATE.format(self_result=self_result, peer_feedback=peer_feedback, aspect=aspect)
        print(f"ReflectAndDecide Prompt: {prompt}")
        rsp = await self._aask(prompt)
        return rsp

    async def _aask(self, prompt: str) -> str:
        message = self.llm._user_msg(prompt)
        rsp = await self.llm.acompletion_text([message], stream=False, timeout=self.llm.get_timeout(None))
        return rsp


class HumanoidRobot(Role):
    name: str = "HumanoidRobot"
    profile: str = "Debris Cleaner"
    teammate_name: str = "RobotDog"
    aspect: str = "debris"
    peer_aspect: str = "survivor"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessDebris(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessDebris, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessDebris):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessSurvivor"][0]
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

        print(f"{self.name} to {self.teammate_name}: {rsp}")
        return msg


class RobotDog(Role):
    name: str = "RobotDog"
    profile: str = "Survivor Seeker"
    teammate_name: str = "HumanoidRobot"
    aspect: str = "survivor"
    peer_aspect: str = "debris"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([AssessSurvivor(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, AssessSurvivor, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == self.name]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:
        logger.info(f"{self.name}: to do {todo.name}")

        if isinstance(todo, AssessSurvivor):
            rsp = await todo.run()
            self.assessment_result = rsp
        elif isinstance(todo, EvaluatePeer):
            peer_result = [msg.content for msg in message_pool if msg.sent_from == self.teammate_name and msg.cause_by == "__main__.AssessDebris"][0]
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

        print(f"{self.name} to {self.teammate_name}: {rsp}")
        return msg


class Arbiter:
    def __init__(self):
        self.weights = {"HumanoidRobot": 0.5, "RobotDog": 0.5}

    def arbitrate(self, humanoid_decision: str, dog_decision: str) -> str:
        humanoid_score = 1 if "yes" in humanoid_decision.lower() else 0
        dog_score = 1 if "yes" in dog_decision.lower() else 0

        final_score = (humanoid_score * self.weights["HumanoidRobot"]) + (dog_score * self.weights["RobotDog"])
        print(f"Current Weights: HumanoidRobot={self.weights['HumanoidRobot']}, RobotDog={self.weights['RobotDog']}")
        print(f"Final Score: {final_score}")

        if final_score > 0.5:
            return "yes"  # 表示可以继续搜救
        else:
            return "no"  # 表示不可以继续搜救

async def cooperate(idea: str, investment: float = 3.0, n_round: int = 2):
    rospy.init_node('joint_rescue_mission', anonymous=True)

    start_time = time.time()

    humanoid_robot = HumanoidRobot()
    robot_dog = RobotDog()
    arbiter = Arbiter()

    team = Team()
    team.hire([humanoid_robot, robot_dog])
    team.invest(investment)
    team.run_project(idea, send_to="HumanoidRobot")

    message_pool = []
    for _ in range(n_round):
        assess_tasks = [
            humanoid_robot._act(AssessDebris(), message_pool),
            robot_dog._act(AssessSurvivor(), message_pool)
        ]
        humanoid_assess_result, dog_assess_result = await asyncio.gather(*assess_tasks)
        message_pool.extend([humanoid_assess_result, dog_assess_result])

        evaluate_tasks = [
            humanoid_robot._act(EvaluatePeer(), message_pool),
            robot_dog._act(EvaluatePeer(), message_pool)
        ]
        humanoid_evaluate_result, dog_evaluate_result = await asyncio.gather(*evaluate_tasks)
        message_pool.extend([humanoid_evaluate_result, dog_evaluate_result])

        decide_tasks = [
            humanoid_robot._act(ReflectAndDecide(), message_pool),
            robot_dog._act(ReflectAndDecide(), message_pool)
        ]
        humanoid_decide_result, dog_decide_result = await asyncio.gather(*decide_tasks)
        message_pool.extend([humanoid_decide_result, dog_decide_result])

        humanoid_decision = message_pool[-2].content.strip()
        dog_decision = message_pool[-1].content.strip()

        final_decision = arbiter.arbitrate(humanoid_decision, dog_decision)
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

# rosrun mixed_debris_few_persons main.py --idea "joint rescue mission"
