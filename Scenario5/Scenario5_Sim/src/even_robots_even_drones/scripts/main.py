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

class EvaluatePeer(Action):
    PROMPT_TEMPLATE: str = """
    ## CONTEXT
    You have analyzed the {aspect}.
    Scene1 is sub-region1, scene2 is sub-region2, scene3 is sub-region3.
    ## YOUR ANALYSIS RESULT
    {self_result}
    ## ALL PEERS' ANALYSIS RESULT
    {peer_results}
    ## REQUIRMENT
    1. At least one drone is required in each sub-region and the total number of drones cannot be changed.
    2. Under the premise of ensuring Condition 1, where there are more humanoid robots, there should be more drones
    ## TASK
    Based on the self analysis, peers' analysis and requirments, provide feedback and suggest whether the drone distribution needs to be adjusted based on humanoid robots distribution.
    Provide a concise response in no more than 30 words.
    """
    name: str = "EvaluatePeer"

    async def run(self, self_result: str, peer_results: str, aspect: str, peer_aspect: list):
        prompt = self.PROMPT_TEMPLATE.format(self_result=self_result, peer_results=peer_results, aspect=aspect, peer_aspect=peer_aspect)
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
    You have received feedback from your peers.
    Scene1 is sub-region1, scene2 is sub-region2, scene3 is sub-region3.
    ## SELF FEEDBACK
    {self_result}
    ## PEERS FEEDBACK
    {peer_feedback}
    ## REQUIRMENT
    1. At least one drone is required in each sub-region and the total number of drones cannot be changed.
    2. Under the premise of ensuring Condition 1, where there are more humanoid robots, there should be more drones
    ## TASK
    Based on the self feedback, peers' feedback and requirments, make a final decision: Should the drone distribution needs to be adjusted? 
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

class SubRegion1Drone(Role):
    name: str = "SubRegion1Drone"
    profile: str = "Sub-Region Monitor"
    teammates_name: list = ["SubRegion2Drone", "SubRegion3Drone", "CommandDrone"]
    aspect: str = "humanoid robots in sub-region1"
    peer_aspect: list = ["humanoid robots in sub-region2", "humanoid robots in sub-region3", "drones' distribution"]

    def __init__(self, scene: str, **data: Any):
        super().__init__(**data)
        self.scene = scene
        self.set_actions([MonitorTargets(scene), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, MonitorTargets, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:

        if isinstance(todo, MonitorTargets):
            logger.info(f"{self._setting}: to do {todo.name}")
            rsp = await todo.run()
            self.assessment_result = rsp

        elif isinstance(todo, EvaluatePeer):
            for self.teammate_name in self.teammates_name:
                logger.info(f"{self._setting}: to do {todo.name}")
                self_result = f"Sub-region1: " + self.assessment_result
                peer_results_list = [f"Sub-region2: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0],
                                     f"Sub-region3:  " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0],
                                     f"Drones distribution: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0]]
                peer_results = "\n".join(peer_results_list)
                rsp = await todo.run(self_result=self_result, peer_results=peer_results, aspect=self.aspect, peer_aspect=self.peer_aspect)

        else:  # ReflectAndDecide
            logger.info(f"{self._setting}: to do {todo.name}")
            self_result = f"{self.name} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.name and msg.cause_by == "__main__.EvaluatePeer"][0]
            peer_feedback_list = [f"{self.teammates_name[0]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[1]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[2]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by == "__main__.EvaluatePeer"][0]]
            peer_feedback = "\n".join(peer_feedback_list)
            rsp = await todo.run(self_result=self_result, peer_feedback=peer_feedback, aspect=self.aspect)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)
        print(f"{self.name} to {self.teammates_name}: {rsp}")  # 打印对话内容到终端
        return msg
    
class SubRegion2Drone(Role):
    name: str = "SubRegion2Drone"
    profile: str = "Sub-Region Monitor"
    teammates_name: list = ["SubRegion1Drone", "SubRegion3Drone", "CommandDrone"]
    aspect: str = "humanoid robots in sub-region2"
    peer_aspect: list = ["humanoid robots in sub-region1", "humanoid robots in sub-region3", "drones' distribution"]

    def __init__(self, scene:str, **data: Any):
        super().__init__(**data)
        self.scene = scene
        self.set_actions([MonitorTargets(scene), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, MonitorTargets, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:

        if isinstance(todo, MonitorTargets):
            logger.info(f"{self._setting}: to do {todo.name}")
            rsp = await todo.run()
            self.assessment_result = rsp

        elif isinstance(todo, EvaluatePeer):
            for self.teammate_name in self.teammates_name:
                logger.info(f"{self._setting}: to do {todo.name}")
                peer_results_list = [f"Sub-region1: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0],
                                     f"Sub-region3: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0],
                                     f"Drones distribution: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0]]
                peer_results = "\n".join(peer_results_list)
                rsp = await todo.run(self_result=self.assessment_result, peer_results=peer_results, aspect=self.aspect, peer_aspect=self.peer_aspect)

        else:  # ReflectAndDecide
            logger.info(f"{self._setting}: to do {todo.name}")
            self_result = f"{self.name} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.name and msg.cause_by == "__main__.EvaluatePeer"][0]
            peer_feedback_list = [f"{self.teammates_name[0]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[1]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[2]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by == "__main__.EvaluatePeer"][0]]
            peer_feedback = "\n".join(peer_feedback_list)
            rsp = await todo.run(self_result=self_result, peer_feedback=peer_feedback, aspect=self.aspect)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)
        print(f"{self.name} to {self.teammates_name}: {rsp}")  # 打印对话内容到终端
        return msg
    
class SubRegion3Drone(Role):
    name: str = "SubRegion3Drone"
    profile: str = "Sub-Region Monitor"
    teammates_name: list = ["SubRegion1Drone", "SubRegion2Drone", "CommandDrone"]
    aspect: str = "humanoid robots in sub-region3"
    peer_aspect: list = ["humanoid robots in sub-region1", "humanoid robots in sub-region2", "drones' distribution"]

    def __init__(self, scene:str, **data: Any):
        super().__init__(**data)
        self.scene = scene
        self.set_actions([MonitorTargets(scene), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, MonitorTargets, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:

        if isinstance(todo, MonitorTargets):
            logger.info(f"{self._setting}: to do {todo.name}")
            rsp = await todo.run()
            self.assessment_result = rsp

        elif isinstance(todo, EvaluatePeer):
            for self.teammate_name in self.teammates_name:
                logger.info(f"{self._setting}: to do {todo.name}")
                peer_results_list = [f"Sub-region1: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0],
                                     f"Sub-region2: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0],
                                     f"Drones distribution: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by in ["__main__.MonitorTargets", "__main__.CommandDrones"]][0]]
                peer_results = "\n".join(peer_results_list)
                rsp = await todo.run(self_result=self.assessment_result, peer_results=peer_results, aspect=self.aspect, peer_aspect=self.peer_aspect)

        else:  # ReflectAndDecide
            logger.info(f"{self._setting}: to do {todo.name}")
            self_result = f"{self.name} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.name and msg.cause_by == "__main__.EvaluatePeer"][0]
            peer_feedback_list = [f"{self.teammates_name[0]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[1]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[2]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by == "__main__.EvaluatePeer"][0]]
            peer_feedback = "\n".join(peer_feedback_list)
            rsp = await todo.run(self_result=self_result, peer_feedback=peer_feedback, aspect=self.aspect)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)
        print(f"{self.name} to {self.teammates_name}: {rsp}")  # 打印对话内容到终端
        return msg

class CommandDrone(Role):
    name: str = "CommandDrone"
    profile: str = "Drone Commander"
    teammates_name: list = ["SubRegion1Drone", "SubRegion2Drone", "SubRegion3Drone"]
    aspect: str = "drones' distribution"
    peer_aspect: list = ["humanoid robots in sub-region1", "humanoid robots in sub-region2", "humanoid robots in sub-region3"]

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([CommandDrones(), EvaluatePeer(), ReflectAndDecide()])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)
        self._watch([UserRequirement, CommandDrones, EvaluatePeer, ReflectAndDecide])
        self.assessment_result = ""

    async def _observe(self) -> int:
        await super()._observe()
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)

    async def _act(self, todo: Action, message_pool: List[Message]) -> Message:

        if isinstance(todo, CommandDrones):
            logger.info(f"{self._setting}: to do {todo.name}")
            rsp = await todo.run()
            self.assessment_result = rsp

        elif isinstance(todo, EvaluatePeer):
            for self.teammate_name in self.teammates_name:
                logger.info(f"{self._setting}: to do {todo.name}")
                peer_results_list = [f"Sub-region1: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by == "__main__.MonitorTargets"][0],
                                     f"Sub-region2: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by == "__main__.MonitorTargets"][0],
                                     f"Sub-region3: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by == "__main__.MonitorTargets"][0]]
                peer_results = "\n".join(peer_results_list)
                rsp = await todo.run(self_result=self.assessment_result, peer_results=peer_results, aspect=self.aspect, peer_aspect=self.peer_aspect)

        else:  # ReflectAndDecide
            logger.info(f"{self._setting}: to do {todo.name}")
            self_result = f"{self.name} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.name and msg.cause_by == "__main__.EvaluatePeer"][0]
            peer_feedback_list = [f"{self.teammates_name[0]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[0] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[1]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[1] and msg.cause_by == "__main__.EvaluatePeer"][0],
                                  f"{self.teammates_name[2]} evaluate result: " + [msg.content for msg in message_pool if msg.sent_from == self.teammates_name[2] and msg.cause_by == "__main__.EvaluatePeer"][0]]
            peer_feedback = "\n".join(peer_feedback_list)
            rsp = await todo.run(self_result=self_result, peer_feedback=peer_feedback, aspect=self.aspect)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.teammates_name,
        )
        self.rc.memory.add(msg)
        print(f"{self.name} to {self.teammates_name}: {rsp}")  # 打印对话内容到终端
        return msg
    
class Arbiter:
    def __init__(self):
        self.weights = {"SubRegion1Drone": 0.25, "SubRegion2Drone": 0.25, "SubRegion3Drone": 0.25, "CommandDrone": 0.25}

    def arbitrate(self, subregion_decisions: List[str], command_decision: str) -> str:
        subregion_scores = [
            1 if "yes" in decision.lower() else 0
            for decision in subregion_decisions
        ]
        command_score = 1 if "yes" in command_decision.lower() else 0

        # 计算加权总分
        final_score = (
            (subregion_scores[0] * self.weights["SubRegion1Drone"]) +
            (subregion_scores[1] * self.weights["SubRegion2Drone"]) +
            (subregion_scores[2] * self.weights["SubRegion3Drone"]) +
            (command_score * self.weights["CommandDrone"])
        )

        print(f"Current Weights: {self.weights}")
        print(f"Final Score: {final_score}")

        if final_score > 0.5:
            return "yes"  # 表示可以通过
        else:
            return "no"  # 表示不能通过

async def cooperate(idea: str, investment: float = 3.0, n_round: int = 2):
    rospy.init_node('humanoid_robots_monitor_mission', anonymous=True)    

    start_time = time.time()

    subregion1_drone = SubRegion1Drone("scene1")
    subregion2_drone = SubRegion2Drone("scene2")
    subregion3_drone = SubRegion3Drone("scene3")
    command_drone = CommandDrone()
    arbiter = Arbiter()

    team = Team()
    team.hire([subregion1_drone, subregion2_drone, subregion3_drone, command_drone])
    team.invest(investment)
    team.run_project(idea, send_to="SubRegion1Drone")

    message_pool = []
    for _ in range(n_round):
        # 并行执行AssessGroundRobots和AssessAerialView
        assess_tasks = [
            subregion1_drone._act(MonitorTargets("scene1"), message_pool),
            subregion2_drone._act(MonitorTargets("scene2"), message_pool),
            subregion3_drone._act(MonitorTargets("scene3"), message_pool),
            command_drone._act(CommandDrones(), message_pool)
        ]
        subregion1_assess_result, subregion2_assess_result, subregion3_assess_result, command_assess_result = await asyncio.gather(*assess_tasks)
        message_pool.extend([subregion1_assess_result, subregion2_assess_result, subregion3_assess_result, command_assess_result])

        # 并行执行EvaluatePeer
        evaluate_tasks = [
            subregion1_drone._act(EvaluatePeer(), message_pool),
            subregion2_drone._act(EvaluatePeer(), message_pool),
            subregion3_drone._act(EvaluatePeer(), message_pool),
            command_drone._act(EvaluatePeer(), message_pool)
        ]
        subregion1_evaluate_result, subregion2_evaluate_result, subregion3_evaluate_result, command_evaluate_result = await asyncio.gather(*evaluate_tasks)
        message_pool.extend([subregion1_evaluate_result, subregion2_evaluate_result, subregion3_evaluate_result, command_evaluate_result])

        # 并行执行ReflectAndDecide
        decide_tasks = [
            subregion1_drone._act(ReflectAndDecide(), message_pool),
            subregion2_drone._act(ReflectAndDecide(), message_pool),
            subregion3_drone._act(ReflectAndDecide(), message_pool),
            command_drone._act(ReflectAndDecide(), message_pool)
        ]
        subregion1_decide_result, subregion2_decide_result, subregion3_decide_result, command_decide_result = await asyncio.gather(*decide_tasks)
        message_pool.extend([subregion1_decide_result, subregion2_decide_result, subregion3_decide_result, command_decide_result])

        # 仲裁机制
        subregion_decisions = [message_pool[-4].content.strip(), message_pool[-3].content.strip(), message_pool[-2].content.strip()]
        command_decision = message_pool[-1].content.strip()

        final_decision = arbiter.arbitrate(subregion_decisions, command_decision)
        print(f"Final Decision: {final_decision}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

def main(idea: str, investment: float = 10.0, n_round: int = 1):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(cooperate(idea, investment, n_round))

if __name__ == "__main__":
    fire.Fire(main)

# rosrun even_robots_even_drones main.py --idea 'humanoid robots monitor mission'