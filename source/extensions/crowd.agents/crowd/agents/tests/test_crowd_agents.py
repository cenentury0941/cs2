# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import omni.kit.test
import omni.kit.ui_test as ui_test

from crowd.agents.ai import RuleBasedDecisionEngine
from crowd.agents.agent import Agent
from crowd.agents.navigation import NavigationSystem
from crowd.agents.ui import WINDOW_TITLE


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_rule_based_decision_engine(self):
        brain = RuleBasedDecisionEngine()
        agent = Agent(agent_id="a1", prim_path="/World/CrowdAgents/A1", goal="Seat_1")

        self.assertEqual(brain.decide(agent), "GO_TO_SEAT")

        agent.perception["at_destination"] = True
        self.assertEqual(brain.decide(agent), "IDLE")

        agent.perception["fire_visible"] = True
        self.assertEqual(brain.decide(agent), "EXIT")

    async def test_path_follow_steps_toward_waypoint(self):
        nav = NavigationSystem()
        agent = Agent(
            agent_id="walker",
            prim_path="/World/CrowdAgents/Walker",
            position=(0.0, 0.0, 0.0),
            height=1.7,
        )
        agent.waypoints = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        agent.waypoint_index = 1
        agent.speed_mps = 1.0

        nav.apply_intent(agent, "WALK", dt=0.5)
        self.assertGreater(agent.position[0], 0.0)
        self.assertLess(agent.position[0], 2.0)
        self.assertFalse(agent.perception.get("at_destination"))
        # Feet stay on ground; visual center is lifted
        center = agent.center_position()
        self.assertAlmostEqual(center[1], 0.85, places=2)

    async def test_window_smoke_buttons_exist(self):
        spawn = ui_test.find(f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Spawn Agent'")
        dest = ui_test.find(
            f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Set Random Destination'"
        )
        self.assertIsNotNone(spawn)
        self.assertIsNotNone(dest)
