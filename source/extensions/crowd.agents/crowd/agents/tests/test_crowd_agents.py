# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import omni.kit.test
import omni.kit.ui_test as ui_test

from crowd.agents.ai import RuleBasedDecisionEngine
from crowd.agents.agent import Agent
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

    async def test_window_smoke_buttons_exist(self):
        spawn = ui_test.find(f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Spawn Cube'")
        move = ui_test.find(f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Move Cube +1m'")
        self.assertIsNotNone(spawn)
        self.assertIsNotNone(move)
