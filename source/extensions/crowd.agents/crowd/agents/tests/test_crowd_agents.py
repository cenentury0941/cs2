# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import omni.kit.test
import omni.kit.ui_test as ui_test

from crowd.agents.ai import RuleBasedDecisionEngine
from crowd.agents.agent import Agent
from crowd.agents.navigation import NavigationSystem
from crowd.agents.scene import SEAT_PATTERN, SpawnZone, point_in_spawn_horizontal, sample_in_spawn
from crowd.agents.ui import WINDOW_TITLE


class Test(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        pass

    async def tearDown(self):
        pass

    async def test_rule_based_decision_engine(self):
        brain = RuleBasedDecisionEngine()
        agent = Agent(
            agent_id="a1",
            prim_path="/World/CrowdAgents/A1",
            goal="Seat.003",
            assigned_seat="Seat.003",
        )

        self.assertEqual(brain.decide(agent), "GO_TO_SEAT")

        agent.perception["at_destination"] = True
        self.assertEqual(brain.decide(agent), "IDLE")

        agent.perception["fire_visible"] = True
        self.assertEqual(brain.decide(agent), "EXIT")

    async def test_seat_name_pattern(self):
        self.assertIsNotNone(SEAT_PATTERN.match("Seat.000"))
        self.assertIsNotNone(SEAT_PATTERN.match("Seat.024"))
        self.assertIsNone(SEAT_PATTERN.match("Seat_000"))
        self.assertIsNone(SEAT_PATTERN.match("Seat.24"))
        self.assertEqual(int(SEAT_PATTERN.match("Seat.007").group(1)), 7)

    async def test_spawn_xy_bounds_z_up(self):
        zone = SpawnZone(
            prim_path="/World/Spawn",
            min_corner=(0.0, 0.0, -0.1),
            max_corner=(4.0, 2.0, 0.1),
        )
        for _ in range(20):
            p = sample_in_spawn(zone)
            self.assertGreaterEqual(p[0], 0.0)
            self.assertLessEqual(p[0], 4.0)
            self.assertGreaterEqual(p[1], 0.0)
            self.assertLessEqual(p[1], 2.0)
            self.assertTrue(point_in_spawn_horizontal(p, zone))

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

    async def test_window_smoke_buttons_exist(self):
        spawn = ui_test.find(
            f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Spawn Agents (idle on Spawn)'"
        )
        run = ui_test.find(
            f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Run (to seats)'"
        )
        self.assertIsNotNone(spawn)
        self.assertIsNotNone(run)
