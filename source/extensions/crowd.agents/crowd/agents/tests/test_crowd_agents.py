# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import omni.kit.test
import omni.kit.ui_test as ui_test
from pxr import Gf

from crowd.agents.ai import RuleBasedDecisionEngine
from crowd.agents.agent import Agent
from crowd.agents.navigation import NavigationSystem
from crowd.agents.scene import SEAT_PATTERN, SpawnZone, point_in_spawn_horizontal, sample_in_spawn
from crowd.agents.ui import WINDOW_TITLE


def _make_zone(u0=0.0, u1=4.0, v0=0.0, v1=2.0) -> SpawnZone:
    return SpawnZone(
        prim_path="/World/Spawn",
        geom_path="/World/Spawn/Plane",
        local_min=(u0, v0, -0.01),
        local_max=(u1, v1, 0.01),
        axis_u=0,
        axis_v=1,
        axis_n=2,
        local_to_world=Gf.Matrix4d(1.0),
        world_min=(u0, v0, -0.01),
        world_max=(u1, v1, 0.01),
    )


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
            goal="Seat_003",
            assigned_seat="Seat_003",
        )

        self.assertEqual(brain.decide(agent), "GO_TO_SEAT")

        agent.perception["at_destination"] = True
        self.assertEqual(brain.decide(agent), "IDLE")

        agent.perception["fire_visible"] = True
        self.assertEqual(brain.decide(agent), "EXIT")

    async def test_seat_name_pattern(self):
        self.assertIsNotNone(SEAT_PATTERN.match("Seat_001"))
        self.assertIsNotNone(SEAT_PATTERN.match("Seat_100"))
        self.assertIsNotNone(SEAT_PATTERN.match("Seat_024"))
        self.assertIsNone(SEAT_PATTERN.match("Seat.001"))
        self.assertIsNone(SEAT_PATTERN.match("Seat_24"))
        self.assertEqual(int(SEAT_PATTERN.match("Seat_007").group(1)), 7)
        self.assertEqual(int(SEAT_PATTERN.match("Seat_100").group(1)), 100)

    async def test_spawn_samples_fill_xy_bounds(self):
        zone = _make_zone(0.0, 4.0, 0.0, 2.0)
        xs, ys = [], []
        for _ in range(40):
            p = sample_in_spawn(zone)
            self.assertTrue(point_in_spawn_horizontal(p, zone))
            xs.append(p[0])
            ys.append(p[1])
        # Must not collapse to a single point (the old origin bug)
        self.assertGreater(max(xs) - min(xs), 1.0)
        self.assertGreater(max(ys) - min(ys), 0.5)

    async def test_path_follow_steps_toward_waypoint(self):
        nav = NavigationSystem()
        agent = Agent(
            agent_id="walker",
            prim_path="/World/CrowdAgents/Walker",
            position=(0.0, 0.0, 0.0),
            height=1.7,
        )
        agent.destination = (2.0, 0.0, 0.0)
        agent.waypoints = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        agent.waypoint_index = 1
        agent.speed_mps = 1.0

        nav.apply_intent(agent, "WALK", dt=0.5)
        self.assertGreater(agent.position[0], 0.0)
        self.assertLess(agent.position[0], 2.0)
        self.assertFalse(agent.perception.get("at_destination"))

    async def test_lane_offset_helpers(self):
        from crowd.agents.navigation import _horiz_dir, _offset_horizontal

        d = _horiz_dir((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), "Y")
        self.assertIsNotNone(d)
        left = (-d[1], d[0])
        p = _offset_horizontal((0.0, 0.0, 0.0), left, 1.0, "Y")
        self.assertAlmostEqual(p[0], -1.0, places=5)
        self.assertAlmostEqual(p[2], 0.0, places=5)

    async def test_skips_overshot_waypoint(self):
        nav = NavigationSystem()
        agent = Agent(
            agent_id="overshoot",
            prim_path="/World/CrowdAgents/Overshoot",
            position=(1.5, 0.0, 0.0),
            height=1.7,
        )
        agent.destination = (4.0, 0.0, 0.0)
        agent.waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
        agent.waypoint_index = 1  # targeting checkpoint at x=1, but already at x=1.5
        nav._advance_past_waypoints(agent)
        self.assertGreaterEqual(agent.waypoint_index, 2)

    async def test_window_smoke_buttons_exist(self):
        spawn = ui_test.find(
            f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Spawn Agents (idle on Spawn)'"
        )
        run = ui_test.find(
            f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Run (to seats)'"
        )
        self.assertIsNotNone(spawn)
        self.assertIsNotNone(run)
