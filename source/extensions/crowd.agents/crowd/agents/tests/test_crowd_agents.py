# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import omni.kit.test
import omni.kit.ui_test as ui_test
from pxr import Gf

from crowd.agents.ai import (
    RuleBasedDecisionEngine,
    agent_to_slm_dict,
    normalize_decision,
)
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

    async def test_agent_to_slm_dict_shape(self):
        agent = Agent(
            agent_id="Agent_001",
            prim_path="/World/CrowdAgents/Agent_001",
            goal="Seat_003",
            assigned_seat="Seat_003",
            intent="WALK",
        )
        agent.perception.update(
            {
                "fire_visible": False,
                "smoke_visible": False,
                "seat_known": True,
                "at_destination": False,
                "nearest_exit": "Exit_A",
                "exits": ["Exit_A", "Exit_B"],
            }
        )
        row = agent_to_slm_dict(agent)
        self.assertEqual(row["id"], "Agent_001")
        self.assertEqual(row["goal"], "Seat_003")
        self.assertEqual(row["assigned_seat"], "Seat_003")
        self.assertEqual(row["intent"], "WALK")
        self.assertEqual(
            set(row["perception"].keys()),
            {
                "fire_visible",
                "smoke_visible",
                "seat_known",
                "at_destination",
                "nearest_exit",
                "exits",
            },
        )
        self.assertEqual(row["perception"]["nearest_exit"], "Exit_A")
        self.assertEqual(row["perception"]["exits"], ["Exit_A", "Exit_B"])

    async def test_normalize_decision_fallback_on_bad_intent(self):
        rules = RuleBasedDecisionEngine()
        agent = Agent(
            agent_id="Agent_002",
            prim_path="/World/CrowdAgents/Agent_002",
            goal="Seat_001",
            assigned_seat="Seat_001",
        )
        agent.perception["at_destination"] = False
        bad = normalize_decision(
            {"id": "Agent_002", "intent": "DANCE", "destination": None},
            rules,
            agent,
        )
        self.assertEqual(bad.intent, "GO_TO_SEAT")
        self.assertEqual(bad.destination, "Seat_001")

        good = normalize_decision(
            {"id": "Agent_002", "intent": "IDLE", "destination": "Seat_001"},
            rules,
            agent,
        )
        self.assertEqual(good.intent, "IDLE")
        self.assertIsNone(good.destination)

    async def test_fire_in_view_fov_and_range(self):
        from crowd.agents.fire import fire_in_view

        agent_pos = (0.0, 0.0, 0.0)
        facing = (0.0, 0.0, 1.0)  # looking +Z
        # Fire ahead within range
        self.assertTrue(
            fire_in_view(agent_pos, facing, (0.0, 0.0, 5.0), up="Y", view_range_m=20.0)
        )
        # Fire behind — outside 210° FOV (±105°)
        self.assertFalse(
            fire_in_view(agent_pos, facing, (0.0, 0.0, -5.0), up="Y", view_range_m=20.0)
        )
        # Fire ahead but too far
        self.assertFalse(
            fire_in_view(agent_pos, facing, (0.0, 0.0, 50.0), up="Y", view_range_m=20.0)
        )
        # Fire to the side within ~60° half-angle (at 45° should be visible)
        self.assertTrue(
            fire_in_view(agent_pos, facing, (5.0, 0.0, 5.0), up="Y", view_range_m=20.0)
        )

    async def test_los_path_blocking_helpers(self):
        from crowd.agents.fire import _path_blocks_los, _path_is_fire

        self.assertTrue(_path_is_fire("/World/CrowdAgents/Fire"))
        self.assertFalse(_path_blocks_los("/World/CrowdAgents/Fire"))
        self.assertFalse(_path_blocks_los("/World/CrowdAgents/Agent_001"))
        self.assertFalse(_path_blocks_los("/World/NavVolumes/Nav_001"))
        self.assertTrue(_path_blocks_los("/World/Venue/Wall_A"))
        self.assertTrue(_path_blocks_los("/World/Seat_003"))
        self.assertTrue(_path_blocks_los(""))  # fail-closed

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

    async def test_agent_prim_path_helper(self):
        from crowd.agents.simulation import _is_agent_prim_path

        self.assertTrue(_is_agent_prim_path("/World/CrowdAgents/Agent_001"))
        self.assertTrue(_is_agent_prim_path("/World/CrowdAgents/Agent_012/Geom"))
        self.assertFalse(_is_agent_prim_path("/World/CrowdAgents/Fire"))
        self.assertFalse(_is_agent_prim_path("/World/CrowdAgents/TestCube"))
        self.assertFalse(_is_agent_prim_path("/World/Seat_001"))

    async def test_inspect_horizontal_distance(self):
        from crowd.agents.inspect import horizontal_distance

        d = horizontal_distance((0.0, 0.0, 0.0), (3.0, 5.0, 4.0), "Y")
        self.assertAlmostEqual(d, 5.0, places=5)

    async def test_fov_sector_polyline(self):
        from crowd.agents.viz import fov_sector_polyline

        pts = fov_sector_polyline(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            up="Y",
            height=1.7,
            fov_deg=90.0,
            range_m=10.0,
            segments=8,
        )
        # eye → left → arc… → right → eye
        self.assertGreaterEqual(len(pts), 11)
        self.assertAlmostEqual(pts[0][0], pts[-1][0], places=5)
        self.assertAlmostEqual(pts[0][1], pts[-1][1], places=5)
        self.assertAlmostEqual(pts[0][2], pts[-1][2], places=5)
        # Eye lifted on Y
        self.assertGreater(pts[0][1], 1.0)

    async def test_agent_highlight_color_priority(self):
        from crowd.agents.viz import COLOR_HOVERED, COLOR_SELECTED, agent_highlight_color

        self.assertEqual(
            agent_highlight_color(selected=True, hovered=True, evacuating=True),
            COLOR_SELECTED,
        )
        self.assertEqual(
            agent_highlight_color(selected=False, hovered=True, evacuating=False),
            COLOR_HOVERED,
        )

    async def test_ray_capsule_frontmost(self):
        from crowd.agents.agent import Agent
        from crowd.agents.inspect import pick_frontmost_along_ray, ray_capsule_intersection_t

        # Ray along +Z from origin; near agent at z=2, far at z=8
        near = Agent(
            agent_id="near",
            prim_path="/World/CrowdAgents/Agent_001",
            position=(0.0, 0.0, 2.0),
            height=1.7,
            radius=0.35,
        )
        far = Agent(
            agent_id="far",
            prim_path="/World/CrowdAgents/Agent_002",
            position=(0.0, 0.0, 8.0),
            height=1.7,
            radius=0.35,
        )
        origin = (0.0, 0.85, -1.0)
        direction = (0.0, 0.0, 1.0)
        t_near = ray_capsule_intersection_t(
            origin, direction, near.position, near.height, near.radius, up="Y"
        )
        t_far = ray_capsule_intersection_t(
            origin, direction, far.position, far.height, far.radius, up="Y"
        )
        self.assertIsNotNone(t_near)
        self.assertIsNotNone(t_far)
        self.assertLess(t_near, t_far)
        picked = pick_frontmost_along_ray([far, near], origin, direction, up="Y")
        self.assertIsNotNone(picked)
        self.assertEqual(picked.agent_id, "near")

        # Miss: ray beside agents
        miss = ray_capsule_intersection_t(
            (5.0, 0.85, -1.0), direction, near.position, near.height, near.radius, up="Y"
        )
        self.assertIsNone(miss)

    async def test_window_smoke_buttons_exist(self):
        spawn = ui_test.find(
            f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Spawn Agents (idle on Spawn)'"
        )
        run = ui_test.find(
            f"{WINDOW_TITLE}//Frame/**/Button[*].text=='Run (to seats)'"
        )
        self.assertIsNotNone(spawn)
        self.assertIsNotNone(run)
