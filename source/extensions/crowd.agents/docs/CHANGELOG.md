# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [0.1.2] - 2026-08-21
- Phase 5 (partial): Individual Agent Info shows live state for the click-inspected agent
- Viewport click inspects an agent without keeping USD/Stage selection on capsules
- Agent prims marked unpickable; Stage picks are cleared immediately
- Selected agent: green highlight + FOV wireframe (210° / 20 m at eye height)
- Hovered agent: light-green highlight
- Agent pick uses camera-ray vs capsule (frontmost); SceneView NDC aligned to viewport
- State-change log on Agent Info (intent, goal, fire seen, arrival)

## [0.1.1] - 2026-08-21
- Phase 4a: Start Fire / Cancel placement with viewport preview cuboid and click-to-spawn
- Agent debug colors: blue toward seat, red toward exit
- Phase 4b: fire perception via range (20m) + 210° FOV; sticky once seen; no global evacuate
- Phase 4c: wall occlusion via omni.kit.mesh.raycast (fail-closed; no FOV-only fallback)
- LOS: skip non-occluders and re-cast; require 2 consecutive clears; cold BVH stays blocked

## [0.1.0] - 2026-08-20
- Initial Crowd Agents extension with control panel and USD smoke-test actions
- Phase 2: NavMesh bake, single capsule agent, path follow to random destination
- Phase 3: Spawn plane bounds, Seat assignment, Exit_A/B evacuation targets
- Startup layout: Individual Agent Info | Viewport | Stage/Render/Property tabs + Crowd Sim
- Stuck agents replan every 1s until seated; seated agents skip collisions
- Corridor lane jitter capped by NavMesh bake agent radius (no side probes)
- Skip overshot waypoints; replan ~1/s when waypoint closing rate is too low
