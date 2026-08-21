# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [0.1.0] - 2026-08-20
- Initial Crowd Agents extension with control panel and USD smoke-test actions
- Phase 2: NavMesh bake, single capsule agent, path follow to random destination
- Phase 3: Spawn plane bounds, Seat assignment, Exit_A/B evacuation targets
- Startup layout: Individual Agent Info | Viewport | Stage/Render/Property tabs + Crowd Sim
- Stuck agents replan every 1s until seated; seated agents skip collisions
- Corridor lane jitter capped by NavMesh bake agent radius (no side probes)
- Skip overshot waypoints; replan ~1/s when waypoint closing rate is too low
