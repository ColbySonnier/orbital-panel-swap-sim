# Rail Gantry Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the free-flying servicer with an on-board rail gantry that swaps module M(2,7) from a panel stack, and produce a new Vizard replay.

**Architecture:** A pure-Python `GantrySequence` scripts the bridge, carriage, forks and panel bodies in array frame A, with events matching the health monitor's checkout flow. The Basilisk scenario keeps one real spacecraft, the client. Every gantry part and panel is an extra Vizard body whose inertial pose is the client pose ∘ its A-frame pose. The old servicer, guidance, arm and mission-manager code is deleted.

**Tech Stack:** Python 3.14 (`~/miniconda3/bin/python`), Basilisk 2.12.2, Vizard 2.4, numpy, PyYAML, matplotlib, pytest, trimesh, Pillow.

**Spec:** `docs/superpowers/specs/2026-11-19-rail-gantry-design.md`. It amends `docs/superpowers/specs/2026-11-18-orbital-servicer-demo-design.md`.

## Global Constraints

- **Every number lives in `config/mission.yaml`** (new `gantry:` block). No mission literals in code.
- Vizard models are **textured OBJ + MTL**, believable hardware, no plain boxes.
- Speeds: bridge and carriage ≤ **0.05 m/s**. Fork ≤ **0.02 m/s**, and ≤ **0.01 m/s within 0.10 m of contact**.
- Events: `capture`, `release_client`, `insert_contact`, `latched`, `release_tool`.
- Do **not** overwrite `outputs/` until Task G3's final run.
- Commit trailer: a blank line, then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Docs follow the ADHD format: answer first, scannable headers, tables, short blocks.

## Review Focus

1. **A panel passes through the truss, the rails or the other panels** while moving. Expect clean motion. Test with sampled OBB checks against `truss_members`, the gantry parts and the stacked panels (G1).
2. **Gantry parts visibly detached** (carriage not on its bridge, fork not on the carriage). Expect one connected mechanism at all times. Test the attachment geometry at sampled times (G1, G2).
3. **Bridge B hits the bus or standoff.** Expect clearance. Test min distance ≥ 0.1 m over the sequence (G1).
4. **The swapped panel isn't seated in the array slot.** This was the bug in Colby's earlier replay. Expect spare dcm = I and the slot center at the end. Test in A and in the inertial replay frame (G1, G3).
5. **Overlay or panel teleports** at step boundaries. Expect continuity. Test the max jump between 1 s samples (G1).

---

### Task G1: Gantry config and motion sequence

**Files:**
- Create: `src/modules/gantry_motion.py`, `tests/test_gantry_motion.py`
- Modify: `config/mission.yaml` (add `gantry:`)

**Interfaces:**
- Consumes: `src/geometry.module_center`, `models.build_models.truss_members(cfg)`.
- Produces:
  - `gantry_layout(cfg) -> dict`: rail_y, rail_z, bridge_z, park_x (A and B), bay_pitch, bay_z0, stack_dz, the grapple point.
  - `class GantrySequence(cfg, bridge="B")`:
    - `.duration`
    - `.events: list[(t, name)]`
    - `.steps: list[(t_start, name)]`
    - `.at(t) -> GantryFrame`
  - `GantryFrame` fields: `bridge_x: dict[str, float]`, `carriage_y`, `fork_z: dict["W"|"F", float]`, `step: str`, `bodies: dict[name, (pos_A, dcm_A)]` (names: `bridge_A`, `bridge_B`, `carriage`, `fork_W`, `fork_F`, `module_target`, `spare_0..spare_3`), `contact: bool`.

- [ ] **Step 1: Add `gantry:` to the config**
  - rail_y 8.6, rail_z −2.0, bridge_z −2.0
  - park_x {A: −8.6, B: 8.6}, x_limits {A: [−8.6, −2.6], B: [2.6, 8.6]}
  - bay_pitch 2.1, bay_z0 −1.4, stack_dz 0.1, n_fresh 4
  - travel_mps 0.05, fork_mps 0.02, contact_mps 0.01, contact_zone_m 0.10
  - accel_mps2 0.005, latch_s 10, settle_s 5
- [ ] **Step 2: Write the failing tests**

```python
def test_events_in_order():        assert [e[1] for e in seq.events] == ["capture","release_client","insert_contact","latched","release_tool"]
def test_speed_limits():           # sample every 0.25 s
    assert max bridge/carriage speed <= 0.05+1e-9 and fork <= 0.02+1e-9 and fork near contact <= 0.01+1e-9
def test_spare_seated_at_end():    p, R = seq.at(seq.duration).bodies["spare_0"]; assert np.allclose(p, module_center(2,7,cfg), atol=1e-6) and np.allclose(R, np.eye(3))
def test_dead_panel_in_waste_bay():# module_target ends in bay W at bay_z0
def test_no_panel_collisions():    # Review Focus 1: moving panels vs truss_members capsules, the other panels, and the bridge/carriage boxes
def test_bridge_clears_bus():      # Review Focus 3: min distance >= 0.1 m from the bus cube and the standoff
def test_mechanism_connected():    # Review Focus 2: carriage on its bridge, forks on the carriage, held panel on its fork at all samples
def test_no_teleports():           # Review Focus 5: max body jump between 1 s samples <= travel_mps*1 s + 1e-6
def test_ends_parked():            assert frame(seq.duration).bridge_x["B"] == park_x["B"]
```

- [ ] **Step 3: Run** `~/miniconda3/bin/python -m pytest tests/test_gantry_motion.py -v` → FAIL.
- [ ] **Step 4: Implement.** Follow the spec's 9 steps: trapezoidal 1-D moves per axis. A held panel follows its fork tip. Spares 1–3 stay in the stack, and each spare drops one pitch when the one below leaves (optional; it must stay continuous). The end time includes roll home.
- [ ] **Step 5: Run** → PASS. Commit `feat: gantry config and motion sequence`.

---

### Task G2: Gantry 3D models; remove servicer and arm models

**Files:**
- Modify: `models/build_models.py`, `tests/test_models.py`, `models/ASSETS.md`
- MODEL_FILES after: `client` (now includes the static rails), `module`, `bridge`, `carriage`, `fork`, `overlay_red`, `overlay_green`. Drop `servicer`, `arm_link_*`, `tool` and `keepout_sphere`.

**Interfaces:**
- Consumes: `gantry_layout(cfg)`.
- Produces: OBJs in the frames `GantrySequence.bodies` uses:
  - **bridge:** origin at the bridge center, spanning y.
  - **carriage:** origin at the carriage center, between the two bays, with the magazine frame and guides.
  - **fork:** origin at the fork tip (grapple point), the shaft extending along −z.

- [ ] **Step 1: Failing tests:** every key exists with its MTL and textures, and UVs are present. Bridge length spans the rails within 2 cm. The rails are in the client OBJ at y = ±rail_y. The magazine bay pitch matches config. Keep the existing module and truss tests.
- [ ] **Step 2: Implement** believable hardware:
  - Rails: aluminium extrusion with carbon supports down to the keel.
  - Bridge: a carbon box truss with drive trucks at each end.
  - Carriage: an aluminium frame, two bays with guide rails, drive motors, and a cable chain.
  - Forks: a telescoping shaft, with a grapple end effector that matches the module fixture.
- [ ] **Step 3: Delete** the servicer, arm and keep-out builders and their tests. Regenerate the contact sheet.
- [ ] **Step 4: Run** `pytest tests/test_models.py` → PASS. Commit `feat: gantry models, drop servicer and arm models`.

---

### Task G3: Scenario, replay and plots on the gantry; delete the free-flyer code

**Files:**
- Modify: `src/scenario_servicing.py`, `src/viz_setup.py`, `src/plots.py`, `tests/test_scenario.py`, `config/mission.yaml`
- Delete: `src/modules/mission_manager.py`, `src/modules/rel_guidance.py`, `src/modules/arm_motion.py`, and their tests. Also delete the config blocks `servicer`, `waypoints`, `speeds`, `control`, `tolerances`, `arm`, the obsolete `holds_s` keys and the obsolete `phase_timeout_s`.

**Interfaces:**
- Consumes: `GantrySequence`, `MODEL_FILES`, `HealthMonitor` and friends.
- Produces: `run(cfg, viz=True, outdir="outputs") -> RunLog` with t, phase, gantry step, array_power, module_ratios, and the bridge/carriage/fork positions. Also the CLI (`--full-timeline`, `--no-viz`, `--outdir`) and `make_plots`. New plots: `module_ratio.png`, `array_power.png`, `gantry_motion.png`, `timeline.png`.

- [ ] **Step 1: Failing tests:**
  - Phases 0–5 run in order.
  - Red comes before green.
  - Power restored ≥ 99%.
  - Spare seated in the INERTIAL replay frame: the module pose equals the client pose ∘ slot pose (Review Focus 4).
  - Overlay sits on the slot.
  - The replay is < 300 MB.
  - Ambient is applied.
  - Plots are written.
- [ ] **Step 2: Implement.**
  - Mission flow: a small phase machine inside the scenario per the spec's table.
  - Glue: on "failed", open the switch. On `latched`, install the module, then `monitor.reset` and `start_checkout`.
  - The client is the only Spacecraft. Extra bodies are SCStatesMsg tuples.
  - Three cameras: wide, gantry back-side, swap close-up.
  - HUD: phase, step, power.
- [ ] **Step 3: Delete** the obsolete modules, tests and config. The full suite must pass.
- [ ] **Step 4: Final run** into `outputs/`. Commit `feat: gantry scenario and replay`.

---

### Task G4: README and recording guide

**Files:** Create `README.md`

- [ ] **Step 1:** Load the ADHD format. Cover:
  - The 3-step run.
  - A recording guide with one table row per beat (sim time, camera, playback speed).
  - What's in the repo, config, tests, known limits and gotchas.
  - The SPICE kernel auto-download.
  - What the club builds next: a free-flyer resupply ship, Level 2–3 dynamics, test cases, and the pass/fail report.
  - Credits and licenses.
- [ ] **Step 2:** Verify the 3 steps from a clean shell into a temp outdir. Commit `docs: README and recording guide`.
