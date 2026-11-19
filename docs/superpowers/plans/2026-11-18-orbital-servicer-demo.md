# Orbital Servicer Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Basilisk 2.12.2 scenario that flies the ConOps module-swap mission and writes a Vizard 2.4 replay with real textured 3D models, ready to screen-record for a club pitch.

**Architecture:** Pure-Python logic modules (geometry, health monitor, guidance, arm kinematics, mission manager) are unit-tested on their own, then wired into one Basilisk sim as `sysModel.SysModel` Python modules. Client and servicer are real `spacecraft.Spacecraft` objects. Arm links, moving modules and red/green overlays are **extra Vizard bodies** fed by standalone `messaging.SCStatesMsg` messages and passed to `vizSupport.enableUnityVisualization` as `(name, msg)` tuples. Models are OBJ+MTL built or converted by scripts in `models/`.

**Tech Stack:** Python 3.14 (`~/miniconda3/bin/python`), Basilisk 2.12.2, Vizard 2.4, numpy, PyYAML, matplotlib, pytest, trimesh, Pillow.

**Spec:** `docs/superpowers/specs/2026-11-18-orbital-servicer-demo-design.md`

## Global Constraints

- Python: `~/miniconda3/bin/python` (3.14). Basilisk **2.12.2**. Vizard **2.4** (`/Applications/Vizard.app`).
- **Every number lives in `config/mission.yaml`.** Code reads it via `src/config.py`; no literals for mission values in code.
- `demo_mode: true` is the default. `--full-timeline` sets it false.
- Vizard models are **OBJ + MTL** with textures (`vizSupport.createCustomModel`). The only untextured shapes allowed in the final scene are the **semi-transparent keep-out sphere and approach cone**.
- **Downloads need Colby's OK first:** list each file's URL, license and size, then wait. Every third-party asset is recorded in `models/ASSETS.md` with its license.
- Fidelity Level 1: translation by `extForceTorque`, attitude set directly, arm and module motion scripted.
- Arm limits: ±170° per joint, ≤ 2°/s per joint, tool ≤ 2 cm/s free and ≤ 1 cm/s within 10 cm of contact.
- Docs (README) follow the `i-have-adhd` skill: answer first, scannable headers, tables, short blocks.

## Review Focus

1. **Close-up shots are dark** because the servicer works in the array's shadow (array faces the Sun). Expect a visible scene → config `viz.ambient: 0.35`, test asserts it is applied (Task 7).
2. **Arm link meshes don't meet at the joints** after rescaling NASA geometry. Expect a continuous arm → test that each link mesh spans its FK link length within 2 cm (Task 6).
3. **A carried module passes through the array or truss** during swings. Expect clean motion → test that sampled module bodies stay ≥ 0.1 m behind the array plane except during extract/insert (Task 4).
4. **The replay is too large to scrub** in Vizard. Expect smooth scrubbing → test `outputs/servicer_demo.bin` < 300 MB (Task 7).
5. **Rerunning gives a different video** (noise). Expect identical runs → test that two runs with the same `seed` give the same service-request time (Task 2).

---

### Task 0: Scaffold, environment and config

**Files:**
- Create: `config/mission.yaml`, `src/__init__.py`, `src/config.py`, `src/modules/__init__.py`, `tests/test_config.py`, `requirements.txt`, `.gitignore`

**Interfaces:**
- Produces: `load_config(path: str | None = None, full_timeline: bool = False) -> dict` (nested dict mirroring the YAML; `full_timeline=True` sets `demo_mode` false and uses the full hold durations).

- [ ] **Step 1: Init repo and install deps**

Run: `git init && ~/miniconda3/bin/python -m pip install pytest trimesh pillow`
Expected: repo created; `python -c "import pytest, trimesh, PIL"` exits 0.

- [ ] **Step 2: Write `config/mission.yaml`**

Sections and values (all from the spec):
- `seed: 42`, `demo_mode: true`
- `orbit`: alt_km 550, inc_deg 97.6, raan_deg 90, arg_lat_deg 0, epoch `"2027 MARCH 20 12:00:00.0 (UTC)"`
- `client`: bus_size_m 4.0, bus_mass_kg 5000, bus_offset_m 4.0, truss_mass_kg 400, module_mass_kg 12, grid 8, slot_m 2.0, module_thick_m 0.05, bus_slots `[[4,4],[4,5],[5,4],[5,5]]`, target `[2,7]`, keepout_m 15, cone_half_deg 10, cone_len_m 20
- `power`: solar_flux 1361, eff 0.30, area_m2 4.0, aging 0.88, noise 0.01, strike_t_s 600, strike_frac_remaining 0.35, degraded 0.85, failed 0.50, service 0.70, persist_s 60, checkout_band `[0.98, 1.20]`
- `servicer`: mass_kg 250, inertia `[40,40,30]`, size_m `[1.0,1.0,1.2]`, prop_kg 15, isp_s 220, max_thrust_n 2.0
- `waypoints`: H0 `[0,-2000,0]`, H1 `[0,-200,0]`, H2 `[0,-30,0]` (Hill), W3_offset_m 20, H3_offset_m 20, berth_offset_m 2.6
- `speeds`: leg_mps 0.05, final_mps 0.02, slow_zone_m 10
- `control`: wn 0.02, zeta 0.9
- `holds_s` (demo / full): go 60/600, H1 120/900, H2 300/5736, H3 60/600, berth 60/300, checkout 60/300
- `arm`: links_m `[1.1, 1.1, 0.3]`, limit_deg 170, rate_dps 2.0, tip_free_mps 0.02, tip_contact_mps 0.01, rack_side "+y", step_min (6.1–6.12) `[1,6,2,3,2,10,6,10,2,3,2,6]`
- `viz`: save_file "outputs/servicer_demo.bin", rate_s 1.0, ambient 0.35
- `dyn`: dt_far_s 1.0, dt_near_s 0.1, near_m 200

- [ ] **Step 3: Write the failing test**

```python
def test_loads_target_and_demo_default():
    c = load_config()
    assert c["client"]["target"] == [2, 7]
    assert c["demo_mode"] is True
    assert c["holds_s"]["H2"] == 300

def test_full_timeline_uses_full_holds():
    c = load_config(full_timeline=True)
    assert c["demo_mode"] is False
    assert c["holds_s"]["H2"] == 5736
```

- [ ] **Step 4: Run** `pytest tests/test_config.py -v` → FAIL (no module).
- [ ] **Step 5: Implement `load_config`** — store holds as `{demo: x, full: y}` in YAML; the loader flattens to the active value.
- [ ] **Step 6: Run** `pytest tests/test_config.py -v` → PASS.
- [ ] **Step 7: Commit** `git add -A && git commit -m "chore: scaffold config and env"`

---

### Task 1: Geometry and frames

**Files:**
- Create: `src/geometry.py`, `tests/test_geometry.py`

**Interfaces:**
- Consumes: `load_config`.
- Produces:
  - `module_center(r: int, c: int, cfg) -> np.ndarray` (array frame A, m)
  - `module_ids(cfg) -> list[tuple[int,int]]` (60 entries, bus slots excluded)
  - `hill_dcm(r_N, v_N) -> np.ndarray` (dcm_HN)
  - `array_dcm(sun_hat_N, r_N, v_N) -> np.ndarray` (dcm_AN: z_A = sun_hat, y_A = Hill y projected ⊥ z_A, x_A = y_A × z_A)
  - `waypoints_A(cfg, dcm_AN, dcm_HN) -> dict[str, np.ndarray]` (H2, W3, H3, BERTH in A relative to array center)
  - `client_mass_props(cfg, missing: set[tuple]) -> (mass, com_B, inertia_B)`

- [ ] **Step 1: Write the failing tests**

```python
def test_target_module_center():   assert np.allclose(module_center(2, 7, cfg), [5, -5, 0])
def test_sixty_modules():          ids = module_ids(cfg); assert len(ids) == 60 and (4, 4) not in ids
def test_dcms_orthonormal():       for d in (hill_dcm(r, v), array_dcm(s, r, v)): assert np.allclose(d @ d.T, np.eye(3))
def test_berth_and_h3():           w = waypoints_A(cfg, A, H); assert np.allclose(w["BERTH"], [5, -5, -2.6]) and np.allclose(w["H3"], [5, -5, -20])
def test_w3_is_h2_minus_normal():  assert np.allclose(w["W3"] - w["H2"], [0, 0, -20])
def test_client_mass():            m, _, _ = client_mass_props(cfg, set()); assert abs(m - 6120) < 1
```

- [ ] **Step 2: Run** `pytest tests/test_geometry.py -v` → FAIL.
- [ ] **Step 3: Implement** the functions in `src/geometry.py`. Inertia: bus as a solid cube, array as thin plates per module plus the truss as a 16 m plate.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: geometry and frames`

---

### Task 2: Module power and health monitor

**Files:**
- Create: `src/modules/health_monitor.py`, `tests/test_health_monitor.py`

**Interfaces:**
- Consumes: `module_ids`.
- Produces:
  - `module_powers(t: float, cfg, state: PowerState, rng) -> dict[id, float]` (W, with aging, strike, noise, switch state, replaced flag)
  - `class PowerState` (fields: `struck: set`, `open_switch: set`, `replaced: set`)
  - `class HealthMonitor(cfg)` with `step(t, powers) -> list[Event]`; `Event(t, kind: "degraded"|"failed"|"service_request"|"checkout_pass"|"checkout_fail", module)`
  - `array_total(powers) -> float`

- [ ] **Step 1: Write the failing tests**

```python
def test_service_request_about_661s():   # baseline strike at 600 s
    assert 655 <= first_event(run(cfg, strike=True), "service_request").t <= 665
def test_aging_only_no_request():        assert not any(e.kind == "service_request" for e in run(cfg, strike=False))
def test_new_module_ratio_passes():      ratio = replaced_ratio(cfg); assert 1.10 < ratio < 1.18
def test_power_restored():               assert array_total(after) >= 0.99 * array_total(before_fault)
def test_same_seed_same_result():        assert first_event(run(cfg, seed=42), "service_request").t == first_event(run(cfg, seed=42), "service_request").t
```
(`run` is a test helper that steps 1 Hz from 0 to 1000 s.)

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** per spec: ratio = power / median of 60; persistence counters per threshold; "failed" opens the switch and emits `service_request`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: power model and health monitor`

---

### Task 3: Relative guidance

**Files:**
- Create: `src/modules/rel_guidance.py`, `tests/test_rel_guidance.py`

**Interfaces:**
- Produces:
  - `radial_hop_dv(dy_m: float, n: float) -> float` (= dy·n/4)
  - `cw_propagate(x0: np.ndarray[6], n: float, t: float) -> np.ndarray[6]` (Hill frame)
  - `cw_relative_accel(r_H, v_H, n) -> np.ndarray[3]` (feedforward)
  - `class StraightLeg(p0, p1, v_max, a_max)` with `.duration` and `.at(t) -> (pos, vel, acc)`; trapezoidal speed
  - `pd_accel(pos_err, vel_err, wn, zeta) -> np.ndarray[3]`

- [ ] **Step 1: Write the failing tests**

```python
def test_far_hop_dv():           assert abs(radial_hop_dv(1800, 0.0010948) - 0.4926) < 1e-3
def test_hop_lands_on_h1():      # start H0 at rest, radial push, propagate half orbit
    x = cw_propagate([0, -2000, 0, dv, 0, 0], n, np.pi / n); assert abs(x[1] - (-200)) < 1.0 and abs(x[0]) < 1.0
def test_leg_respects_speed():   assert max(norm(leg.at(t)[1]) for t in ts) <= 0.05 + 1e-9
def test_leg_arrives():          assert np.allclose(leg.at(leg.duration)[0], p1, atol=1e-6)
def test_legs_outside_keepout(): # H2→W3→H3 legs sampled
    assert min(norm(p) for p in sampled) >= 15.0
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Use the closed-form CW state-transition matrix.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: CW hops and forced-motion legs`

---

### Task 4: Arm kinematics and swap sequence

**Files:**
- Create: `src/modules/arm_motion.py`, `tests/test_arm_motion.py`

**Interfaces:**
- Consumes: `module_center`, config `arm`.
- Produces:
  - `fk(q: np.ndarray[6], cfg) -> list[tuple[np.ndarray, np.ndarray]]` — pose (pos_S, dcm_SL) of base, each of 6 links and the tool, in servicer body S. Shoulder at front-face center (+x = 0.5 m).
  - `ik(pos_S, dcm_S, q_seed, cfg) -> np.ndarray[6]` — damped least squares; raises `IKError` if unreachable.
  - `rack_slot_pose(k: int, cfg) -> (pos_S, dcm_S)` — slots 0–3 spares, 4–5 stow, on +y face, panels parallel to the S x-z plane, 0.08 m apart outward from y = 0.55 m.
  - `class SwapSequence(cfg, target_fixture_S)` with `.duration` (sum of ConOps step minutes × 60) and `.at(t) -> ArmFrame(q, step: str, held: str|None, bodies: dict[name, (pos_S, dcm_S)], contact: bool)`. Body names: `"module_target"`, `"spare_0".."spare_3"`.
  - `SwapSequence.events` — list of `(t, "capture"|"release_client"|"insert_contact"|"latched"|"release_tool")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_zero_pose_reach():        assert abs(norm(fk(np.zeros(6), cfg)[-1][0] - shoulder) - 2.5) < 1e-6
def test_ik_roundtrip():           q = ik(*fk(q0, cfg)[-1], q_seed, cfg); assert np.allclose(fk(q, cfg)[-1][0], fk(q0, cfg)[-1][0], atol=1e-4)
def test_all_targets_reachable():  # target fixture at berth and all rack slots
    for pose in [target] + [rack_slot_pose(k, cfg) for k in range(6)]: ik(*pose, q_seed, cfg)
def test_joint_limits_and_rate():  # sample every 1 s over seq.duration
    assert all(abs(q) <= deg(170)) and max_rate <= deg(2.0) + 1e-9
def test_tip_speed_near_contact(): assert max tool speed within 0.10 m of contact <= 0.01 + 1e-9
def test_module_clears_array():    # Review Focus 3: outside extract/insert steps
    assert all(z_A(body) <= -0.10 for body in carried_module_samples)
def test_step_order():             assert [e[1] for e in seq.events] == ["capture", "release_client", "insert_contact", "latched", "release_tool"]
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Joint axes: z, y, y, y, z, x (S frame at q=0, arm along +x). Each step is a joint-space trapezoid between IK solutions, with duration from config, stretched if needed to meet rate limits. Steps 6.3, 6.5 and 6.9 are Cartesian straight lines at ≤ 1 cm/s, solved by IK at 1 s samples.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: arm kinematics and swap sequence`

---

### Task 5: Mission manager

**Files:**
- Create: `src/modules/mission_manager.py`, `tests/test_mission_manager.py`

**Interfaces:**
- Consumes: `HealthMonitor` events, `radial_hop_dv`, `StraightLeg`, `cw_relative_accel`, `pd_accel`, `SwapSequence`, `waypoints_A`.
- Produces:
  - `class MissionManager(cfg, geo)` with `update(t, rel_H: np.ndarray[6], rel_A: np.ndarray[6], events) -> Command`
  - `Command(phase: int, phase_name: str, force_N: np.ndarray[3], thrusters_inhibited: bool, arm_t: float | None, overlay: "none"|"red"|"green")`
  - `PHASES = ["Monitoring", "Go / no-go", "Far approach", "Inspection", "Move to back side", "Final approach", "Module exchange", "Checkout", "Departure", "Home"]`
  - `.dv_used` (m/s, integrated |force|/mass)

- [ ] **Step 1: Write the failing tests** — drive with a test-only point-mass CW plant at 1 s.

```python
def test_phase_order():        assert phases_seen == list(range(10))
def test_red_then_green():     assert first_t(overlay == "red") < first_t(overlay == "green")
def test_berth_accuracy():     assert norm(rel_A_at_exchange_start - berth) < 0.05 and norm(vel) < 0.005
def test_ends_at_h0():         assert norm(final_rel_H[:3] - H0) < 5.0
def test_dv_budget():          assert mm.dv_used <= 6.0
def test_corridor():           # inside 15 m sphere → inside 10° cone about -z_A from target
    assert all(in_cone(p) for p in samples if norm(p) < 15)
def test_inhibit_while_linked(): assert all(c.thrusters_inhibited for c in cmds if linked(c))
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** the phase state machine. Hops use finite burns at `max_thrust_n` plus one midpoint correction from `cw_propagate`. Close range uses `StraightLeg` reference + `pd_accel` + `cw_relative_accel` feedforward, converted to force. Hold durations come from `cfg["holds_s"]`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: mission manager phase logic`

---

### Task 6: 3D models

**Files:**
- Create: `models/build_models.py`, `models/fetch_assets.py`, `models/assets_manifest.yaml`, `models/ASSETS.md`, `tests/test_models.py`
- Output: `models/obj/*.obj|mtl|png` (git-ignored except `.gitkeep`)

**Interfaces:**
- Produces: `MODEL_FILES: dict[str, str]` in `models/build_models.py` — keys `client`, `module`, `servicer`, `arm_link_1`..`arm_link_6`, `tool`, `overlay_red`, `overlay_green`; values are OBJ paths.

- [ ] **Step 1: Find assets and get approval (gated)**

Search NASA 3D Resources (public domain) for a robot arm (Canadarm2 / Dextre) and a satellite bus, and Poly Haven (CC0) for metal, gold-foil/MLI and carbon textures. Write `models/assets_manifest.yaml` (name, url, license, size_mb). **Stop and show Colby the list. Download only after his OK.**

- [ ] **Step 2: Write the failing tests**

```python
def test_all_models_exist():      assert all(Path(p).exists() and Path(p).with_suffix(".mtl").exists() for p in MODEL_FILES.values())
def test_textures_referenced():   assert all("map_Kd" in Path(p).with_suffix(".mtl").read_text() for p in MODEL_FILES.values())
def test_module_dims():           assert np.allclose(extents("module"), [2.0, 2.0, 0.05], atol=0.02)
def test_client_has_59_modules(): # M(2,7) is its own body
    assert count_module_groups("client") == 59
def test_arm_links_meet():        # Review Focus 2
    for i, L in enumerate(link_lengths): assert abs(extent_along_axis(f"arm_link_{i+1}") - L) < 0.02
def test_assets_licensed():       assert every manifest entry appears in ASSETS.md with a license
```

- [ ] **Step 3: Run** `pytest tests/test_models.py -v` → FAIL.
- [ ] **Step 4: Implement `build_models.py`.**
  - **Solar cell texture generated by script** (Pillow): dark-blue cells with silver busbars and fingers, white frame, 2048 px.
  - Client: foil-textured bus, carbon truss, 59 modules with the cell texture (M(2,7) excluded).
  - Module, rack and overlays (semi-emissive red/green planes, 2.02 m square).
  - Servicer: downloaded bus if approved, else built from foil and metal textures.
  - Arm: downloaded meshes converted with `trimesh` and **split and rescaled** onto the 6 link lengths. If the asset can't be split cleanly, build links as textured tubes with joint housings and note it in `ASSETS.md`.
- [ ] **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** `feat: textured 3D models`

---

### Task 7: Scenario, Vizard replay and plots

**Files:**
- Create: `src/scenario_servicing.py`, `src/viz_setup.py`, `src/plots.py`, `tests/test_scenario.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run(cfg, viz: bool = True, outdir="outputs") -> RunLog` (`RunLog` holds time series: t, rel_H, rel_A, phase, dv_used, array_power, module_ratios, arm q)
  - CLI: `python -m src.scenario_servicing [--full-timeline] [--no-viz]`
  - `setup_viz(sim, task, client_sc, servicer_sc, extra_bodies: dict[name, SCStatesMsg], cfg) -> viz`
  - `make_plots(log: RunLog, outdir) -> list[Path]` — `hill_path.png`, `range_speed.png`, `module_ratio.png`, `delta_v.png`

- [ ] **Step 1: Write the failing tests** (`--no-viz` run for speed, plus one viz run marked `slow`)

```python
def test_mission_completes():      assert log.phase[-1] == 9
def test_power_restored():         assert log.array_power[-1] >= 0.99 * log.array_power[log.t < 600].mean()
def test_dv_budget():              assert log.dv_used[-1] <= 6.0
@pytest.mark.slow
def test_viz_file_small():         assert 0 < Path("outputs/servicer_demo.bin").stat().st_size < 300e6   # Review Focus 4
@pytest.mark.slow
def test_ambient_applied():        assert viz.settings.ambient == cfg["viz"]["ambient"]                  # Review Focus 1
def test_plots_written():          assert all(p.exists() for p in make_plots(log, tmp_path))
```

- [ ] **Step 2: Run** `pytest tests/test_scenario.py -v` → FAIL.
- [ ] **Step 3: Implement the sim.**
  - `SimulationBaseClass` with dynamics, FSW (0.5 s) and viz (`viz.rate_s`) tasks. Dynamics dt switches far/near at `near_m`.
  - Earth gravity, `spiceInterface` Sun at the epoch, `eclipse` on.
  - Client attitude set each step to `array_dcm` (Sun-pointing, inertially fixed).
  - Servicer gets `extForceTorque` from `Command.force_N`, its attitude set directly (camera at client far out, matching client close in), and its hub mass updated at module handoffs.
  - A Python `SysModel` writes `SCStatesMsg` for every extra body each step: arm links and tool from `fk`, modules from `SwapSequence.bodies`, overlays parked 1 km below the client until `Command.overlay` says otherwise.
- [ ] **Step 4: Implement `setup_viz`.** Pass `scList=[client, servicer, *(name, msg) tuples]`, `saveFile=cfg["viz"]["save_file"]`. Then:
  - `createCustomModel` for each body using `MODEL_FILES`.
  - `createConeInOut` for the approach cone, plus a transparent keep-out sphere.
  - `createStandardCamera` ×3: wide, servicer→client, arm close-up.
  - Set `viz.settings.ambient`, hide labels on the extra bodies, and put the phase name in the servicer label if no text overlay is available.
- [ ] **Step 5: Implement `make_plots`** (matplotlib, follow the `dataviz` skill).
- [ ] **Step 6: Run** `pytest tests/ -v -m "not slow"` then `pytest tests/test_scenario.py -v -m slow` → PASS.
- [ ] **Step 7: Open the replay in Vizard** and check all 8 beats from the spec's shot list are visible. Take a screenshot of beat 6 for Colby.
- [ ] **Step 8: Commit** `feat: full scenario, Vizard replay and plots`

---

### Task 8: README and recording guide

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README** (load the `i-have-adhd` skill first):
  - **Run it in 3 steps:** setup, build models, run the scenario.
  - **Recording guide:** one table row per shot-list beat, with sim time range, camera, Vizard playback speed and what to show. Use macOS screen recording (Cmd-Shift-5).
  - Basilisk/Vizard versions and module substitutions.
  - Known limits (Level 1, scripted arm).
  - **What the club builds next:** Levels 2–3, T2–T8, Monte Carlo, pass/fail report.
- [ ] **Step 2: Verify** by following the 3 steps from a clean shell → replay file produced.
- [ ] **Step 3: Commit** `docs: README and recording guide`
