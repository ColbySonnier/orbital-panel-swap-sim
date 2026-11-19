# Orbital servicer demo: design

A Basilisk + Vizard replay of a robotic servicer swapping a damaged solar module on an orbital data center. **It's built to get people excited** about the project at a new aerospace club's pitch. Mission content comes from the ConOps *Robotic solar module replacement for orbital data centers* (Nov 18, 2026).

## TL;DR

- **What:** one Python script that runs a Basilisk sim and writes a **Vizard replay file** of the whole mission, about **3 minutes** of playback.
- **The main event:** a close-up of the **robot arm pulling the dead module and inserting a spare**.
- **Look:** **real textured 3D models** (arm, servicer, solar cells, foil). No plain boxes. Render resolution doesn't matter.
- **Physics:** real orbit and real approach maneuvers in Basilisk. Arm and module motion are **scripted** (ConOps Fidelity Level 1).
- **Out of scope:** test cases T2–T8, Monte Carlo, Levels 2–3 and the pass/fail report. Those are **what the club builds next**.

## Intent

| Question | Answer |
|---|---|
| **Who watches** | People who might join a new aerospace club |
| **What they should feel** | "This looks real and I want to work on it" |
| **What success looks like** | The replay plays smoothly at the pitch, the swap looks like real hardware, and the code can be the club's starting point |
| **How it's shown** | **Recorded video.** Colby screen-records the Vizard replay (speeding through hops, switching cameras), trims it, and plays the clip in the pitch slides. Vizard stays available for questions afterward |
| **Anti-goals** | Looks like boxes · long stretches where nothing moves · anything that can break during the pitch itself |

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Basilisk + Vizard** | Interactive, uses the club's real toolchain. Rendering is Unity-grade, not photoreal | **Picked** |
| B. Basilisk → web 3D page (three.js) | Shareable link and full visual control, but it isn't the club's toolchain | Dropped |
| C. Basilisk → Blender render | Photoreal, but needs a separate install and hours of rendering | Dropped. Colby wants real models, not render quality |

**Why A:** the demo is also the club's starter code. When people ask "what would we work on?", the answer is "this."

## Constraints

| Item | Value |
|---|---|
| Basilisk | **2.12.2**, miniconda Python 3.14 (`~/miniconda3`) |
| Vizard | **2.4**, `/Applications/Vizard.app` |
| Model format | Vizard loads **OBJ + MTL** with textures and normal maps (`vizSupport.createCustomModel`) |
| Platform | macOS |
| Repo | `robotics-club-demo/`, currently empty and **not a git repo** |

## The story (shot list)

| # | Beat | On screen | Length |
|---|---|---|---|
| 1 | **The data center** | Wide orbit shot: foil-wrapped bus, truss and a 16 m solar array over Earth | 15 s |
| 2 | **The strike** | Module **M(2,7) turns red**. Label: "Module power down 65%: service requested" | 15 s |
| 3 | **Servicer launches** | Servicer at 2 km fires its thrusters. Its trail draws the curved hop | 30 s |
| 4 | **Inspect and go around back** | Stops at 30 m, then moves to the back side. **Keep-out sphere and approach cone** appear | 25 s |
| 5 | **Berthing** | Slow straight-in approach to 2.6 m behind the module | 20 s |
| 6 | **The swap** | Arm close-up camera: **grab → pull → stow → take spare → insert → latch** | 60 s |
| 7 | **Power back** | Module **turns green**. Label: "Array power restored" | 10 s |
| 8 | **Departure** | Servicer backs out and heads home | 15 s |

Lengths are at the presenter's chosen Vizard playback speed. They're targets, not hard limits.

## Simulation

### Geometry (from the ConOps)

| Item | Value |
|---|---|
| **Client attitude** | Array normal points **at the Sun**, held **inertially fixed**. In a dawn-dusk orbit that's close to Hill +z (orbit normal) |
| **Frames** | Hill H: x radial out, y along velocity, z orbit normal. Array A: origin at array center, +z toward the Sun, x/y along columns/rows. **A is inertial, H rotates once per orbit**, so close-range guidance runs in A with relative-gravity feedforward |
| **Array grid** | 8×8 slots of 2 m. M(r,c) center: **x = 2c − 9, y = 2r − 9, z = 0** (r, c = 1–8). **Center 2×2 slots** (rows 4–5, cols 4–5) are the bus mount, which leaves 60 modules |
| **Bus** | 4 m cube, center **4 m behind** the array plane (−z in A) on a 2 m standoff |
| **Target** | **M(2,7)** at A (5, −5, 0) |
| **Keep-out sphere** | **15 m** radius around the array center |
| **Approach cone** | Tip at the target module, axis along −z in A, **10° half-angle, 20 m long** |

### Waypoints

| Name | Frame | Position |
|---|---|---|
| H0 park / **home** | Hill | (0, −2000, 0) m |
| H1 | Hill | (0, −200, 0) m |
| H2 | Hill | (0, −30, 0) m |
| **W3** | — | **H2 + 20 m along −array normal** |
| H3 | Array | Target + (0, 0, −20) m |
| Berth | Array | Target + (0, 0, −2.6) m. Servicer front face 2.0 m behind the array plane |

**Path:** H0 → H1 → H2 (radial hops, 0.49 and 0.047 m/s pushes) → W3 (5 cm/s) → H3 (5 cm/s, parallel to the array, staying ≥ 20 m from center) → berth (5 cm/s, then 2 cm/s inside 10 m). Departure reverses it back to **H0**.

### Start state and fault

- **t = 0:** servicer at H0, client at argument of latitude 0.
- **t = 600 s:** M(2,7) loses 65% of its strings, so its output drops to 35%.
- **Detection:** each module's power divided by the 60-module median. Below **0.50 for 60 s** = failed, so the switch opens and **service is requested at about 661 s**. All modules carry a 0.88 aging factor and 1% noise.
- **Checkout:** the new module's ratio is about 1.14 (pass band 0.98–1.20).

### Arm (from the ConOps, link lengths are my call)

| Item | Value |
|---|---|
| Mount | Center of the servicer's front (+x) face |
| Joints | Shoulder yaw → shoulder pitch → elbow pitch → wrist pitch → wrist yaw → wrist roll, ±170°, ≤ 2°/s |
| Lengths *(my call)* | Upper arm **1.1 m**, forearm **1.1 m**, wrist + tool **0.3 m** = 2.5 m reach |
| Spares rack | On the servicer's **+y side**: 4 spares and 2 stow slots *(side is my call)* |
| Tip speed | ≤ 2 cm/s free, ≤ 1 cm/s within 10 cm of contact |

**NASA arm mesh:** used as **visual skin only**. Its link meshes are rescaled onto the 6-joint, 2.5 m kinematics above. Canadarm2's 7th joint is dropped.

### Swap steps (ConOps §6.9)

| Step | Action | ConOps min | Demo duration *(my call)* |
|---|---|---|---|
| 6.1 | Confirm module switch open | 1 | 2 s |
| 6.2 | Unstow arm, tool to 10 cm behind fixture | 6 | 8 s |
| 6.3 | Capture fixture at 1 cm/s | 2 | 5 s |
| 6.4 | Release latches, demate connector | 3 | 4 s |
| 6.5 | Pull module 0.5 m back | 2 | 5 s |
| 6.6 | Swing to stow slot, latch | 10 | 8 s |
| 6.7 | Grapple spare from rack | 6 | 6 s |
| 6.8 | Move spare to 0.5 m behind empty slot | 10 | 8 s |
| 6.9 | Insert at 1 cm/s | 2 | 5 s |
| 6.10 | Close latches, mate connector | 3 | 4 s |
| 6.11 | Release, back off 10 cm | 2 | 3 s |
| 6.12 | Stow arm | 6 | 6 s |

**Demo durations are playback seconds.** In demo mode the arm runs at its real speed limits and the replay is sampled so the swap plays in about 60 s.

### What is real physics

- **Two Basilisk spacecraft:** client (data center) and servicer, in the ConOps orbit (550 km dawn-dusk SSO, epoch 2027-03-20 12:00 UTC).
- **Earth point-mass gravity, SPICE sun**, and Vizard's lit Earth.
- **Servicer translation:** forces applied with `extForceTorque` (Level 1). It flies **Clohessy-Wiltshire radial hops** H0→H1→H2 and **forced-motion legs** H2→W3→H3→berth in the array frame. Departure reverses the path.
- **Attitude:** ideal. Set to the commanded attitude (Level 1).

### What is scripted

- **Fault:** M(2,7) drops to 35% at 600 s. Detection follows ConOps §5 in simplified form.
- **Arm:** 6 joints driven by **trapezoidal joint-space paths** through the ConOps §6.9 steps. Forward kinematics places each link.
- **Modules in transit:** the damaged module and the spare follow the gripper while held, and snap to their slot or rack when released.

### Demo mode (time compression)

The full timeline is about 8 h, mostly waiting. **`demo_mode: true` (default)** shortens it:

| Item | Full | Demo |
|---|---|---|
| Ground go/no-go wait | 10 min | 1 min |
| H1 hold | 15 min | 2 min |
| H2 inspection hold | 95.6 min | 5 min |
| H3 hold, berth hold | 10 min, 5 min | 1 min each |
| Arm steps | 53 min total | Same steps, **shorter durations** so the swap reads well on screen |

Orbital hops keep their real half-orbit duration. **The presenter speeds through them with Vizard's playback slider.**

## Visuals

### 3D assets

| Object | Source plan |
|---|---|
| **Robot arm** | Public-domain **NASA 3D Resources** arm (Canadarm2 or Dextre class). **Split into 6 link meshes** at the joints |
| **Servicer bus** | NASA 3D Resources or a CC-BY Sketchfab satellite bus. Fallback: built from real textures |
| **Data center** | **No such model exists**, so it's built by script: foil-wrapped bus, truss, 60 cell modules using **Poly Haven (CC0)** solar-cell, foil and metal textures |
| **Solar module** | Textured panel mesh with cell grid, frame and grapple fixture on the back face |
| **Spares rack** | Built by script from the same textures |

**Download rule:** before downloading, list every file with its **source, license and size** and get Colby's OK. Record each asset's license in `models/ASSETS.md`.

**Conversion:** downloaded GLB/FBX/3DS files are converted to OBJ + MTL with a Python mesh library such as `trimesh` (pip install, approved as part of this plan).

### How moving parts show in Vizard

Vizard draws one model per spacecraft. **Each arm link, each moving module and each overlay gets its own Vizard body.** Its state message is written each step from the kinematics. *(Builder's call: separate `Spacecraft` objects with states set directly, or standalone state messages fed to `vizInterface`. Use whichever 2.12.2 supports cleanly.)*

### Red / green module

Vizard can't change a texture mid-run. **Plan:** red and green overlay bodies park out of view and **move onto M(2,7)** at the fault (red) and after checkout (green).

### Cameras and labels

- **Cameras:** wide pair view, servicer-to-client view, arm close-up for beat 6.
- **Keep-out sphere and approach cone:** semi-transparent, shown from beat 4 on.
- **Labels:** phase name, range and closing speed. Use whatever on-screen text Vizard 2.4 supports from Basilisk. If none, put the phase name in the spacecraft label.

## Deliverables

| Path | Content |
|---|---|
| `config/mission.yaml` | Every number, including `demo_mode` |
| `src/scenario_servicing.py` | Runs the sim and writes the Vizard file. `--full-timeline` flag |
| `src/modules/` | `rel_guidance.py`, `mission_manager.py`, `arm_motion.py`, `health_monitor.py` |
| `models/` | Asset download/convert script, built-mesh script, `ASSETS.md` licenses |
| `outputs/` | Vizard `.bin` file, plus 3–4 backup-slide plots (Hill-frame path, range vs time, module power ratio, delta-v) |
| `README.md` | **Run it in 3 steps**, a **recording guide** (which camera and playback speed for each beat, using macOS screen recording), and what the club could build next |

## Left to the builder

- Exact Vizard API calls for cameras, transparent shapes and text in 2.4.
- How to split the downloaded arm mesh into links.
- Arm joint-angle waypoints, as long as they look natural and stay within the ±170° limits.
- Logging rate and replay file size, which must stay small enough to scrub smoothly.

## Next steps for the club (not in this build)

- Fidelity Levels 2–3 (real thrusters, reaction wheels, multi-body arm, grapple constraint).
- Test cases T2–T8 and Monte Carlo.
- Pass/fail report against ConOps §13.
