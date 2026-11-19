# Orbital data center: rail gantry solar-module swap

A **Basilisk + Vizard** demo. An on-board rail gantry on an orbital data center pulls a dead solar module and drops in a spare. Built to pitch a new aerospace club.

## TL;DR

- **Run it:** `~/miniconda3/bin/python -m src.scenario_servicing` (about **2 s**).
- **Watch it:** open Vizard, **Select** `outputs/servicer_demo.bin`, **Start Visualization**.
- **Record it:** use the table in [Recording guide](#recording-guide). Sim time is **1670 s**.
- **First run needs internet** (SPICE kernels auto-download).

## Quick start (3 steps)

| # | Do this | You should see |
|---|---|---|
| 1 | **Build the models** (first time, **and again after any pull that touches `models/`**):<br>`~/miniconda3/bin/python -m models.fetch_assets`<br>`~/miniconda3/bin/python -m models.build_models` | OBJ + MTL + textures in `models/obj/`. The sim **refuses to run with Vizard** on stale OBJs and tells you to rerun `build_models` |
| 2 | **Run the sim:**<br>`~/miniconda3/bin/python -m src.scenario_servicing` | Phase list, event times, `replay ... (4.8 MB)`, 4 plot paths |
| 3 | **Open Vizard:** Select, then `outputs/servicer_demo.bin`, then **Start Visualization** | The array over Earth, HUD in the corner |

Install deps once with `~/miniconda3/bin/python -m pip install -r requirements.txt`. Basilisk 2.12.2 is already in the conda env.

## Command options

| Flag | Effect |
|---|---|
| `--outdir DIR` | Write the replay and plots to `DIR` instead of `outputs/` |
| `--no-viz` | Skip the Vizard replay (plots and logic only) |
| `--full-timeline` | Real ConOps hold times (go/no-go 600 s, checkout 300 s) instead of the short demo holds |

## Recording guide

Colby screen-records Vizard and trims the clip. **Start the recording at about 540 s** to skip the quiet monitoring phase.

Speeds are **suggestions**. At these speeds the swap plays in about **1.5 to 2 minutes**.

| Beat | Sim time (s) | Camera | Speed |
|---|---|---|---|
| **Fault**: strike to service request | 600 - 660 | **Wide orbit view** | 4x |
| **Go / no-go**: red frame appears on the slot | 660 - 720 | Wide orbit view | 4x |
| **Roll out**: bridge B drives to the slot | 720 - 807 | **Gantry view** | 16x |
| **Align waste bay** | 807 - 901 | Gantry view | 16x |
| **Grab**: fork captures the dead module (capture at 977) | 901 - 987 | **Swap close-up** | 8x |
| **Pull**: module leaves the slot (release at 995) | 987 - 1067 | Swap close-up | 8x |
| **Shift**: carriage moves to the spare | 1067 - 1124 | Swap close-up | 16x |
| **Drop in**: spare seats (contact at 1189) | 1124 - 1200 | Swap close-up | 8x |
| **Latch** (latched at 1210) | 1200 - 1215 | Swap close-up | 4x |
| **Retract**: fork F pulls back to the stack (release at 1215) | 1215 - 1295 | Swap close-up | 16x |
| **Checkout hold**, then **green at 1330** | 1296 - 1330 | Swap close-up | 8x |
| **Roll home**: bridge drives back to park | 1330 - 1548 | Gantry view, then Wide orbit view | 32x |
| **Done tail** (nothing moves) | 1548 - 1670 | Wide orbit view | Trim it |

### Cameras

| Name | Best for |
|---|---|
| **Wide orbit view** | Opening, closing, the red/green frame. Earth limb stays in frame all run. |
| **Gantry view** | Bridge rolling along the rails |
| **Swap close-up** | Forks and panels at slot M(2,7) |

All three are client cameras, mounted on the data center. Poses live in `viz.cameras` in `config/mission.yaml`.

## Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| First run hangs or errors offline | **SPICE kernels auto-download** on first run | Run once with internet. They are cached after. |
| Segfault or `std::bad_alloc` in `VizInterface` | Vizard keeps **raw pointers to lights**. A garbage-collected `Light` crashes it. | Already fixed: `viz_setup._LIVE_LIGHTS` holds them. **Keep any new lights in that list.** |
| Replay is named `servicer_demo.bin` | **Historical name** from the old free-flyer design | Rename via `viz.save_file` in `config/mission.yaml` if you want |
| Red/green frame looks like **four bars** from behind | The back frame has open corners to clear the truss | Expected. The front rim is a full ring. |
| `Stale or missing model files` error | `models/obj/` is git-ignored, so OBJs built before a frame fix linger | `python -m models.build_models` |
| Model loads untextured | MTL can't find textures | Keep OBJ, MTL and textures together in `models/obj/` |
| Green arrives mid-retract | Checkout hold shorter than about 86 s | Keep `holds_s.checkout` above **86 s** |

## Known limits

- **A failed checkout ends the run.** The sim stops at the failed checkout, leaves the gantry retracted, and prints `CHECKOUT FAILED`. It does not roll home.
- **Fidelity Level 1:** client attitude is **ideal**, and gantry motion is **scripted**.
- **Column 6 is tight** against the bus standoff for the magazine footprint. The demo uses **column 7**.
- **Visuals not hand-tuned in Vizard yet:** camera framing, work light, and the overlay look came from geometry. Adjust `viz.cameras` and `viz.work_light` in config. `viz.rate_s` sets the Vizard pose-writer step.
- **No Earth albedo:** vizSupport 2.12.2 has no earthshine setting. Lighting is ambient (0.55) plus a work-light spotlight.

## What's in the repo

| Path | What it is |
|---|---|
| `src/scenario_servicing.py` | Main script: runs the sim, writes the replay and plots |
| `src/viz_setup.py` | Vizard models, cameras, lights, HUD |
| `src/geometry.py`, `src/config.py` | Array geometry and config loader |
| `src/modules/gantry_motion.py` | Scripted gantry sequence (roll, grab, pull, drop in, retract) |
| `src/modules/health_monitor.py` | Power monitor: fault, go/no-go, checkout |
| `src/plots.py` | The 4 plots |
| `config/mission.yaml` | **All parameters**: orbit, power, gantry, cameras |
| `models/` | Model builders, asset fetcher, `ASSETS.md` |
| `tests/` | 94 tests |
| `docs/superpowers/specs/` | Design docs |
| `outputs/` | Replay and plots (generated) |

## Run the tests

```
~/miniconda3/bin/python -m pytest -q
```

Expect **94 passed** in about 2 minutes.

## Outputs

| File | Shows |
|---|---|
| `servicer_demo.bin` | Vizard replay (about 4.8 MB) |
| `module_ratio.png` | Module power vs. pre-fault |
| `array_power.png` | Array power, dip and recovery |
| `gantry_motion.png` | Gantry axes over time |
| `timeline.png` | Phases and events |

## What the club builds next

- **Free-flyer resupply ship** that delivers the fresh panels
- **Level 2-3 dynamics**: real attitude control and gantry disturbances
- **Test cases** beyond the nominal swap
- **Pass/fail report** generated from each run

## Credits and licenses

- **All models are built by script** from **CC0** ambientCG textures: SolarPanel001, Foil002, Metal051A, Fabric004.
- **No attribution required.** No third-party mesh ships.
- Full table and the NASA meshes we evaluated and skipped: `models/ASSETS.md`.
- Simulation: **Basilisk** (AVS Lab, CU Boulder). Viewer: **Vizard**.
