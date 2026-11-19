# Rail gantry panel swap: design

An **on-board rail gantry** on the orbital data center swaps a damaged solar module from a **panel stack**. It **replaces the free-flying servicer** in `2026-11-18-orbital-servicer-demo-design.md`. That doc still governs anything this one doesn't change: intent, audience, Vizard, textured models, recorded video.

## TL;DR

- **What changed:** no free-flyer, no hops, no berthing, no 6-joint arm. A **gantry riding rails on the back of the array** does the swap.
- **Why:** Colby's Vizard check. The robot is meant to be **part of the satellite**, a device on rails that **drops a panel in place from a stack**. The old replay also had arm-link gaps, and the swap didn't land in the big array.
- **Video:** about 1–2 minutes. Fault → gantry rolls out → swap → green → gantry rolls home.
- **Kept:** Basilisk orbit and Sun, client model and textures, fault detection, config file, Vizard, red/green overlays, recorded-video workflow.

## Decisions (Colby)

| Question | Answer |
|---|---|
| Free-flyer? | **Removed entirely** (option A) |
| Robot style | **Gantry + panel stack** (not an arm on a rail cart) |
| Proposed design below | "Looks right, go build it" |

## Hardware

All positions are in array frame A: origin at the array center, +z toward the Sun, back side is −z. Every number goes in `config/mission.yaml` under `gantry:`.

| Part | Design |
|---|---|
| **Rails** | Two rails along x at **y = ±8.6 m, z = −2.0 m**, behind the truss (truss keel at z = −1.0). Static, so they're part of the client model |
| **Bridges** | **Two**, each spanning y (17.2 m) at z = −2.0. **Bridge A** covers columns 1–3 (x ≤ −2.6), **bridge B** covers columns 6–8 (x ≥ +2.6). Two are needed because the center bus mount and standoff block a single bridge. Parked at x = ±8.6 |
| **Carriage** | Rides its bridge in y and holds a **two-bay magazine** |
| **Magazine** | **Bay W (waste):** empty at start, receives the dead panel. **Bay F (fresh):** a stack of **4 fresh panels**. Bays sit side by side in y with **2.1 m pitch**, panels parallel to the array. The bottom panel is at z = −1.25, stack pitch 0.17 m (amended: both changed from −1.4 / 0.1 to clear fork F's crossbar) |
| **Forks** | One lift fork per bay. Each moves along z from its bay down to the array's grapple point |

## Swap sequence (bridge B, target M(2,7) at A (5, −5, 0))

| # | Step | Motion |
|---|---|---|
| 1 | Roll out | Bridge B from x = 8.6 to 5.0 |
| 2 | Align waste bay | Carriage to y = −5 (bay W over the slot) |
| 3 | Grab | Fork W descends to the panel back, capture |
| 4 | Pull | Fork W lifts the dead panel straight back into bay W |
| 5 | Shift | Carriage moves 2.1 m in y so bay F is over the slot |
| 6 | Drop in | Fork F lowers the bottom fresh panel into the slot |
| 7 | Latch | Latch hold, then release |
| 8 | Retract | Fork F rises back to the stack |
| 9 | Roll home | Carriage and bridge return to park |

**Speeds:** bridge and carriage ≤ **5 cm/s**, fork ≤ **2 cm/s**, and ≤ **1 cm/s within 10 cm of contact**. Trapezoidal profiles. The whole swap takes on the order of 10–20 minutes of sim time, which the video compresses.

**Events:** the same names as before, so fault detection and checkout carry over: `capture`, `release_client`, `insert_contact`, `latched`, `release_tool`.

## Mission flow

| Phase | Name | Ends when |
|---|---|---|
| 0 | Monitoring | `service_request` (~660 s), red overlay on |
| 1 | Go / no-go | `holds_s.go` |
| 2 | Gantry swap | Gantry sequence reaches step 8 |
| 3 | Checkout | `checkout_pass`, green overlay on |
| 4 | Return | Gantry parked |
| 5 | Done | — |

## Visuals

- **Separate Vizard bodies:** bridge A, bridge B, carriage + magazine frame, fork W, fork F, module_target, 4 fresh panels, 2 overlays. Each pose = client pose ∘ A-frame pose.
- **Cameras:**
  - Wide view over Earth.
  - Back-side gantry view.
  - Swap close-up.
- **HUD:** phase, gantry step, array power.
- **Removed from the scene:** keep-out sphere, approach cone, servicer.

## Left to the builder

- Exact visual detail of the rails, bridges, carriage and forks (textured, believable hardware, no plain boxes).
- Fork grapple look.
- Camera poses.

## Known limits

- **Column 6 may be tight** against the bus standoff for the magazine footprint. The demo target is column 7. Document it.
- Fidelity Level 1: client attitude is ideal, and gantry motion is scripted.
