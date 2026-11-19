"""Vizard 2.4 setup for the rail-gantry replay.

Bodies (one Vizard model each):
  client            Basilisk Spacecraft. Body frame B = A axes, origin at the
                    client CoM, so client.obj (origin at the array center,
                    rails included) is drawn with offset = -CoM_A.
  bridge_A/B,       standalone SCStatesMsg bodies written by BodyWriter from
  carriage,         GantrySequence.at(t).bodies (array frame A):
  fork_W/F,           r_N = r_client + dcm_NA @ (p_A - com_A),  dcm_NB = dcm_NA @ R_A.
  fork_*_stage1..4, Every model is built in its body frame with A axes
  module_target,    (models/build_models.py), and every gantry/panel body has
  spare_0..3        R_A = I, so no model rotation is applied.
  overlay_red,      module-frame plates on the slot M(2,7), or parked 1 km
  overlay_green     along -z_A when hidden.

OBJ axes: Vizard 2.4 imports OBJ with the Dummiesman runtime OBJLoader. It
mirrors x to convert right-handed OBJ data into Unity's left-handed space,
the same handedness change Vizard applies to Basilisk body frames, so every
OBJ is authored directly in its Basilisk body frame and passed with rotation
0. MTL map_Bump becomes a tangent-space normal map and MTL `d` alpha, so
createCustomModel's normalMapPath is left empty.
"""
from pathlib import Path

import numpy as np
from Basilisk.simulation import vizInterface
from Basilisk.utilities import vizSupport

from models.build_models import MODEL_FILES
from src.geometry import client_mass_props

OVERLAYS = ["overlay_red", "overlay_green"]
CAMERAS = ["Wide orbit view", "Gantry view", "Swap close-up"]
_LIVE_LIGHTS = []  # keeps the Light structs alive while vizInterface points at them


def model_key(body):
    """MODEL_FILES key drawn for a gantry/panel/overlay body."""
    if body.startswith("bridge_"):
        return "bridge"
    if body in ("fork_W", "fork_F"):
        return f"{body}_head"
    if body == "module_target" or body.startswith("spare_"):
        return "module"
    return body  # carriage, fork_*_stageN, overlay_*


def _reset_vizsupport_lists():
    """vizSupport keeps module-level lists; enableUnityVisualization clears
    most of them but not the standard-camera list (matters when run() is
    called twice in one process, e.g. in tests)."""
    del vizSupport.stdCameraList[:]


def _look(cam, com):
    """(position_B, unit pointing_B) of a camera given in A (eye, target)."""
    eye, tgt = np.array(cam["eye_A"], float), np.array(cam["target_A"], float)
    d = tgt - eye
    return list(eye - com), list(d / np.linalg.norm(d))


def work_light(cfg, com):
    """Client-mounted spotlight on the back-side work zone (viz.work_light)."""
    w = cfg["viz"]["work_light"]
    pos, nrm = _look({"eye_A": w["pos_A"], "target_A": w["target_A"]}, com)
    li = vizInterface.Light()
    li.label = "Work light"
    li.position = pos
    li.normalVector = nrm
    li.fieldOfView = np.radians(w["fov_deg"])
    li.range = float(w["range_m"])
    li.intensity = float(w["intensity"])
    li.lightOn = 1
    li.showLightMarker = -1
    li.showLensFlare = -1
    return li


def setup_viz(sim, task, client_sc, bodies, cfg):
    """Create the vizInterface (saved replay) with every model, the work light,
    three cameras and the display settings. `bodies` maps a body name to its
    standalone SCStatesMsg (gantry, panels, overlays). Returns the viz module."""
    _reset_vizsupport_lists()
    names = list(bodies)
    sc_list = [client_sc, *((n, bodies[n]) for n in names)]
    off = [0, 0, 0, 1]  # true-path color alpha 1/255 ~ off (0 = Vizard default)
    save = Path(cfg["viz"]["save_file"])
    if not save.is_absolute():
        save = Path(__file__).resolve().parents[1] / save
    save.parent.mkdir(parents=True, exist_ok=True)
    _, com, _ = client_mass_props(cfg, set())
    lights = [[work_light(cfg, com)]] + [None] * len(names)
    viz = vizSupport.enableUnityVisualization(sim, task, sc_list, saveFile=str(save),
                                              trueOrbitColorList=[off] * len(sc_list),
                                              lightList=lights)
    _LIVE_LIGHTS[:] = lights[0]  # VizSpacecraftData.lightList holds raw Light* pointers
    name_c = client_sc.ModelTag

    # --- models (no rotation: every OBJ is in its body frame, A axes) ----------
    vizSupport.createCustomModel(viz, MODEL_FILES["client"], simBodiesToModify=[name_c],
                                 offset=list(-com))
    groups = {}
    for n in names:
        groups.setdefault(model_key(n), []).append(n)
    for key, members in groups.items():
        vizSupport.createCustomModel(viz, MODEL_FILES[key], simBodiesToModify=members)

    # --- cameras on the client (positions/targets in A, viz.cameras) -----------
    for display, key in zip(CAMERAS, ("wide", "gantry", "closeup")):
        cam = cfg["viz"]["cameras"][key]
        pos, pnt = _look(cam, com)
        vizSupport.createStandardCamera(viz, spacecraftName=name_c, setMode=1,
                                        pointingVector_B=pnt, position_B=pos,
                                        fieldOfView=np.radians(cam["fov_deg"]),
                                        displayName=display)

    # --- settings ---------------------------------------------------------------
    s = viz.settings
    s.ambient = cfg["viz"]["ambient"]
    s.showSpacecraftLabels = -1        # no per-body label switch in 2.12.2
    s.orbitLinesOn = -1                # a 7000 km orbit line cuts through close-ups
    s.trueTrajectoryLinesOn = -1
    s.showLightLabels = -1
    s.showMissionTime = 1
    s.forceStartAtSpacecraftLocalView = 1
    viz.vizEventDialogs.clear()
    return viz


def hud_dialog():
    """HUD panel: one VizEventDialog re-sent every viz frame with new text."""
    d = vizInterface.VizEventDialog()
    d.eventHandlerID = "HUD"
    d.durationOfDisplay = 0
    return d
