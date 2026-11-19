"""Vizard 2.4 setup for the servicer demo replay.

Bodies (one Vizard model each):
  client          Basilisk Spacecraft. Body frame B = A axes, origin at the
                  client CoM, so client.obj (origin at the array center) is
                  drawn with offset = -CoM_A.
  servicer        Basilisk Spacecraft. Body frame = S (servicer.obj is in S).
  arm_link_1..6,  standalone SCStatesMsg bodies, each in its fk link frame.
  tool
  module_target,  module.obj. The OBJ is built in A axes when seated (+z =
  spare_0..3      cell normal); arm_motion's M frame has x_M = normal, so
                  each body's attitude is dcm_NO = dcm_NM @ R_MO (R_MO maps
                  OBJ coordinates into M: OBJ +z -> x_M).
  overlay_red,    module-frame plates, seated on M(2,7) or parked 1 km below
  overlay_green   the client.
  keepout         keepout_sphere.obj (models/build_models.py, MTL d < 1)
                  at the array center from inspection on, parked otherwise.

OBJ axes: Vizard 2.4 imports OBJ with the Dummiesman runtime OBJLoader
(OBJLoader / MTLLoader classes in Vizard's Assembly-CSharp.dll). It mirrors
x to convert right-handed OBJ data into Unity's left-handed space, the same
handedness change Vizard applies to Basilisk body frames, so every OBJ is
authored directly in its Basilisk body frame and passed with rotation 0.
The same loader maps MTL map_Bump to Unity's _BumpMap with the _NORMALMAP
keyword (a tangent-space normal map), and MTL `d` to alpha, so the MTLs are
used as written and createCustomModel's normalMapPath is left empty.
"""
from pathlib import Path

import numpy as np
from Basilisk.simulation import vizInterface
from Basilisk.utilities import vizSupport

from models.build_models import MODEL_FILES
from src.geometry import client_mass_props, module_center
from src.modules.arm_motion import _ry

# OBJ (seated A axes, +z normal) -> module frame M (x_M normal): v_M = R_MO v_O.
R_MO = _ry(np.pi / 2)

ARM_BODIES = [f"arm_link_{i}" for i in range(1, 7)] + ["tool"]
MODULE_BODIES = ["module_target"] + [f"spare_{k}" for k in range(4)]
OVERLAYS = ["overlay_red", "overlay_green"]
CLIENT_SIDE = ["module_target", *OVERLAYS, "keepout"]  # listed after the client
EXTRA_BODIES = CLIENT_SIDE + ARM_BODIES + [f"spare_{k}" for k in range(4)]

CAMERAS = ["Wide pair view", "Servicer to client", "Arm close-up"]


def module_dcm_NO(dcm_NS, dcm_SM):
    """Attitude (dcm_N<-OBJ) of a module body given its M frame in S."""
    return np.asarray(dcm_NS) @ np.asarray(dcm_SM) @ R_MO


def _reset_vizsupport_lists():
    """vizSupport keeps module-level lists; enableUnityVisualization clears
    most of them but not the standard-camera list (matters when run() is
    called twice in one process, e.g. in tests)."""
    del vizSupport.stdCameraList[:]


def setup_viz(sim, task, client_sc, servicer_sc, extra_bodies, cfg):
    """Create the vizInterface (saved replay) with every model, the corridor
    shapes, three cameras and the display settings. extra_bodies maps a
    body name to its standalone SCStatesMsg. Returns the viz module."""
    _reset_vizsupport_lists()
    client_extra = [n for n in CLIENT_SIDE if n in extra_bodies]
    servicer_extra = [n for n in extra_bodies if n not in client_extra]
    sc_list = [client_sc, *((n, extra_bodies[n]) for n in client_extra),
               servicer_sc, *((n, extra_bodies[n]) for n in servicer_extra)]
    # true-path colors: only the servicer's trail is drawn (alpha 1/255 ~ off;
    # 0 would mean "Vizard default")
    off = [0, 0, 0, 1]
    colors = [[60, 220, 255, 255] if sc is servicer_sc else off for sc in sc_list]
    save = Path(cfg["viz"]["save_file"])
    if not save.is_absolute():
        save = Path(__file__).resolve().parents[1] / save
    save.parent.mkdir(parents=True, exist_ok=True)
    viz = vizSupport.enableUnityVisualization(sim, task, sc_list, saveFile=str(save),
                                              trueOrbitColorList=colors)

    c = cfg["client"]
    _, com, _ = client_mass_props(cfg, set())
    name_c, name_s = client_sc.ModelTag, servicer_sc.ModelTag

    # --- models ---------------------------------------------------------------
    vizSupport.createCustomModel(viz, MODEL_FILES["client"], simBodiesToModify=[name_c],
                                 offset=list(-com))
    vizSupport.createCustomModel(viz, MODEL_FILES["servicer"], simBodiesToModify=[name_s])
    for n in ARM_BODIES:
        if n in extra_bodies:
            vizSupport.createCustomModel(viz, MODEL_FILES[n], simBodiesToModify=[n])
    mods = [n for n in MODULE_BODIES if n in extra_bodies]
    if mods:
        vizSupport.createCustomModel(viz, MODEL_FILES["module"], simBodiesToModify=mods)
    for n in OVERLAYS:
        if n in extra_bodies:
            vizSupport.createCustomModel(viz, MODEL_FILES[n], simBodiesToModify=[n])
    if "keepout" in extra_bodies:
        vizSupport.createCustomModel(viz, MODEL_FILES["keepout_sphere"], simBodiesToModify=["keepout"])

    # --- approach cone: tip at M(2,7), axis -z_A, on the client body ----------
    tip_B = module_center(*c["target"], cfg) - com
    vizSupport.createConeInOut(viz, fromBodyName=name_c, toBodyName=name_s,
                               coneColor=[80, 200, 255, 60], isKeepIn=True,
                               position_B=list(tip_B), normalVector_B=[0.0, 0.0, -1.0],
                               incidenceAngle=np.radians(c["cone_half_deg"]),
                               coneHeight=float(c["cone_len_m"]), coneName="Approach cone")

    # --- cameras ---------------------------------------------------------------
    # 1 wide: on the client, back (servicer) side, looking at the CoM
    p_wide = np.array([30.0, -55.0, -45.0])
    vizSupport.createStandardCamera(viz, spacecraftName=name_c, setMode=1,
                                    pointingVector_B=list(-p_wide / np.linalg.norm(p_wide)),
                                    position_B=list(p_wide), fieldOfView=np.radians(40),
                                    displayName=CAMERAS[0])
    # 2 servicer front camera (+x_S), at the stereo camera on the front face
    vizSupport.createStandardCamera(viz, spacecraftName=name_s, setMode=1,
                                    pointingVector_B=[1.0, 0.0, 0.0], position_B=[0.75, 0.3, 0.3],
                                    fieldOfView=np.radians(50), displayName=CAMERAS[1])
    # 3 arm close-up: side-on to the arm plane (y_S = 0) from -y_S, covering the
    # rack (x_S = -0.6) to the array (x_S = 2.6)
    vizSupport.createStandardCamera(viz, spacecraftName=name_s, setMode=1,
                                    pointingVector_B=[0.05, 1.0, -0.05], position_B=[1.0, -4.6, 0.4],
                                    fieldOfView=np.radians(60), displayName=CAMERAS[2])

    # --- settings ---------------------------------------------------------------
    s = viz.settings
    s.ambient = cfg["viz"]["ambient"]
    s.showSpacecraftLabels = -1        # no per-body label switch in 2.12.2
    s.orbitLinesOn = -1                # a 7000 km orbit line cuts through close-ups
    s.trueTrajectoryLinesOn = 1
    s.truePathRelativeBody = name_c    # servicer trail relative to the client ...
    s.truePathRotatingFrame = f"earth {name_c}"  # ... in the rotating (Hill-like) frame
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
