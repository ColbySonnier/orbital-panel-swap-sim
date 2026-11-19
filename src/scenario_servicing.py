r"""Servicer demo: the full Basilisk scenario and its Vizard replay.

    python -m src.scenario_servicing [--full-timeline] [--no-viz]

Tasks (one process; at equal times, higher priority runs first):
  env      1 s     SPICE (Earth, Sun; zero base Earth)
  dynFar   1 s  \  client + servicer (Earth + Sun point-mass gravity) and the
  dynNear  0.1 s/  servicer's extForceTorque. Exactly one is enabled: near
                   below dyn.near_m, far again above 1.25 x near_m.
  fsw      0.5 s   eclipse, then Glue: frames, power + health monitor,
                   MissionManager, force command, attitudes, rigid link, mass.
  viz      viz.rate_s  BodyWriter (extra-body states + HUD), then vizInterface.

Frames: A is inertial (the client body frame, set from the Sun direction at
t = 0 with geometry.array_dcm and held); H is Hill about the client CoM.
"""
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import eclipse, extForceTorque, spacecraft
from Basilisk.utilities import (RigidBodyKinematics as rbk, SimulationBaseClass,
                                macros, orbitalMotion, simIncludeGravBody)

from src.config import load_config
from src.geometry import (array_dcm, client_mass_props, hill_dcm, module_center,
                          module_ids)
from src.modules.arm_motion import SwapSequence, _log, berth_pose_A, fk
from src.modules.health_monitor import (HealthMonitor, PowerState, array_total,
                                        module_powers)
from src.modules.mission_manager import PHASES, MissionManager
from src.viz_setup import module_dcm_NO

REPO = Path(__file__).resolve().parents[1]
CHUNK_S = 10.0          # outer-loop step: dt switching and the stop check
BLEND_S = 180.0         # servicer attitude blend between point and berth modes
PARK_M = 1000.0         # hidden bodies sit this far below the client
HOME_TAIL_S = 600.0
FSW_S = 0.5     # keep flying this long after reaching Home
T_MAX_S = 60000.0


@dataclass
class RunLog:
    """Time series at the FSW rate (0.5 s)."""
    t: np.ndarray
    rel_H: np.ndarray          # (N, 6) servicer rel. client CoM, Hill
    rel_A: np.ndarray          # (N, 6) servicer rel. array center, A
    phase: np.ndarray
    dv_used: np.ndarray        # m/s
    array_power: np.ndarray    # W
    module_ratios: np.ndarray  # (N, n_modules) power / median
    module_ids: list
    arm_q: np.ndarray          # (N, 6) rad, NaN when the arm is idle
    sub: list
    overlay: list
    events: list               # (t, kind, module)
    phase_start: dict          # phase -> first sim time (s)
    mass: np.ndarray           # servicer hub mass (kg)
    cfg: dict = field(repr=False, default=None)
    viz: object = field(repr=False, default=None)
    wall_s: float = 0.0
    replay: Path | None = None


def _exp(w):
    """Rotation matrix of rotation vector w."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Glue(sysModel.SysModel):
    """FSW-rate glue between Basilisk and the mission modules."""

    def __init__(self, cfg, client, servicer, sun_msg, eclipse_msg, force_msg, swap):
        super().__init__()
        self.ModelTag = "glue"
        self.cfg = cfg
        self.client, self.servicer = client, servicer
        self.sun_msg, self.eclipse_msg, self.force_msg = sun_msg, eclipse_msg, force_msg
        self.swap = swap
        self.mm = MissionManager(cfg, geo={"swap": swap})
        self.rng = np.random.default_rng(cfg["seed"])
        self.pstate, self.monitor = PowerState(), HealthMonitor(cfg)
        self.target = tuple(cfg["client"]["target"])
        _, self.com, _ = client_mass_props(cfg, set())
        self.mass0 = float(cfg["servicer"]["mass_kg"])
        self.m_mod = float(cfg["client"]["module_mass_kg"])
        ev = {name: float(t) for t, name in swap.events}
        self.t_capture, self.t_release_tool = ev["capture"], ev["release_tool"]
        self.dcm_AN = None
        self.dcm_AS_berth = berth_pose_A(cfg)[1]
        self.cmd = None
        self.lock = None
        self.installed = False
        self.svc = self.passed = self.failed = False
        self.home_t = None
        self.range = None
        self.pre_fault_power = None
        self._att = {"mode": None, "t0": 0.0, "start": np.eye(3)}
        self.dcm_SN = np.eye(3)
        self.phase_start = {}
        self.L = {k: [] for k in ("t", "rel_H", "rel_A", "phase", "dv", "power", "ratios",
                                  "q", "sub", "overlay", "mass")}
        self.events = []
        self.period_ns = macros.sec2nano(FSW_S)  # set by run() to max(FSW_S, dyn dt)
        self._pos = None  # servicer state objects exist only after init

    def _bind_states(self):
        s, dm = self.servicer.hub, self.servicer.dynManager
        self._pos = dm.getStateObject(s.nameOfHubPosition)
        self._vel = dm.getStateObject(s.nameOfHubVelocity)
        self._sig = dm.getStateObject(s.nameOfHubSigma)
        self._omg = dm.getStateObject(s.nameOfHubOmega)

    # --- helpers -------------------------------------------------------------

    def _init_frames(self, r_c, v_c):
        sun = np.array(self.sun_msg.read().PositionVector) - r_c
        self.dcm_AN = array_dcm(sun / np.linalg.norm(sun), r_c, v_c)
        c = self.client.hub
        dm = self.client.dynManager
        dm.getStateObject(c.nameOfHubSigma).setState(
            [[x] for x in rbk.C2MRP(self.dcm_AN)])
        dm.getStateObject(c.nameOfHubOmega).setState([[0.0], [0.0], [0.0]])

    def _target_dcm_SN(self, mode, r_c, v_c, r_s):
        if mode == "berth":
            return self.dcm_AS_berth.T @ self.dcm_AN
        x = r_c - r_s
        x /= np.linalg.norm(x)
        h = np.cross(r_c, v_c)
        z = h - (h @ x) * x
        z /= np.linalg.norm(z)
        return np.vstack([x, np.cross(z, x), z])

    def _attitude(self, t, cmd, r_c, v_c, r_s):
        mode = "berth" if (4 <= cmd.phase <= 7 or (cmd.phase == 8 and cmd.frame == "A")) else "point"
        a = self._att
        if a["mode"] is None:
            a["mode"], a["t0"], a["start"] = mode, -BLEND_S, self.dcm_SN
        elif mode != a["mode"]:
            a["mode"], a["t0"], a["start"] = mode, t, self.dcm_SN.copy()
        tgt = self._target_dcm_SN(mode, r_c, v_c, r_s)
        s = _smooth((t - a["t0"]) / BLEND_S)
        if s >= 1.0:
            dcm = tgt
        else:
            D = tgt @ a["start"].T
            dcm = _exp(s * _log(D)) @ a["start"]
        self.dcm_SN = dcm
        self._sig.setState([[x] for x in rbk.C2MRP(dcm)])
        self._omg.setState([[0.0], [0.0], [0.0]])

    def servicer_mass(self, cmd):
        """+module mass while the damaged module is held and the spare is still
        aboard (capture .. release_tool); otherwise the base mass."""
        if cmd.phase == 6 and cmd.arm_t is not None \
                and self.t_capture <= cmd.arm_t < self.t_release_tool:
            return self.mass0 + self.m_mod
        return self.mass0

    # --- step -------------------------------------------------------------------

    def UpdateState(self, t_ns):
        # Basilisk integrates [t - dt, t] with the force it reads at t, so a
        # command must be held for a whole dynamics step: with 1 s far-range
        # dynamics the glue acts only on whole seconds (period_ns).
        if t_ns % self.period_ns:
            return
        t = t_ns * 1e-9
        cs = self.client.scStateOutMsg.read()
        r_c, v_c = np.array(cs.r_BN_N), np.array(cs.v_BN_N)
        if self.dcm_AN is None:
            self._init_frames(r_c, v_c)
            self._bind_states()
        dcm_AN = self.dcm_AN
        if self.lock is not None:  # rigid arm link: co-move with the array
            r_s = r_c + dcm_AN.T @ (self.lock - self.com)
            v_s = v_c.copy()
            self._pos.setState([[x] for x in r_s])
            self._vel.setState([[x] for x in v_s])
        else:
            ss = self.servicer.scStateOutMsg.read()
            r_s, v_s = np.array(ss.r_BN_N), np.array(ss.v_BN_N)
        dcm_HN = hill_dcm(r_c, v_c)
        n = np.linalg.norm(np.cross(r_c, v_c)) / (r_c @ r_c)
        d, dv = r_s - r_c, v_s - v_c
        rH = dcm_HN @ d
        rel_H = np.concatenate([rH, dcm_HN @ dv - np.cross([0.0, 0.0, n], rH)])
        rel_A = np.concatenate([dcm_AN @ d + self.com, dcm_AN @ dv])
        self.range = float(np.linalg.norm(d))

        sun_factor = float(self.eclipse_msg.read().shadowFactor)
        powers = module_powers(t, self.cfg, self.pstate, self.rng, strike=True,
                               sun_factor=sun_factor)
        events = self.monitor.step(t, powers)
        for e in events:
            self.events.append((t, e.kind, tuple(e.module)))
            if e.kind == "failed":
                self.pstate.open_switch.add(e.module)
            if e.kind == "service_request":
                self.svc = True
            if tuple(e.module) == self.target:
                self.passed |= e.kind == "checkout_pass"
                self.failed |= e.kind == "checkout_fail"
        cmd = self.mm.update(t, rel_H, rel_A, events, dcm_HN, dcm_AN)
        if cmd.module_installed and not self.installed:
            self.installed = True
            self.pstate.open_switch.discard(self.target)
            self.pstate.replaced.add(self.target)
            self.monitor.reset(self.target)
            self.monitor.start_checkout(self.target, t, self.cfg["holds_s"]["checkout"])
        self.phase_start.setdefault(cmd.phase, t)
        if cmd.phase == 9 and self.home_t is None:
            self.home_t = t

        pl = messaging.CmdForceInertialMsgPayload()
        pl.forceRequestInertial = list(cmd.force_N)
        self.force_msg.write(pl, t_ns, self.moduleID)
        m = self.servicer_mass(cmd)
        if m != self.servicer.hub.mHub:
            self.servicer.hub.mHub = m
        self._attitude(t, cmd, r_c, v_c, r_s)
        if cmd.thrusters_inhibited:
            if self.lock is None:
                self.lock = rel_A[:3].copy()
        else:
            self.lock = None
        self.cmd = cmd

        tot = array_total(powers)
        if t < self.cfg["power"]["strike_t_s"]:
            self.pre_fault_power = tot if self.pre_fault_power is None else self.pre_fault_power
        med = float(np.median(list(powers.values())))
        L = self.L
        L["t"].append(t); L["rel_H"].append(rel_H); L["rel_A"].append(rel_A)
        L["phase"].append(cmd.phase); L["dv"].append(self.mm.dv_used); L["power"].append(tot)
        L["ratios"].append([powers[k] / med if med > 0 else np.nan for k in self.ids])
        L["q"].append(self.swap.at(cmd.arm_t).q if cmd.arm_t is not None else np.full(6, np.nan))
        L["sub"].append(cmd.sub); L["overlay"].append(cmd.overlay); L["mass"].append(m)

    @property
    def ids(self):
        if not hasattr(self, "_ids"):
            self._ids = module_ids(self.cfg)
        return self._ids


class BodyWriter(sysModel.SysModel):
    """Viz-rate writer: inertial states of the arm, module, overlay and
    keep-out bodies, plus the HUD text."""

    def __init__(self, cfg, glue, msgs, viz=None, hud=None):
        super().__init__()
        self.ModelTag = "bodyWriter"
        self.cfg, self.g, self.msgs, self.viz, self.hud = cfg, glue, msgs, viz, hud
        swap = glue.swap
        self.frame0, self.frame_end = swap.at(0.0), swap.at(swap.duration)
        self.spare = f"spare_{cfg['arm']['rack']['spare_slot']}"
        self.tgt_A = module_center(*cfg["client"]["target"], cfg)
        self.p = messaging.SCStatesMsgPayload()
        drop = 100.0 * (1 - cfg["power"]["strike_frac_remaining"])
        r, c = cfg["client"]["target"]
        self.svc_text = f"M({r},{c}) power down {drop:.0f}%: service requested"
        self.mod_name = f"M({r},{c})"

    def _write(self, name, r_N, dcm_NO, v_N, t_ns):
        if name not in self.msgs:
            return
        p = self.p
        p.r_BN_N = list(r_N)
        p.r_CN_N = list(r_N)
        p.v_BN_N = list(v_N)
        p.v_CN_N = list(v_N)
        p.sigma_BN = list(rbk.C2MRP(dcm_NO.T))
        self.msgs[name].write(p, t_ns, self.moduleID)

    def UpdateState(self, t_ns):
        g, cmd = self.g, self.g.cmd
        if cmd is None or g.dcm_AN is None:
            return
        cs, ss = g.client.scStateOutMsg.read(), g.servicer.scStateOutMsg.read()
        r_c, v_c = np.array(cs.r_BN_N), np.array(cs.v_BN_N)
        r_s, v_s = np.array(ss.r_BN_N), np.array(ss.v_BN_N)
        dcm_NA = g.dcm_AN.T
        dcm_NS = rbk.MRP2C(np.array(ss.sigma_BN)).T
        ph, arm_t = cmd.phase, cmd.arm_t
        if arm_t is None:
            fr = self.frame0 if ph < 6 else self.frame_end
            arm_t = 0.0 if ph < 6 else g.swap.duration
        else:
            fr = g.swap.at(arm_t)
        # arm links and tool
        poses = fk(fr.q, self.cfg)
        for i in range(1, 7):
            p, R = poses[i]
            self._write(f"arm_link_{i}", r_s + dcm_NS @ p, dcm_NS @ R, v_s, t_ns)
        p, R = poses[7]
        self._write("tool", r_s + dcm_NS @ p, dcm_NS @ R, v_s, t_ns)
        # modules: seated ones ride the client, the rest the servicer
        seat_r = r_c + dcm_NA @ (self.tgt_A - g.com)
        for name, (c_S, dcm_SM) in fr.bodies.items():
            seated = ((name == "module_target" and (ph < 6 or arm_t < g.t_capture))
                      or (name == self.spare and (ph > 6 or arm_t >= g.t_release_tool)))
            if seated:
                self._write(name, seat_r, dcm_NA, v_c, t_ns)
            else:
                self._write(name, r_s + dcm_NS @ c_S, module_dcm_NO(dcm_NS, dcm_SM), v_s, t_ns)
        park = r_c - PARK_M * r_c / np.linalg.norm(r_c)
        for color in ("red", "green"):
            on = cmd.overlay == color
            self._write(f"overlay_{color}", seat_r if on else park, dcm_NA, v_c, t_ns)
        center = r_c - dcm_NA @ g.com
        self._write("keepout", center if 3 <= ph <= 8 else park, dcm_NA, v_c, t_ns)
        if self.viz is not None and self.hud is not None:
            self.hud.displayString = self.hud_text(t_ns * 1e-9, cmd, r_c, v_c, r_s, v_s)
            self.viz.vizEventDialogs.append(self.hud)

    def hud_text(self, t, cmd, r_c, v_c, r_s, v_s):
        g = self.g
        d = r_s - r_c
        rng = float(np.linalg.norm(d))
        closing = -float(d @ (v_s - v_c)) / max(rng, 1e-9)
        sub = cmd.sub
        if sub == "timeout":
            sub = "over time"
        lines = [f"Phase {cmd.phase}: {cmd.phase_name}" + (f"  ({sub})" if sub else ""),
                 f"Range {rng:,.1f} m   closing {100 * closing:+.1f} cm/s",
                 f"Delta-v used {g.mm.dv_used:.2f} m/s"]
        if g.L["power"]:
            pct = 100 * g.L["power"][-1] / g.pre_fault_power
            lines.append(f"Array power {g.L['power'][-1] / 1000:.1f} kW ({pct:.1f}% of nominal)")
        if g.failed:
            lines.append(f"{self.mod_name} CHECKOUT FAILED")
        elif g.passed:
            lines.append("Array power restored")
        elif g.svc:
            lines.append(self.svc_text)
        return "\n".join(lines)


def _initial_states(cfg):
    o = cfg["orbit"]
    mu = o["mu_km3s2"] * 1e9
    oe = orbitalMotion.ClassicElements()
    oe.a = (o["r_earth_km"] + o["alt_km"]) * 1e3
    oe.e = 0.0
    oe.i = np.radians(o["inc_deg"])
    oe.Omega = np.radians(o["raan_deg"])
    oe.omega = 0.0
    oe.f = np.radians(o["arg_lat_deg"])
    r_c, v_c = (np.array(x, float) for x in orbitalMotion.elem2rv(mu, oe))
    # servicer on the same circular orbit, phased H0 along-track behind
    h = np.cross(r_c, v_c)
    h /= np.linalg.norm(h)
    R = _exp(h * cfg["waypoints"]["H0"][1] / oe.a)
    return (r_c, v_c), (R @ r_c, R @ v_c)


def build(cfg, viz=True):
    """Assemble the simulation. Returns a dict of the parts."""
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("proc")
    dyn = cfg["dyn"]
    proc.addTask(sim.CreateNewTask("env", macros.sec2nano(1.0)), 400)
    proc.addTask(sim.CreateNewTask("dynFar", macros.sec2nano(dyn["dt_far_s"])), 300)
    proc.addTask(sim.CreateNewTask("dynNear", macros.sec2nano(dyn["dt_near_s"])), 300)
    proc.addTask(sim.CreateNewTask("fsw", macros.sec2nano(FSW_S)), 200)
    proc.addTask(sim.CreateNewTask("viz", macros.sec2nano(cfg["viz"]["rate_s"])), 100)

    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    grav.createSun()
    spice = grav.createSpiceInterface(time=cfg["orbit"]["epoch"], epochInMsg=True)
    spice.zeroBase = "Earth"
    sim.AddModelToTask("env", spice)

    (r_c, v_c), (r_s, v_s) = _initial_states(cfg)
    m_c, _, I_c = client_mass_props(cfg, set())
    client = spacecraft.Spacecraft()
    client.ModelTag = "client"
    client.hub.mHub = m_c
    client.hub.IHubPntBc_B = I_c.tolist()
    client.hub.r_CN_NInit = r_c.tolist()
    client.hub.v_CN_NInit = v_c.tolist()
    servicer = spacecraft.Spacecraft()
    servicer.ModelTag = "servicer"
    servicer.hub.mHub = float(cfg["servicer"]["mass_kg"])
    servicer.hub.IHubPntBc_B = np.diag(cfg["servicer"]["inertia"]).tolist()
    servicer.hub.r_CN_NInit = r_s.tolist()
    servicer.hub.v_CN_NInit = v_s.tolist()
    ext = extForceTorque.ExtForceTorque()
    ext.ModelTag = "servicerForce"
    servicer.addDynamicEffector(ext)
    force_msg = messaging.CmdForceInertialMsg()
    ext.cmdForceInertialInMsg.subscribeTo(force_msg)
    for sc in (client, servicer):
        grav.addBodiesTo(sc)
    for task in ("dynFar", "dynNear"):
        sim.AddModelToTask(task, ext, 20)
        sim.AddModelToTask(task, client, 10)
        sim.AddModelToTask(task, servicer, 10)

    ecl = eclipse.Eclipse()
    ecl.ModelTag = "eclipse"
    ecl.addSpacecraftToModel(client.scStateOutMsg)
    ecl.addPlanetToModel(spice.planetStateOutMsgs[0])
    ecl.sunInMsg.subscribeTo(spice.planetStateOutMsgs[1])
    sim.AddModelToTask("fsw", ecl, 20)

    swap = SwapSequence(cfg)
    glue = Glue(cfg, client, servicer, spice.planetStateOutMsgs[1], ecl.eclipseOutMsgs[0],
                force_msg, swap)
    sim.AddModelToTask("fsw", glue, 10)

    parts = dict(sim=sim, client=client, servicer=servicer, glue=glue, spice=spice,
                 grav=grav, viz=None, writer=None)
    if viz:
        from src.viz_setup import EXTRA_BODIES, hud_dialog, setup_viz
        msgs = {n: messaging.SCStatesMsg() for n in EXTRA_BODIES}
        p0 = messaging.SCStatesMsgPayload()
        p0.r_BN_N = list(r_c - PARK_M * r_c / np.linalg.norm(r_c))
        for m in msgs.values():  # valid states before the first writer call
            m.write(p0, 0)
        v = setup_viz(sim, "viz", client, servicer, msgs, cfg)
        v.epochInMsg.subscribeTo(grav.epochMsg)  # Vizard shows the real date
        hud = hud_dialog()
        writer = BodyWriter(cfg, glue, msgs, v, hud)
        sim.AddModelToTask("viz", writer, 100)
        parts.update(viz=v, writer=writer, msgs=msgs, hud=hud)
    return parts


def run(cfg=None, viz=True, outdir=REPO / "outputs", verbose=False):
    """Fly the whole mission. Returns a RunLog (and writes the replay if viz)."""
    cfg = cfg or load_config()
    wall0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if viz:
        cfg = dict(cfg, viz=dict(cfg["viz"], save_file=str(outdir / Path(cfg["viz"]["save_file"]).name)))
    P = build(cfg, viz=viz)
    sim, glue = P["sim"], P["glue"]
    near_m = cfg["dyn"]["near_m"]
    near = False
    dts = {False: cfg["dyn"]["dt_far_s"], True: cfg["dyn"]["dt_near_s"]}
    sim.disableTask("dynNear")
    glue.period_ns = macros.sec2nano(max(FSW_S, dts[near]))
    sim.InitializeSimulation()
    t = 0.0
    last_phase = None
    while t < T_MAX_S:
        t += CHUNK_S
        sim.ConfigureStopTime(macros.sec2nano(t))
        sim.ExecuteSimulation()
        r = glue.range
        if not near and r is not None and r < near_m:
            near = True
            sim.disableTask("dynFar")
            sim.enableTask("dynNear")
            glue.period_ns = macros.sec2nano(max(FSW_S, dts[near]))
        elif near and r is not None and r > 1.25 * near_m:
            near = False
            sim.disableTask("dynNear")
            sim.enableTask("dynFar")
            glue.period_ns = macros.sec2nano(max(FSW_S, dts[near]))
        if verbose and glue.cmd is not None and glue.cmd.phase != last_phase:
            last_phase = glue.cmd.phase
            print(f"  t={t:8.0f} s  phase {last_phase} {PHASES[last_phase]}"
                  f"  range {r:8.1f} m  wall {time.time() - wall0:6.1f} s", flush=True)
        if glue.home_t is not None and t - glue.home_t >= HOME_TAIL_S:
            break
    P["grav"].unloadSpiceKernels()
    L = glue.L
    log = RunLog(
        t=np.array(L["t"]), rel_H=np.array(L["rel_H"]), rel_A=np.array(L["rel_A"]),
        phase=np.array(L["phase"]), dv_used=np.array(L["dv"]),
        array_power=np.array(L["power"]), module_ratios=np.array(L["ratios"]),
        module_ids=glue.ids, arm_q=np.array(L["q"]), sub=L["sub"], overlay=L["overlay"],
        events=glue.events, phase_start=dict(glue.phase_start), mass=np.array(L["mass"]),
        cfg=cfg, viz=P["viz"])
    if viz:
        vm = P["viz"]
        log.replay = Path(vm.protoFilename)
        del P, sim  # closes the replay file
    log.wall_s = time.time() - wall0
    return log


def summary(log):
    lines = [f"wall time       {log.wall_s:8.1f} s",
             f"sim time        {log.t[-1]:8.0f} s",
             f"final phase     {log.phase[-1]} ({PHASES[log.phase[-1]]})",
             f"delta-v used    {log.dv_used[-1]:8.3f} m/s",
             f"power restored  {100 * log.array_power[-1] / log.array_power[log.t < 600].mean():8.2f} % of pre-fault"]
    if log.replay is not None and log.replay.exists():
        lines.append(f"replay          {log.replay} ({log.replay.stat().st_size / 1e6:.1f} MB)")
    lines.append("phase starts (sim s):")
    for k in sorted(log.phase_start):
        lines.append(f"  {k} {PHASES[k]:20s} {log.phase_start[k]:8.1f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full-timeline", action="store_true", help="real ConOps holds")
    ap.add_argument("--no-viz", action="store_true", help="skip the Vizard replay")
    a = ap.parse_args(argv)
    from src.plots import make_plots
    cfg = load_config(full_timeline=a.full_timeline)
    log = run(cfg, viz=not a.no_viz, verbose=True)
    paths = make_plots(log, REPO / "outputs")
    print(summary(log))
    print("plots:", *(str(p) for p in paths), sep="\n  ")


if __name__ == "__main__":
    main()
