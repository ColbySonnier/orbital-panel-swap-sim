r"""Rail-gantry panel swap: the full Basilisk scenario and its Vizard replay.

    python -m src.scenario_servicing [--full-timeline] [--no-viz] [--outdir DIR]

Tasks (one process, all at run.dt_s; at equal times higher priority runs first):
  env   SPICE (Earth, Sun; zero base Earth)
  dyn   client Spacecraft (Earth + Sun point-mass gravity). The client is the
        only Spacecraft: no servicer, no forces.
  fsw   eclipse, then Glue: power + health monitor, the mission phase machine
        and the gantry sequence clock.
  viz   BodyWriter (inertial states of every gantry/panel/overlay body + HUD),
        then vizInterface.

Frames: A is the array frame (geometry.array_dcm from the Sun direction at
t = 0), held inertially; the client body frame B = A, origin at the CoM.
Every extra body: r_N = r_client + dcm_NA @ (p_A - com_A), dcm_NB = dcm_NA @ R_A.

Mission flow (spec table):
  0 Monitoring   until service_request (red overlay on)
  1 Go / no-go   for holds_s.go
  2 Gantry swap  sequence clock 0 -> start of step roll_home (retract done)
  3 Checkout     gantry holds still, until checkout_pass (green overlay on)
  4 Return       the sequence clock resumes: roll home to the end
  5 Done         hold run.home_tail_s, then stop
"""
import argparse
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import eclipse, spacecraft
from Basilisk.utilities import (RigidBodyKinematics as rbk, SimulationBaseClass,
                                macros, orbitalMotion, simIncludeGravBody)

from src.config import load_config
from src.geometry import array_dcm, client_mass_props, module_center, module_ids
from src.modules.gantry_motion import GantrySequence
from src.modules.health_monitor import (HealthMonitor, PowerState, array_total,
                                        module_powers)

REPO = Path(__file__).resolve().parents[1]

PHASES = ["Monitoring", "Go / no-go", "Gantry swap", "Checkout", "Return", "Done"]
STEP_LABELS = {"roll_out": "Roll out", "align_waste": "Align waste bay", "grab": "Grab",
               "pull": "Pull", "shift": "Shift", "drop_in": "Drop in", "latch": "Latch",
               "retract": "Retract", "roll_home": "Roll home"}
TRACE_BODIES = ("module_target", "spare_0", "overlay_red", "overlay_green", "bridge_B")


@dataclass
class RunLog:
    """Time series, one row per glue update (every run.dt_s)."""
    t: np.ndarray
    phase: np.ndarray
    step: list                 # HUD gantry step label
    seq_t: np.ndarray          # gantry sequence clock (s)
    array_power: np.ndarray    # W
    module_ratios: np.ndarray  # (N, n_modules) power / median
    module_ids: list
    bridge_x: np.ndarray       # bridge B x in A (m)
    carriage_y: np.ndarray     # carriage y in A (m)
    fork_z: np.ndarray         # (N, 2) fork W, fork F tip z in A (m)
    overlay: list              # None | "red" | "green"
    events: list               # (t, kind, module)
    phase_start: dict          # phase -> first sim time (s)
    seq_steps: list            # [(seq_t start, step)]
    seq_duration: float
    pre_fault_power: float
    trace: dict = field(repr=False, default_factory=dict)  # replay-message read-backs
    cfg: dict = field(repr=False, default=None)
    viz_settings: dict = field(default_factory=dict)  # snapshot; the viz module is released
    wall_s: float = 0.0
    replay: Path | None = None


class Glue(sysModel.SysModel):
    """FSW-rate glue: power + health monitor, phase machine, gantry clock."""

    def __init__(self, cfg, eclipse_msg, seq):
        super().__init__()
        self.ModelTag = "glue"
        self.cfg, self.eclipse_msg, self.seq = cfg, eclipse_msg, seq
        self.rng = np.random.default_rng(cfg["seed"])
        self.pstate, self.monitor = PowerState(), HealthMonitor(cfg)
        self.target = tuple(cfg["client"]["target"])
        self.ids = module_ids(cfg)
        self.dt = float(cfg["run"]["dt_s"])
        self.t_hold = next(t for t, n in seq.steps if n == "roll_home")  # retract done
        self.phase, self.phase_start = 0, {0: 0.0}
        self.svc = self.passed = self.failed = self.installed = False
        self.fired = set()
        self.seq_t, self.frame = 0.0, seq.at(0.0)
        self.overlay = None
        self.step_label = "Parked"
        self.pre_fault = []
        self.events = []
        self.L = {k: [] for k in ("t", "phase", "step", "seq_t", "power", "ratios",
                                  "bx", "cy", "fz", "overlay")}

    @property
    def pre_fault_power(self):
        return float(np.mean(self.pre_fault)) if self.pre_fault else None

    def _enter(self, ph, t):
        self.phase = ph
        self.phase_start.setdefault(ph, t)

    def _clock(self, t):
        """Phase transitions and the gantry sequence time for this tick."""
        ph, ps = self.phase, self.phase_start
        if ph == 0 and self.svc:
            self._enter(1, t)
        if self.phase == 1 and t - ps[1] >= self.cfg["holds_s"]["go"] - 1e-9:
            self._enter(2, t)
        if self.phase == 3 and self.passed:
            self._enter(4, t)
        if self.phase == 2:
            s = t - ps[2]
            if s >= self.t_hold:
                self._enter(3, t)
                return self.t_hold
            return s
        if self.phase == 3:
            return self.t_hold
        if self.phase == 4:
            s = self.t_hold + (t - ps[4]) + self.dt
            if s >= self.seq.duration:
                self._enter(5, t)
                return self.seq.duration
            return s
        if self.phase == 5:
            return self.seq.duration
        return 0.0

    def _install(self, t):
        self.installed = True
        self.pstate.open_switch.discard(self.target)
        self.pstate.replaced.add(self.target)
        self.monitor.reset(self.target)
        self.monitor.start_checkout(self.target, t, self.cfg["holds_s"]["checkout"])

    def UpdateState(self, t_ns):
        t = t_ns * 1e-9
        sun_factor = float(self.eclipse_msg.read().shadowFactor)
        powers = module_powers(t, self.cfg, self.pstate, self.rng, strike=True,
                               sun_factor=sun_factor)
        for e in self.monitor.step(t, powers):
            self.events.append((t, e.kind, tuple(e.module)))
            if e.kind == "failed":
                self.pstate.open_switch.add(e.module)
            if tuple(e.module) == self.target:
                self.svc |= e.kind == "service_request"
                self.passed |= e.kind == "checkout_pass"
                self.failed |= e.kind == "checkout_fail"

        s = self._clock(t)
        self.seq_t = s
        for te, name in self.seq.events:
            if name not in self.fired and te <= s + 1e-9:
                self.fired.add(name)
                self.events.append((t, name, self.target))
                if name == "latched":
                    self._install(t)
        self.frame = f = self.seq.at(s)
        self.overlay = "green" if self.passed else ("red" if self.svc else None)
        if self.phase in (0, 1, 5):
            self.step_label = "Parked"
        elif self.phase == 3:
            self.step_label = "Retracted, holding"
        else:
            self.step_label = STEP_LABELS.get(f.step, f.step)

        tot = array_total(powers)
        if t < self.cfg["power"]["strike_t_s"]:
            self.pre_fault.append(tot)
        med = float(np.median(list(powers.values())))
        b = self.seq.bridge
        L = self.L
        L["t"].append(t); L["phase"].append(self.phase); L["step"].append(self.step_label)
        L["seq_t"].append(s); L["power"].append(tot)
        L["ratios"].append([powers[k] / med if med > 0 else np.nan for k in self.ids])
        L["bx"].append(f.bridge_x[b]); L["cy"].append(f.carriage_y)
        L["fz"].append([f.fork_z["W"], f.fork_z["F"]]); L["overlay"].append(self.overlay)


class BodyWriter(sysModel.SysModel):
    """Viz-rate writer: inertial states of every gantry, panel and overlay body
    (client pose o A-frame pose), the HUD text, and a read-back trace of the
    written messages for the tests."""

    def __init__(self, cfg, client, glue, msgs, viz=None, hud=None):
        super().__init__()
        self.ModelTag = "bodyWriter"
        self.cfg, self.client, self.g, self.msgs = cfg, client, glue, msgs
        self.viz, self.hud = viz, hud
        _, self.com, _ = client_mass_props(cfg, set())
        self.slot_A = module_center(*cfg["client"]["target"], cfg)
        self.park_A = self.slot_A + np.array([0.0, 0.0, -float(cfg["viz"]["park_m"])])
        self.p = messaging.SCStatesMsgPayload()
        r, c = cfg["client"]["target"]
        self.mod_name = f"M({r},{c})"
        drop = 100.0 * (1 - cfg["power"]["strike_frac_remaining"])
        self.svc_text = f"{self.mod_name} power down {drop:.0f}%: service requested"
        self.trace = {"t": [], "r_c": [], "sigma_c": [], "hud": [],
                      **{n: {"r": [], "sigma": []} for n in TRACE_BODIES}}

    def _write(self, name, r_N, dcm_NB, v_N, t_ns):
        p = self.p
        p.r_BN_N = list(r_N)
        p.r_CN_N = list(r_N)
        p.v_BN_N = list(v_N)
        p.v_CN_N = list(v_N)
        p.sigma_BN = list(rbk.C2MRP(dcm_NB.T))
        self.msgs[name].write(p, t_ns, self.moduleID)

    def UpdateState(self, t_ns):
        g = self.g
        cs = self.client.scStateOutMsg.read()
        r_c, v_c = np.array(cs.r_BN_N), np.array(cs.v_BN_N)
        dcm_NA = rbk.MRP2C(np.array(cs.sigma_BN)).T   # client B = A
        pose = lambda p_A: r_c + dcm_NA @ (np.asarray(p_A) - self.com)
        for name, (p_A, R_A) in g.frame.bodies.items():
            if name in self.msgs:
                self._write(name, pose(p_A), dcm_NA @ R_A, v_c, t_ns)
        for color in ("red", "green"):
            on = g.overlay == color
            self._write(f"overlay_{color}", pose(self.slot_A if on else self.park_A),
                        dcm_NA, v_c, t_ns)
        text = self.hud_text()
        tr = self.trace
        tr["t"].append(t_ns * 1e-9); tr["r_c"].append(list(cs.r_BN_N))
        tr["sigma_c"].append(list(cs.sigma_BN)); tr["hud"].append(text)
        for n in TRACE_BODIES:
            m = self.msgs[n].read()
            tr[n]["r"].append(list(m.r_BN_N)); tr[n]["sigma"].append(list(m.sigma_BN))
        if self.viz is not None and self.hud is not None:
            self.hud.displayString = text
            self.viz.vizEventDialogs.append(self.hud)

    def hud_text(self):
        g = self.g
        lines = [f"Phase {g.phase}: {PHASES[g.phase]}", f"Gantry: {g.step_label}"]
        if g.L["power"] and g.pre_fault_power:
            w = g.L["power"][-1]
            lines.append(f"Array power {w / 1000:.1f} kW ({100 * w / g.pre_fault_power:.1f}% of pre-fault)")
        if g.failed:
            lines.append(f"{self.mod_name} CHECKOUT FAILED")
        elif g.passed:
            lines.append("Array power restored")
        elif g.installed:
            lines.append(f"New {self.mod_name} latched: checkout running")
        elif g.svc:
            lines.append(self.svc_text)
        return "\n".join(lines)


def _initial_state(cfg):
    o = cfg["orbit"]
    mu = o["mu_km3s2"] * 1e9
    oe = orbitalMotion.ClassicElements()
    oe.a = (o["r_earth_km"] + o["alt_km"]) * 1e3
    oe.e = 0.0
    oe.i = np.radians(o["inc_deg"])
    oe.Omega = np.radians(o["raan_deg"])
    oe.omega = 0.0
    oe.f = np.radians(o["arg_lat_deg"])
    return tuple(np.array(x, float) for x in orbitalMotion.elem2rv(mu, oe))


def sun_position_at_epoch(cfg):
    """Sun position (m, Earth-centered inertial) at the epoch, from a one-step
    SPICE-only sim, so the client starts with its array attitude at t = 0."""
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("p")
    proc.addTask(sim.CreateNewTask("t", macros.sec2nano(1.0)))
    grav = simIncludeGravBody.gravBodyFactory()
    grav.createEarth().isCentralBody = True
    grav.createSun()
    spice = grav.createSpiceInterface(time=cfg["orbit"]["epoch"], epochInMsg=True)
    spice.zeroBase = "Earth"
    sim.AddModelToTask("t", spice)
    sim.InitializeSimulation()
    sim.ConfigureStopTime(0)
    sim.ExecuteSimulation()
    sun = np.array(spice.planetStateOutMsgs[1].read().PositionVector)
    grav.unloadSpiceKernels()
    return sun


def client_dcm_AN(cfg):
    """dcm_AN (= the client's dcm_BN) held for the whole run."""
    r_c, v_c = _initial_state(cfg)
    sun = sun_position_at_epoch(cfg) - r_c
    return array_dcm(sun / np.linalg.norm(sun), r_c, v_c)


def build(cfg, viz=True):
    """Assemble the simulation. Returns a dict of the parts."""
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("proc")
    dt = macros.sec2nano(cfg["run"]["dt_s"])
    for name, prio in (("env", 400), ("dyn", 300), ("fsw", 200), ("viz", 100)):
        proc.addTask(sim.CreateNewTask(name, dt), prio)

    dcm_AN = client_dcm_AN(cfg)
    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True
    grav.createSun()
    spice = grav.createSpiceInterface(time=cfg["orbit"]["epoch"], epochInMsg=True)
    spice.zeroBase = "Earth"
    sim.AddModelToTask("env", spice)

    r_c, v_c = _initial_state(cfg)
    m_c, _, I_c = client_mass_props(cfg, set())
    client = spacecraft.Spacecraft()
    client.ModelTag = "client"
    client.hub.mHub = m_c
    client.hub.IHubPntBc_B = I_c.tolist()
    client.hub.r_CN_NInit = r_c.tolist()
    client.hub.v_CN_NInit = v_c.tolist()
    client.hub.sigma_BNInit = [[x] for x in rbk.C2MRP(dcm_AN)]  # B = A, held (no torques)
    client.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    grav.addBodiesTo(client)
    sim.AddModelToTask("dyn", client, 10)

    ecl = eclipse.Eclipse()
    ecl.ModelTag = "eclipse"
    ecl.addSpacecraftToModel(client.scStateOutMsg)
    ecl.addPlanetToModel(spice.planetStateOutMsgs[0])
    ecl.sunInMsg.subscribeTo(spice.planetStateOutMsgs[1])
    sim.AddModelToTask("fsw", ecl, 20)

    seq = GantrySequence(cfg)
    glue = Glue(cfg, ecl.eclipseOutMsgs[0], seq)
    sim.AddModelToTask("fsw", glue, 10)

    names = [*seq.at(0.0).bodies, "overlay_red", "overlay_green"]
    msgs = {n: messaging.SCStatesMsg() for n in names}
    p0 = messaging.SCStatesMsgPayload()
    p0.r_BN_N = list(r_c)
    for m in msgs.values():  # valid states before the first writer call
        m.write(p0, 0)
    v = hud = None
    if viz:
        from src.viz_setup import hud_dialog, setup_viz
        v = setup_viz(sim, "viz", client, msgs, cfg)
        v.epochInMsg.subscribeTo(grav.epochMsg)  # Vizard shows the real date
        hud = hud_dialog()
    writer = BodyWriter(cfg, client, glue, msgs, v, hud)
    sim.AddModelToTask("viz", writer, 100)
    return dict(sim=sim, client=client, glue=glue, spice=spice, grav=grav, viz=v,
                writer=writer, msgs=msgs, hud=hud, seq=seq, dcm_AN=dcm_AN)


def run(cfg=None, viz=True, outdir=REPO / "outputs", verbose=False):
    """Fly the whole mission. Returns a RunLog (and writes the replay if viz)."""
    cfg = cfg or load_config()
    wall0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if viz:
        cfg = dict(cfg, viz=dict(cfg["viz"], save_file=str(outdir / Path(cfg["viz"]["save_file"]).name)))
    P = build(cfg, viz=viz)
    sim, glue, writer, seq = P["sim"], P["glue"], P["writer"], P["seq"]
    rc = cfg["run"]
    sim.InitializeSimulation()
    t, last = 0.0, None
    while t < rc["t_max_s"]:
        t += rc["chunk_s"]
        sim.ConfigureStopTime(macros.sec2nano(t))
        sim.ExecuteSimulation()
        if verbose and glue.phase != last:
            last = glue.phase
            print(f"  t={t:7.0f} s  phase {last} {PHASES[last]:12s}"
                  f"  wall {time.time() - wall0:6.1f} s", flush=True)
        if glue.phase == 5 and glue.L["t"][-1] - glue.phase_start[5] >= rc["home_tail_s"]:
            break
    P["grav"].unloadSpiceKernels()
    L = glue.L
    log = RunLog(
        t=np.array(L["t"]), phase=np.array(L["phase"]), step=L["step"],
        seq_t=np.array(L["seq_t"]), array_power=np.array(L["power"]),
        module_ratios=np.array(L["ratios"]), module_ids=glue.ids,
        bridge_x=np.array(L["bx"]), carriage_y=np.array(L["cy"]), fork_z=np.array(L["fz"]),
        overlay=L["overlay"], events=glue.events, phase_start=dict(glue.phase_start),
        seq_steps=list(seq.steps), seq_duration=float(seq.duration),
        pre_fault_power=glue.pre_fault_power, trace=writer.trace, cfg=cfg)
    if viz:
        from Basilisk.utilities import vizSupport
        vm = P["viz"]
        log.replay = Path(vm.protoFilename)
        log.viz_settings = {"ambient": vm.settings.ambient,
                            "cameras": [c.displayName for c in vizSupport.stdCameraList]}
        del vm
    # Drop every reference to the sim and vizInterface: its destructor closes
    # (and flushes) the replay stream before anyone measures the file.
    P.clear()
    del sim, glue, writer, P
    gc.collect()
    log.wall_s = time.time() - wall0
    return log


def beats(log):
    """Video beats: [(label, t0, t1)] in sim seconds."""
    ev = {}
    for t, k, _ in log.events:
        ev.setdefault(k, t)
    ps = log.phase_start
    t2 = ps[2]
    out = [("Fault (strike -> service request)", log.cfg["power"]["strike_t_s"], ev["service_request"]),
           ("Go / no-go (red on)", ps[1], ps[2])]
    steps = log.seq_steps
    for k, (s0, name) in enumerate(steps):
        if name == "roll_home":
            continue
        s1 = steps[k + 1][0]
        out.append((f"Gantry: {STEP_LABELS[name]}", t2 + s0, t2 + s1))
    out += [("Checkout hold, then green", ps[3], ps[4]),
            ("Roll home", ps[4], ps[5]),
            ("Done (tail)", ps[5], float(log.t[-1]))]
    return out


def summary(log):
    pre = log.pre_fault_power
    lines = [f"wall time       {log.wall_s:8.1f} s",
             f"sim time        {log.t[-1]:8.0f} s",
             f"final phase     {log.phase[-1]} ({PHASES[log.phase[-1]]})",
             f"power restored  {100 * log.array_power[-1] / pre:8.2f} % of pre-fault"]
    if log.replay is not None and log.replay.exists():
        lines.append(f"replay          {log.replay} ({log.replay.stat().st_size / 1e6:.1f} MB)")
    if log.viz_settings.get("cameras"):
        lines.append("cameras         " + ", ".join(log.viz_settings["cameras"]))
    lines.append("phase starts (sim s):")
    for k in sorted(log.phase_start):
        lines.append(f"  {k} {PHASES[k]:12s} {log.phase_start[k]:8.1f}")
    lines.append("events (sim s):")
    seen = set()
    for t, k, m in log.events:
        if (k, m) not in seen:
            seen.add((k, m))
            lines.append(f"  {t:8.1f}  {k} M{m}")
    lines.append("video beats (sim s):")
    for name, a, b in beats(log):
        lines.append(f"  {a:7.0f} - {b:7.0f}  {name}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full-timeline", action="store_true", help="real ConOps holds")
    ap.add_argument("--no-viz", action="store_true", help="skip the Vizard replay")
    ap.add_argument("--outdir", default=str(REPO / "outputs"), help="replay + plots folder")
    a = ap.parse_args(argv)
    from src.plots import make_plots
    cfg = load_config(full_timeline=a.full_timeline)
    log = run(cfg, viz=not a.no_viz, outdir=a.outdir, verbose=True)
    paths = make_plots(log, a.outdir)
    print(summary(log))
    print("plots:", *(str(p) for p in paths), sep="\n  ")


if __name__ == "__main__":
    main()
