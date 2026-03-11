# Description: Simulation for the procedure controller.
# The robot starts in stance (static hold). Trigger procedures by writing a key to a file
# (MuJoCo/viewer run in same process so stdin is unreliable).
#   S = static hold | F=FL G=FR H=RL J=RR = single-leg trot | C = clear (normal gait)
# Based on simulation/simulation.py
# Usage: python3 quadruped_pympc/helpers/custom_procedures/keyboard_interface.py 

import pathlib
import time
from os import PathLike
from pprint import pprint

import copy
import numpy as np

# File-based procedure trigger: sim checks this file each step. From another terminal:
#   echo f > procedure_key.txt
PROCEDURE_KEY_FILE = pathlib.Path.cwd() / "procedure_key.txt"

import mujoco

from MujocoSim_quadruped.quadruped_env import QuadrupedEnv
from MujocoSim_quadruped.utils.mujoco.visual import render_sphere, render_vector
from MujocoSim_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper
from quadruped_pympc.helpers.custom_procedures import MovementProcedures, SimulationController


def run_procedure_controller_simulation(
    qpympc_cfg,
    num_seconds=300,
    friction_coeff=(1.0, 1.0), # (static, kinetic)
    seed=0,
    render=True,
    recording_path: PathLike = None,
):
    """
    Run simulation with procedure controller: robot starts in stance, then key presses run procedures.
    """
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed)

    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    robot_leg_joints = qpympc_cfg.robot_leg_joints
    robot_feet_geom_names = qpympc_cfg.robot_feet_geom_names
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    state_obs_names = []

    # Zero ref velocity: pass (0, 0) as (min, max) so the env uses fixed zero
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray((0.0, 0.0)) * hip_height,
        ref_base_ang_vel=(0.0, 0.0),
        ground_friction_coeff=friction_coeff[1], # kinetic friction
        base_vel_command_type="human",
        state_obs_names=tuple(state_obs_names)
    )
    pprint(env.get_hyperparameters())
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant

    # Initial pose: nominal stance height (ref_z) and robot-specific joint defaults
    ref_z = qpympc_cfg.simulation_params["ref_z"]
    env.mjModel.qpos0[2] = ref_z  # base height in model default
    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    env.reset(random=False)
    # Force base height in actual state (env reset may overwrite from elsewhere)
    env.mjData.qpos[2] = float(ref_z)
    mujoco.mj_forward(env.mjModel, env.mjData)

    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]
    heightmaps = None


    quadrupedpympc_observables_names = (
        "ref_base_height",
        "ref_base_angles",
        "ref_feet_pos",
        "nmpc_GRFs",
        "nmpc_footholds",
        "swing_time",
        "phase_signal",
        "lift_off_positions",
    )

    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=quadrupedpympc_observables_names,
    )

    movement_procedures = MovementProcedures(quadrupedpympc_wrapper.wb_interface)
    sim_controller = SimulationController(movement_procedures)

    # Start in stance (static hold)
    movement_procedures.static_hold()
    print("Procedure controller: robot started in STANCE (static hold).")
    SimulationController.print_help()
    print(f"To trigger: run the keyboard interface in another terminal (same dir):  python -m quadruped_pympc.helpers.custom_procedures.keyboard_interface")
    print(f"Or manually:  echo <key> > {PROCEDURE_KEY_FILE}")

    if render:
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        if hasattr(env.viewer, "user_key_callback"):
            env.viewer.user_key_callback = sim_controller.key_callback

    if recording_path is not None:
        from MujocoSim_quadruped.utils.data.h5py import H5Writer

        root_path = pathlib.Path(recording_path)
        root_path.mkdir(exist_ok=True)
        dataset_path = (
            root_path
            / f"{robot_name}/{scene_name}"
            / f"procedure_controller_steps={int(num_seconds // simulation_dt):d}.h5"
        )
        h5py_writer = H5Writer(file_path=dataset_path, env=env, extra_obs=None)
        print(f"\nRecording data to: {dataset_path.absolute()}")
    else:
        h5py_writer = None

    RENDER_FREQ = 30  # Hz
    N_STEPS = int(num_seconds // simulation_dt)
    last_render_time = time.time()

    # For one-shot scratch: run a single scratch cycle per command, then return to normal
    scratch_active = False
    scratch_end_time = None

    ep_state_history, ep_ctrl_state_history, ep_time = [], [], []
    for _ in tqdm(range(N_STEPS), desc="Procedure controller steps:", total=N_STEPS):
        # File-based trigger: no stdin needed, works with MuJoCo single-threaded loop
        try:
            if PROCEDURE_KEY_FILE.exists():
                content = PROCEDURE_KEY_FILE.read_text().strip()
                PROCEDURE_KEY_FILE.unlink(missing_ok=True)
                for c in content:
                    if c:
                        sim_controller.handle_key(c)
                        # If this key started a scratch procedure, schedule its end time
                        if movement_procedures.current_procedure == "scratch":
                            scratch_active = True
                            scratch_end_time = env.simulation_time + float(
                                getattr(
                                    quadrupedpympc_wrapper.wb_interface.stc,
                                    "scratch_period",
                                    0.0,
                                )
                            )
        except OSError:
            pass

        # World frame (used by controller)
        feet_pos = env.feet_pos(frame="world")
        feet_vel = env.feet_vel(frame="world")
        hip_pos = env.hip_positions(frame="world")
        base_lin_vel = env.base_lin_vel(frame="world")

        # Base frame (available for logging / custom use)
        feet_pos_base = env.feet_pos(frame="base")
        feet_vel_base = env.feet_vel(frame="base")
        hip_pos_base = env.hip_positions(frame="base")
        base_lin_vel_base = env.base_lin_vel(frame="base")
        base_ang_vel = env.base_ang_vel(frame="base")
        base_ori_euler_xyz = env.base_ori_euler_xyz
        base_pos = copy.deepcopy(env.base_pos)
        com_pos = copy.deepcopy(env.com)

        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
        ref_base_lin_vel, ref_base_ang_vel = sim_controller.apply_ref_velocity_override(
            ref_base_lin_vel, ref_base_ang_vel
        )

        if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
            inertia = env.get_base_inertia().flatten()
        else:
            inertia = qpympc_cfg.inertia.flatten()

        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        legs_qvel_idx = env.legs_qvel_idx
        legs_qpos_idx = env.legs_qpos_idx
        joints_pos = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR, RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)

        legs_mass_matrix = env.legs_mass_matrix
        legs_qfrc_bias = env.legs_qfrc_bias
        legs_qfrc_passive = env.legs_qfrc_passive

        feet_jac = env.feet_jacobians(frame="world", return_rot_jac=False)
        feet_jac_dot = env.feet_jacobians_dot(frame="world", return_rot_jac=False)

        tau = quadrupedpympc_wrapper.compute_actions(
            com_pos,
            base_pos,
            base_lin_vel,
            base_ori_euler_xyz,
            base_ang_vel,
            feet_pos,
            hip_pos,
            joints_pos,
            heightmaps,
            legs_order,
            simulation_dt,
            ref_base_lin_vel,
            ref_base_ang_vel,
            env.step_num,
            qpos,
            qvel,
            feet_jac,
            feet_jac_dot,
            feet_vel,
            legs_qfrc_passive,
            legs_qfrc_bias,
            legs_mass_matrix,
            legs_qpos_idx,
            legs_qvel_idx,
            tau,
            inertia,
            env.mjData.contact,
            ref_override=movement_procedures.get_ref_override(),
        )
        for leg in ["FL", "FR", "RL", "RR"]:
            tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
            tau[leg] = np.clip(tau[leg], tau_min, tau_max)

        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL
        action[env.legs_tau_idx.FR] = tau.FR
        action[env.legs_tau_idx.RL] = tau.RL
        action[env.legs_tau_idx.RR] = tau.RR

        state, reward, is_terminated, is_truncated, info = env.step(action=action)

        ctrl_state = quadrupedpympc_wrapper.get_obs()
        base_poz_z_err = ctrl_state["ref_base_height"] - base_pos[2]
        ctrl_state["base_poz_z_err"] = base_poz_z_err

        ep_state_history.append(state)
        ep_time.append(env.simulation_time)
        ep_ctrl_state_history.append(ctrl_state)

        # Auto-finish scratch after one full cycle: return to stance (static hold),
        # not to the previous trot gait.
        if (
            scratch_active
            and movement_procedures.current_procedure == "scratch"
            and scratch_end_time is not None
            and env.simulation_time >= scratch_end_time
        ):
            movement_procedures.static_hold()
            scratch_active = False
            scratch_end_time = None

        if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
            _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

            feet_traj_geom_ids = plot_swing_mujoco(
                viewer=env.viewer,
                swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                swing_time=LegsAttr(
                    FL=ctrl_state["swing_time"][0],
                    FR=ctrl_state["swing_time"][1],
                    RL=ctrl_state["swing_time"][2],
                    RR=ctrl_state["swing_time"][3],
                ),
                lift_off_positions=ctrl_state["lift_off_positions"],
                nmpc_footholds=ctrl_state["nmpc_footholds"],
                ref_feet_pos=ctrl_state["ref_feet_pos"],
                early_stance_detector=quadrupedpympc_wrapper.wb_interface.esd,
                geom_ids=feet_traj_geom_ids,
            )

            if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
                for leg_id, leg_name in enumerate(legs_order):
                    data = heightmaps[leg_name].data
                    if data is not None:
                        for i in range(data.shape[0]):
                            for j in range(data.shape[1]):
                                heightmaps[leg_name].geom_ids[i, j] = render_sphere(
                                    viewer=env.viewer,
                                    position=([data[i][j][0][0], data[i][j][0][1], data[i][j][0][2]]),
                                    diameter=0.01,
                                    color=[0, 1, 0, 0.5],
                                    geom_id=heightmaps[leg_name].geom_ids[i, j],
                                )

            for leg_id, leg_name in enumerate(legs_order):
                feet_GRF_geom_ids[leg_name] = render_vector(
                    env.viewer,
                    vector=feet_GRF[leg_name],
                    pos=feet_pos[leg_name],
                    scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                    color=np.array([0, 1, 0, 0.5]),
                    geom_id=feet_GRF_geom_ids[leg_name],
                )

            env.render()
            last_render_time = time.time()

        if is_terminated or is_truncated:
            print("Environment terminated or truncated.")
            break

    env.close()
    if h5py_writer is not None:
        ep_obs_history = collate_obs(ep_state_history)
        ep_traj_time = np.asarray(ep_time)[:, np.newaxis]
        h5py_writer.append_trajectory(state_obs_traj=ep_obs_history, time=ep_traj_time)
        return h5py_writer.file_path


def collate_obs(list_of_dicts) -> dict[str, np.ndarray]:
    if not list_of_dicts:
        raise ValueError("Input list is empty.")
    keys = list_of_dicts[0].keys()
    collated = {key: np.stack([d[key] for d in list_of_dicts], axis=0) for key in keys}
    collated = {key: v[:, None] if v.ndim == 1 else v for key, v in collated.items()}
    return collated


if __name__ == "__main__":
    from quadruped_pympc import config as cfg

    qpympc_cfg = cfg
    run_procedure_controller_simulation(
        qpympc_cfg,
        num_seconds=300,
        render=True,
    )
