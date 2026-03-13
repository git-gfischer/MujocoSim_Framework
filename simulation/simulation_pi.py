# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Gabriel Fischer

import sys
sys.path.append('../ProprioceptiveImage')
#sys.path.append('../deep-contact-estimator')

#from inference_mujoco import DeepContactInference

import time
import numpy as np
from tqdm import tqdm
import pprint
import copy
import pickle
import pathlib
from pprint import pprint

import mujoco

# Gym and Simulation related imports
from MujocoSim_quadruped.quadruped_env import QuadrupedEnv
from MujocoSim_quadruped.utils.quadruped_utils import LegsAttr

from MujocoSim_quadruped.utils.live_plotter import MujocoPlotter
from MujocoSim_quadruped.sensors.imu import IMU



# Helper functions for plotting
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from MujocoSim_quadruped.utils.mujoco.visual import render_vector
from MujocoSim_quadruped.utils.mujoco.visual import render_sphere

# PyMPC controller imports
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper
from quadruped_pympc.helpers.custom_procedures import MovementProcedures, SimulationController

# HeightMap import
#from MujocoSim_quadruped.sensors.heightmap import HeightMap

# Camera import
import cv2
from MujocoSim_quadruped.sensors.rgbd_camera import Camera

from collections import deque

# Neural Network imports
#import torch

# Proprioceptive Image
from ProprioceptiveImage import ProprioceptiveImage
from Visualizer import PropriceptiveImageVisualizer
from Dataset import PI_Dataset
#from ProprioceptiveImage_NN.inference import PI_Inference
#from ProprioceptiveImage_NN.utils.inf_func import print_table


# File-based procedure trigger: simulation checks this file each step.
PROCEDURE_KEY_FILE = pathlib.Path.cwd() / "procedure_key.txt"


def run_simulation( qpympc_cfg,
                    process=0, 
                    num_seconds=300,
                    return_dict=None, 
                    seed_number=0, 
                    friction_coeff=(1.0, 1.0), # (static, kinetic)
                    learning_iteration=0, 
                    render=True, 
                    proprioceptive_config="", 
                    proprioceptive_inference="",
                    ):

#if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed_number) 
    
    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    robot_leg_joints = qpympc_cfg.robot_leg_joints
    robot_feet_geom_names = qpympc_cfg.robot_feet_geom_names
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]
    # state_observables_names = ('base_pos', 'base_lin_vel', 'base_ori_euler_xyz', 'base_ori_quat_wxyz', 'base_ang_vel',
    #                            'qpos_js', 'qvel_js', 'tau_ctrl_setpoint',
    #                            'feet_pos_base', 'feet_vel_base', 'contact_state', 'contact_forces_base',)
    
    state_obs_names = ()

    state_obs_names += tuple(IMU.ALL_OBS)

    imu_kwargs = {
        'accel_name': 'imu_acc',
        'gyro_name': 'imu_gyro',
        'imu_site_name': 'imu',
        'accel_noise': 0.0,
        'gyro_noise': 0.0,
        'accel_bias_rate': 0.0,
        'gyro_bias_rate': 0.0,
    }
    imu_acc = [0,0,0]
    imu_gyro = [0,0,0]

    
    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray((0.0, 0.0)) * hip_height,
        ref_base_ang_vel=(0.0, 0.0),
        ground_friction_coeff=friction_coeff[1], # kinetic friction
        base_vel_command_type="human",
        state_obs_names=tuple(state_obs_names),
        sensors=(IMU,),
        sensors_kwargs=(imu_kwargs,),
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


    # Initialization of variables used in the main control loop --------------------------------
    # Jacobian matrices (for finite-difference Jacobian derivative)
    jac_feet_prev = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    jac_feet_dot = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    # Torque vector
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    
    # State / visualization helpers
    feet_pos = None
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]


    # Create HeightMap -----------------------------------------------------------------------
    if(qpympc_cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
        resolution_vfa = 0.04
        dimension_vfa = 7
        heightmaps = LegsAttr(FL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData),
                        FR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData),
                        RL=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData),
                        RR=HeightMap(n=dimension_vfa, dist_x=resolution_vfa, dist_y=resolution_vfa, mj_model=env.mjModel, mj_data=env.mjData))
    else:
        heightmaps = None

    # Proprioceptive Image ---------------------------------------------------------------
    proprioImg = ProprioceptiveImage(config_file=proprioceptive_config, enable_noise = True)
    proprioImgVis = PropriceptiveImageVisualizer(config_file=proprioceptive_config)
    ProprioDataset = PI_Dataset(config_file=proprioceptive_config)
    proprioceptive_image_frequency = 100# hz
    proprioceptive_data_frequency = 200 # hz
    inference_startup_delay = 200
    
    # contact states inference (leg model)
    #proprioImg_inference = PI_Inference(config_file=proprioceptive_inference)
    # inference_flag = proprioImg_inference_FL.enable
    # gt_data = 1
    # PI_acc = {"FL":0, "FR":0, "RL":0, "RR":0}
    
    # Camera------------------------------------------------------------------------------
    # cam = Camera(width=640,
    #             height=480,
    #             fps=30,
    #             model=env.robot_model,
    #             data=env.sim_data,
    #             cam_name="robotcam", # camera must be inserted on the .xml file of the robot in order to work
    #             save_dir="data_")

    # Quadruped PyMPC controller (procedure-controller style) ----------------------------
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
    print("Procedure controller (PI sim): robot started in STANCE (static hold).")
    SimulationController.print_help()
    print(f"To trigger: echo <key> > {PROCEDURE_KEY_FILE}")

    if render:
        # Ensure viewer exists before assigning callback
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        if hasattr(env.viewer, "user_key_callback"):
            env.viewer.user_key_callback = sim_controller.key_callback


    # For one-shot scratch: run a single scratch cycle per command, then return to stance
    scratch_active = False
    scratch_end_time = None
    # --------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    N_STEPS = int(num_seconds // simulation_dt)
    #N_EPISODES = 500
    N_STEPS_PER_EPISODE = 20000 if env.base_vel_command_type != "human" else 20000
    last_render_time = time.time()

    old_base_lin_vel = np.array([0,0,0])

    state_obs_history, ctrl_state_history = [], []

    # mujoco_plotter = MujocoPlotter(enable=False)
    # window_size = 200
    # mujoco_plotter.torque_plot(enable=False, window_size=window_size)
    # mujoco_plotter.jointpos_plot(enable=False, window_size=window_size)
    # mujoco_plotter.lin_acc_plot(enable=False, window_size=window_size)
    # mujoco_plotter.ang_vel_plot(enable=False, window_size=window_size)
    # mujoco_plotter.footContact_plot(enable=False, window_size=window_size, plot_per_ax=2)
    #mujoco_plotter.start()


    # Deep Contact Inference -------------------------------------------------------------
    # deep_contact = False
    # if(deep_contact):
    #     deep_contact_inference = DeepContactInference(model_path="../deep-contact-estimator/results_final2/_best_val_acc.pt")
    #     deep_contact_sequence = deque(maxlen=150)

    # Threshiold Contact Estimation -------------------------------------------------------------
    threshold_contact = False
    if(threshold_contact):
        from ProprioceptiveImage_NN.Inference.grf_th import GRFEstimator
        th_contact_inference = GRFEstimator(th=33.0)  # Threshold for contact estimation
    ep_state_history, ep_ctrl_state_history, ep_time = [], [], []
    nn_state_history = []
    nn_output_history = []
    for _ in tqdm(range(N_STEPS), desc="Procedure controller steps:", total=N_STEPS):

        # Handle procedure-controller keyboard commands (file-based trigger)
        try:
            if PROCEDURE_KEY_FILE.exists():
                content = PROCEDURE_KEY_FILE.read_text().strip()
                PROCEDURE_KEY_FILE.unlink(missing_ok=True)
                for c in content:
                    if c:
                        sim_controller.handle_key(c)
                        # One-shot scratch handling: schedule end time
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
            print("No procedure key file found")
            

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
        joints_pos = LegsAttr(FL=qpos[7:10], FR=qpos[10:13], RL=qpos[13:16], RR=qpos[16:19])
        joints_vel = LegsAttr(FL=qvel[6:9], FR=qvel[9:12], RL=qvel[12:15], RR=qvel[15:18])

        legs_mass_matrix = env.legs_mass_matrix
        legs_qfrc_bias = env.legs_qfrc_bias
        legs_qfrc_passive = env.legs_qfrc_passive

        feet_jac = env.feet_jacobians(frame="world", return_rot_jac=False)
        feet_jac_dot = env.feet_jacobians_dot(frame="world", return_rot_jac=False)

        # Quadruped PyMPC whole-body controller (procedure-controller style) ----------------
        tau = quadrupedpympc_wrapper.compute_actions(
                                                    env.com,
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
                                                    jac_feet_dot,
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

        # Apply soft torque limits
        for leg in ["FL", "FR", "RL", "RR"]:
            tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
            tau[leg] = np.clip(tau[leg], tau_min, tau_max)

    
        # Store the state and control for the episode using NMPC observables
        ctrl_state = quadrupedpympc_wrapper.get_obs()
        base_lin_vel_err = ref_base_lin_vel - base_lin_vel
        base_ang_vel_err = ref_base_ang_vel - base_ang_vel
        base_poz_z_err = ctrl_state["ref_base_height"] - base_pos[2]
        base_ang_err = ctrl_state["ref_base_angles"] - base_ori_euler_xyz
        base_ang_err[2] = 0.0  # Ignore yaw error
        joints_position = env.mjData.qpos[7:]
        joints_velocity = env.mjData.qvel[6:]
        nn_state = np.concatenate(([base_poz_z_err], base_ang_err,
                                    base_lin_vel_err, base_ang_vel_err,
                                    joints_position, joints_velocity,
                                    ctrl_state["phase_signal"]))

        

        # Create Proprioceptive Image----------------------------------------
        if env.step_num % round(1 / (proprioceptive_data_frequency * simulation_dt)) == 0:

            # convert feet_pos to base frame
            feet_pos_base_frame = env.feet_pos(frame='base')
        
            # convert feet_vel to base frame
            feet_vel_base_frame = env.feet_vel(frame='base')

        
            proprioImg.set_leg_sensor_values(sensor="FootPos", leg="LF", values= feet_pos_base_frame.FL)
            proprioImg.set_leg_sensor_values(sensor="FootPos", leg="RF", values= feet_pos_base_frame.FR)
            proprioImg.set_leg_sensor_values(sensor="FootPos", leg="LH", values= feet_pos_base_frame.RL)
            proprioImg.set_leg_sensor_values(sensor="FootPos", leg="RH", values= feet_pos_base_frame.RR)

            proprioImg.set_leg_sensor_values(sensor="FootVel", leg="LF",values =feet_vel_base_frame.FL)
            proprioImg.set_leg_sensor_values(sensor="FootVel", leg="RF",values =feet_vel_base_frame.FR)
            proprioImg.set_leg_sensor_values(sensor="FootVel", leg="LH",values =feet_vel_base_frame.RL)
            proprioImg.set_leg_sensor_values(sensor="FootVel", leg="RH",values =feet_vel_base_frame.RR)

            proprioImg.set_leg_sensor_values(sensor="JointVel", leg="LF",values=joints_vel.FL) 
            proprioImg.set_leg_sensor_values(sensor="JointVel", leg="RF",values=joints_vel.FR) 
            proprioImg.set_leg_sensor_values(sensor="JointVel", leg="LH",values=joints_vel.RL) 
            proprioImg.set_leg_sensor_values(sensor="JointVel", leg="RH",values=joints_vel.RR)

            proprioImg.set_leg_sensor_values(sensor="JointPos", leg="LF",values=joints_pos.FL) 
            proprioImg.set_leg_sensor_values(sensor="JointPos", leg="RF",values=joints_pos.FR) 
            proprioImg.set_leg_sensor_values(sensor="JointPos", leg="LH",values=joints_pos.RL) 
            proprioImg.set_leg_sensor_values(sensor="JointPos", leg="RH",values=joints_pos.RR)

            proprioImg.set_leg_sensor_values(sensor="Torque", leg="LF",values=tau.FL) 
            proprioImg.set_leg_sensor_values(sensor="Torque", leg="RF",values=tau.FR) 
            proprioImg.set_leg_sensor_values(sensor="Torque", leg="LH",values=tau.RL) 
            proprioImg.set_leg_sensor_values(sensor="Torque", leg="RH",values=tau.RR) 

        #   proprioImg.set_leg_sensor_values(sensor="GRF", leg="LF",values=nmpc_GRFs.FL)
        #   proprioImg.set_leg_sensor_values(sensor="GRF", leg="RF",values=nmpc_GRFs.FR)
        #   proprioImg.set_leg_sensor_values(sensor="GRF", leg="LH",values=nmpc_GRFs.RL)
        #   proprioImg.set_leg_sensor_values(sensor="GRF", leg="RH",values=nmpc_GRFs.RR)

            proprioImg.set_trunk_sensor_values(sensor="Lin_acc", values = imu_acc)
            proprioImg.set_trunk_sensor_values(sensor= "Ang_vel", values= imu_gyro)
            proprioImg.set_trunk_sensor_values(sensor="Ang_vel", values=[0.0,0.0,ref_base_ang_vel[2]],ref=True)


            proprioImgResult = proprioImg.get_PI_image() 
            # proprioElapsed = proprioImg.get_elapsed_time()
            # pprint.pprint(proprioElapsed)
            proprioImgVis.show_PI(proprioImgResult)
            
            # get Contact states ground truth
            GT_contact_states = env.feet_contact_state()[0]
            ProprioDataset.save(proprioImgResult, GT_contact_states)
            ProprioDataset.pickel.append(proprioImg.proprioceptive_data,GT_contact_states)

        
            # Predict contact states
            # if(env.step_num > inference_startup_delay): 
            #     list_pi_images = proprioImg_inference.preprocess(proprioImgResult)
            #     prediction,prob, pred_bin = proprioImg_inference.inference_rnn(list_pi_images)
            #     proprioImg_inference.print_contact_metrics(prediction, GT_contact_states, prob)
            # else: 
            #     prediction = None
            
            # if(prediction is not None):
            #     contact_estimate = [int(pred_bin[0]), int(pred_bin[1]),int(pred_bin[2]), int(pred_bin[3])]
            # else: contact_estimate = [-1,-1,-1,-1]

            # Deep Contact Inference ---------------------------------------------------------------
            # if(deep_contact):
            #     deep_contact_input= np.concatenate([joints_pos.FL, joints_pos.FR, joints_pos.RL, joints_pos.RR,
            #                                         joints_vel.FL, joints_vel.FR, joints_vel.RL, joints_vel.RR,
            #                                         imu_acc,
            #                                         imu_gyro,
            #                                         feet_pos.FL, feet_pos.FR, feet_pos.RL, feet_pos.RR,
            #                                         feet_vel.FL, feet_vel.FR, feet_vel.RL, feet_vel.RR,])
            #     deep_contact_sequence.append(deep_contact_input)
                
                # if(len(deep_contact_sequence) == 150):
                #     deep_contact_prediction, pred_index = deep_contact_inference.predict(deep_contact_sequence)
                #     acc, f1 = deep_contact_inference.print_contact_metrics(pred_index, GT_contact_states)
                #     print(f" Deep Contact prediction: {deep_contact_prediction} ACC: {acc} F1: {f1}")
                #     contact_estimate = [int(deep_contact_prediction[0]), int(deep_contact_prediction[1]),int(deep_contact_prediction[2]), int(deep_contact_prediction[3])]
                # else: 
                #     contact_estimate = [-1,-1,-1,-1]
            # Deep Contact Inference end ---------------------------------------------------------------

            # Estimate contact estimation based on the torques------------------------------------------
            if(threshold_contact):
                
                # get Corolis information
                h = env.legs_qfrc_bias

                # Get the jacobians from each foot
                J = env.feet_jacobians(frame='world', return_rot_jac=False) # [18x12]
                
                th_contact_inference(jacobian=J, coriolis_forces=h,joint_torques=tau)

                contact_estimate = th_contact_inference.contact_estimate()
                acc = th_contact_inference.print_contact_metrics(contact_estimate, GT_contact_states)
                contact_estimate = [int(contact_estimate['LF']), int(contact_estimate['RF']),int(contact_estimate['LH']), int(contact_estimate['RH'])]
                print("--------------------------------------------------------------------------------------------")

            #--------------------------------------------------------------------------------------------
            GT_contact_states = [int(GT_contact_states.FL), int(GT_contact_states.FR),int(GT_contact_states.RL), int(GT_contact_states.RR)]
                
                #mujoco_plotter.contact_update([GT_contact_states, contact_estimate] ,LegsAttr=[False,False])

            """nn_output = quadrupedpympc_observables["nmpc_footholds"]
            nn_output.FL = base_pos - nn_output.FL
            nn_output.FR = base_pos - nn_output.FR
            nn_output.RL = base_pos - nn_output.RL
            nn_output.RR = base_pos - nn_output.RR"""
            #nn_output = [tau]
        #------------------------------------------------End Proprioceptive Image---------------
        nn_output = np.concatenate((tau.FL, tau.FR, tau.RL, tau.RR))


        # TODO: Define frequency of append in the history
        nn_state_history.append(nn_state)
        nn_output_history.append(nn_output)

        # Set control and mujoco step ----------------------------------------------------------------------
        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL
        action[env.legs_tau_idx.FR] = tau.FR
        action[env.legs_tau_idx.RL] = tau.RL
        action[env.legs_tau_idx.RR] = tau.RR

        #action_noise = np.random.normal(0, 2, size=env.mjModel.nu)
        #action += action_noise

        # Apply the action to the environment
        state, reward, is_terminated, is_truncated, info = env.step(action=action)

        #update imu data-------------------------------
        imu_acc = state['imu_acc'] # linear acceleration
        imu_gyro = state['imu_gyro'] # angular velocity
        #------------------------------------------------

        old_base_lin_vel = base_lin_vel

        ctrl_state = quadrupedpympc_wrapper.get_obs()
        base_poz_z_err = ctrl_state["ref_base_height"] - base_pos[2]
        ctrl_state["base_poz_z_err"] = base_poz_z_err

        ep_state_history.append(state)
        ep_time.append(env.simulation_time)
        ep_ctrl_state_history.append(ctrl_state)
        _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

        # Render only at a certain frequency -----------------------------------------------------------------
        if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
            

            # Plot the swing trajectory using NMPC swing controller
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
            
            
            # # Update and Plot the heightmap
            # if(cfg.simulation_params['visual_foothold_adaptation'] != 'blind'):
            #     #if(stc.check_apex_condition(current_contact, interval=0.01)):
            #     for leg_id, leg_name in enumerate(legs_order):
            #         data = heightmaps[leg_name].data#.update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
            #         if(data is not None):
            #             for i in range(data.shape[0]):
            #                 for j in range(data.shape[1]):
            #                         heightmaps[leg_name].geom_ids[i, j] = render_sphere(viewer=env.viewer,
            #                                                                             position=([data[i][j][0][0],data[i][j][0][1],data[i][j][0][2]]),
            #                                                                             diameter=0.01,
            #                                                                             color=[0, 1, 0, .5],
            #                                                                             geom_id=heightmaps[leg_name].geom_ids[i, j]
            #                                                                             )
                        
            # Plot the GRF
            for leg_id, leg_name in enumerate(legs_order):
                feet_GRF_geom_ids[leg_name] = render_vector(env.viewer,
                                                            vector=feet_GRF[leg_name],
                                                            pos=feet_pos[leg_name],
                                                            scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                                                            color=np.array([0, 1, 0, .5]),
                                                            geom_id=feet_GRF_geom_ids[leg_name])

            env.render()
            
            # mujoco_plotter.lin_acc_update(imu_acc)
            # mujoco_plotter.ang_vel_update(imu_gyro)
            # mujoco_plotter.torque_update(tau, LegsAttr=True)
            # mujoco_plotter.jointpos_update(joints_position)


            last_render_time = time.time()

        # Auto-finish scratch after one full cycle: return to stance (static hold)
        if (
            scratch_active
            and movement_procedures.current_procedure == "scratch"
            and scratch_end_time is not None
            and env.simulation_time >= scratch_end_time
        ):
            movement_procedures.static_hold()
            scratch_active = False
            scratch_end_time = None

        # Reset the environment if the episode is terminated ------------------------------------------------
        if env.step_num >= N_STEPS_PER_EPISODE or is_terminated or is_truncated:
            if is_terminated:
                print("Environment terminated")

            ProprioDataset.pickel.save()
            proprioImg.reset()
            #proprioImg_inference.reset()
            if(threshold_contact): th_contact_inference.reset()
            #mujoco_plotter.reset()
            env.reset(random=False)
            quadrupedpympc_wrapper.wb_interface.reset(env.feet_pos(frame='world'))
            
            #return_dict['process'+str(process)+'_nn_state_history_ep'+str(episode_num)] = np.array(nn_state_history).reshape(-1, len(nn_state))
            #return_dict['process'+str(process)+'_nn_output_history_ep'+str(episode_num)] = np.array(nn_output_history).reshape(-1, len(nn_output))
            break
                
    #proprioImg_inference.contact_visualizer.close()
    #mujoco_plotter.stop()
    proprioImg.stop()
    env.close()

    return return_dict  


if __name__ == '__main__':
    
    from quadruped_pympc import config as cfg # Config imports
    qpympc_cfg = cfg
    #manager = multiprocessing.Manager()
    #return_dict = manager.dict()
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    proprioceptive_config = "../ProprioceptiveImage/config/main_config.yaml"
    proprioceptive_inference = "../ProprioceptiveImage/ProprioceptiveImage_NN/config/NN_config.yaml"
    run_simulation(qpympc_cfg=qpympc_cfg, num_seconds=300, #return_dict=return_dict, 
                   proprioceptive_config= proprioceptive_config,
                   proprioceptive_inference = proprioceptive_inference)