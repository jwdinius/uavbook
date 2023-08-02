"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
        2/24/2020 - RWB
"""
#AP_MODEL = "PID"
#AP_MODEL = "LQR"
AP_MODEL = "TECS"

from dotenv import load_dotenv
load_dotenv()
import os
import sys
sys.path.append(os.environ["UAVBOOK_HOME"])
from copy import deepcopy
import numpy as np
import pyqtgraph as pg
import parameters.simulation_parameters as SIM
from tools.signals import Signals
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from models.trim import compute_trim
if AP_MODEL == "LQR":
    from control.autopilot_lqr import Autopilot
elif AP_MODEL == "TECS":
    from control.autopilot_tecs import Autopilot
else:  # PID
    from control.autopilot import Autopilot
from viewers.mav_viewer import MavViewer
from viewers.data_viewer import DataViewer
from tools.quit_listener import QuitListener

quitter = QuitListener()

VIDEO = False
PLOTS = True
ANIMATION = True
SAVE_PLOT_IMAGE = False

# video initialization
if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap6_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)

#initialize the visualization
if ANIMATION or PLOTS:
    app = pg.QtWidgets.QApplication([]) # use the same main process for Qt applications
if ANIMATION:
    mav_view = MavViewer(app=app)  # initialize the mav viewer
if PLOTS:
    # initialize view of data plots
    data_view = DataViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)

# initialize elements of the architecture
#wind = WindSimulation(SIM.ts_simulation)
wind = WindSimulation(SIM.ts_simulation, steady_state=np.array([[1.5, 1, 2]]).T)
mav = MavDynamics(SIM.ts_simulation)
autopilot = Autopilot(SIM.ts_simulation)

# get the trim condition for the vehicle
# use compute_trim function to compute trim state and trim input
# for wings-level, constant speed flight
Va = 25.
gamma = 0 * np.pi/180.
trim_state, trim_input = compute_trim(mav, Va, gamma)
mav._state = trim_state  # set the initial state of the mav to the trim state
mav._update_true_state()

true_state_copy = deepcopy(mav.true_state)
if AP_MODEL == "LQR":
    autopilot.set_trim_state(true_state_copy)

autopilot.set_trim_input(trim_input)

# autopilot commands
from message_types.msg_autopilot import MsgAutopilot
commands = MsgAutopilot()
Va_command = Signals(dc_offset=Va,
                     amplitude=3.0,
                     start_time=2.0,
                     frequency=0.01)
altitude_command = Signals(dc_offset=true_state_copy.altitude,
                           amplitude=20.0,
                           start_time=0.0,
                           frequency=0.02)
course_command = Signals(dc_offset=true_state_copy.chi,
                         amplitude=np.radians(180),
                         start_time=5.0,
                         frequency=0.015)
roll_feedforward_command = Signals(dc_offset=0,
                                   amplitude=np.radians(30),
                                   start_time=3.0,
                                   frequency=0.01)

# initialize the simulation time
sim_time = SIM.start_time
end_time = 100

# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < end_time:
    '''
    Tuning steps:
    1. Fix altitude and airspeed.  Set chi_c = chi, put in step response to roll inner loop with phi_forward term
    2. Fix altitude and airspeed.  Put step response in for chi_c, set phi_forward=0
    3. Fix altitude and airspeed.  Tune sideslip washout filter to remove ringing in sideslip introduced by step response in chi
    4. Fix airspeed and chi_c (= initial heading angle) and put in step response on altitude
    5. Fix altitude and heading, and put in step response on airspeed

    In all of the above steps, set wind to zero.  As a stress test, add wind
    '''
    # -------autopilot commands-------------
    #commands.airspeed_command = Va_command.square(sim_time)
    commands.airspeed_command = Va 
    #commands.course_command = course_command.square(sim_time)
    commands.course_command = true_state_copy.chi  # XXX comment this line for Step 2: course step response
    #commands.altitude_command = altitude_command.square(sim_time)
    commands.altitude_command = true_state_copy.altitude
    #commands.phi_feedforward = roll_feedforward_command.square(sim_time)  # only needed for tuning the roll inner loop
    #commands.phi_feedforward = 0

    # -------autopilot-------------
    estimated_state = mav.true_state  # uses true states in the control
    delta, commanded_state = autopilot.update(commands, estimated_state)

    # -------physical system-------------
    #current_wind = wind.update(mav.true_state.altitude, mav.true_state.Va)  # get the new wind vector
    current_wind = np.zeros((6,1))  # when tuning the autopilot, set wind to zero
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    # ------- animation -------
    if ANIMATION:
        mav_view.update(mav.true_state)  # plot body of MAV
    if PLOTS:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                            None,  # estimated states
                            commanded_state,  # commanded states
                            delta)  # inputs to aircraft
    if ANIMATION or PLOTS:
        app.processEvents()
    if VIDEO is True:
        video.update(sim_time)
        
    # -------Check to Quit the Loop-------
    if quitter.check_quit():
        break

    # -------increment time-------------
    sim_time += SIM.ts_simulation

if SAVE_PLOT_IMAGE:
    data_view.save_plot_image("ch6_plot")

if VIDEO is True:
    video.close()




