"""
mavsim_python
    - Chapter 12 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        4/3/2019 - BGM
        2/27/2020 - RWB
        3/30/2022 - RWB
"""
import sys
sys.path.append('../..')
import pyqtgraph as pg
import parameters.simulation_parameters as SIM
import parameters.planner_parameters as PLAN
import numpy as np
from models.mav_dynamics_sensors import MavDynamics
from models.trim import compute_trim
from models.wind_simulation import WindSimulation
from control.autopilot import Autopilot
from estimation.observer import Observer
from planning.path_follower import PathFollower
from planning.path_manager import PathManager
from planning.path_planner import PathPlanner
from viewers.mav_world_viewer import MAVWorldViewer
from viewers.data_viewer import DataViewer
from tools.quit_listener import QuitListener
from message_types.msg_world_map import MsgWorldMap
quitter = QuitListener()

VIDEO = False
DATA_PLOTS = True
ANIMATION = True
PLANNING_VIEWER = True

# video initialization
if VIDEO is True:
    from viewers.video_writer import VideoWriter
    video = VideoWriter(video_name="chap12_video.avi",
                        bounding_box=(0, 0, 1000, 1000),
                        output_rate=SIM.ts_video)
    
#initialize the visualization
if ANIMATION or DATA_PLOTS:
    app = pg.QtWidgets.QApplication([]) # use the same main process for Qt applications
if ANIMATION:
    world_view = MAVWorldViewer(app=app) # initialize the viewer
if DATA_PLOTS:
    data_view = DataViewer(app=app,dt=SIM.ts_simulation, plot_period=SIM.ts_plot_refresh, 
                           data_recording_period=SIM.ts_plot_record_data, time_window_length=30)

# initialize elements of the architecture
wind = WindSimulation(SIM.ts_simulation, steady_state = np.array([[3., -3, 0]]).T)
#wind = WindSimulation(SIM.ts_simulation)
mav = MavDynamics(SIM.ts_simulation, use_biases=False, debug=False)
autopilot = Autopilot(SIM.ts_simulation, use_truth=False)
observer = Observer(SIM.ts_simulation, initial_measurements=mav.sensors())
path_follower = PathFollower()
path_manager = PathManager()
# planner_flag = 'simple_straight'  # return simple waypoint path
# planner_flag = 'simple_dubins'  # return simple dubins waypoint path
planner_flag = 'rrt_straight'  # plan path through city using straight-line RRT
# planner_flag = 'rrt_dubins'  # plan path through city using dubins RRT
path_planner = PathPlanner(app=app, planner_flag=planner_flag, show_planner=PLANNING_VIEWER)
world_map = MsgWorldMap()

# initialize the simulation time
sim_time = SIM.start_time
end_time = 200

Va = PLAN.Va0
gamma = np.radians(0)
trim_state, trim_input = compute_trim(mav, Va, gamma)
mav._state = trim_state  # set the initial state of the mav to the trim state
#building_width = PLAN.city_width / PLAN.num_blocks * (1 - PLAN.street_width)
#mav._state[0] = PLAN.city_width / 2 - building_width 
#mav._state[1] = PLAN.city_width / 2 + building_width
mav._update_true_state()

#true_state_copy = deepcopy(mav.true_state)
#autopilot.set_trim_state(true_state_copy)
#autopilot.set_trim_input(trim_input)

delta_prev = trim_input
# main simulation loop
print("Press 'Esc' to exit...")
while sim_time < SIM.end_time:
    # -------observer-------------
    measurements = mav.sensors()  # get sensor measurements
    observer.set_elevator(delta_prev.elevator)
    estimated_state = observer.update(measurements)  # estimate states from measurements
    # Observer occasionally gives bad results, true states always work.
    #estimated_state = mav.true_state
    # -------path planner - ----
    if path_manager.manager_requests_waypoints is True:
        waypoints = path_planner.update(world_map, estimated_state, PLAN.R_min)

    # -------path manager-------------
    path = path_manager.update(waypoints, PLAN.R_min, estimated_state)
    waypoints.flag_waypoints_changed = False

    # -------path follower-------------
    autopilot_commands = path_follower.update(path, estimated_state)

    # -------autopilot-------------
    delta, commanded_state = autopilot.update(autopilot_commands, estimated_state)
    delta_prev = delta

    # -------physical system-------------
    current_wind = wind.update(mav.true_state.altitude, mav.true_state.Va)  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

        # -------update viewer-------------
    if ANIMATION:
        world_view.update(mav.true_state, path, waypoints, world_map)  # plot path and MAV
    if DATA_PLOTS:
        plot_time = sim_time
        data_view.update(mav.true_state,  # true states
                         estimated_state,  # estimated states
                         commanded_state,  # commanded states
                         delta)  # inputs to aircraft
    if ANIMATION or DATA_PLOTS or PLANNING_VIEWER:
        app.processEvents()

    # -------increment time-------------
    sim_time += SIM.ts_simulation

    # -------Check to Quit the Loop-------
    if quitter.check_quit():
        break

if VIDEO is True:
    video.close()



