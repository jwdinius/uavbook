# rrt dubins path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/16/2019 - RWB
import numpy as np
from message_types.msg_waypoints import MsgWaypoints
from viewers.draw_waypoints import DrawWaypoints
from viewers.draw_map import DrawMap
from planning.dubins_parameters import DubinsParameters
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from viewers.planner_viewer import PlannerViewer

SAFETY_RADIUS = 1  # meters
SAFETY_HEIGHT = 1  # meters

SEED = 0
rng = np.random.default_rng(SEED)

class RRTDubins:
    def __init__(self, app, show_planner=True):
        self.segment_length = 450  # standard length of path segments
        self.dubins_path = DubinsParameters()
        # initialize Qt gui application and window
        self.show_planner = show_planner
        self.planner_viewer = None
        if show_planner:
            self.planner_viewer = PlannerViewer(app)

    def update(self, start_pose, end_pose, Va, world_map, radius):
        self.segment_length = 4 * radius
        tree = MsgWaypoints()
        tree.type = 'dubins'

        ##### TODO #####
        # add the start pose to the tree
        tree.add(ned=start_pose[:3, 0].reshape((3, 1)), airspeed=Va, course=start_pose.item(3), cost=0, parent=-1, connect_to_goal=0)
        
        # check to see if start_pose connects directly to end_pose
        finished = False
        while not finished:
            finished = self.extendTree(tree, end_pose, Va, world_map, radius)
            # existFeasiblePath (to goal)
            if self.planner_viewer:
                self.planner_viewer.draw_tree_and_map(world_map, tree, MsgWaypoints(), MsgWaypoints(), radius, dubins_path=self.dubins_path)
                self.planner_viewer.process_app()
            if finished:
                node_parent = len(tree.cost)-1
                node_cost = tree.cost[-1] + self.distance(tree.ned[:, -1].reshape((3, 1)), tree.course[-1], end_pose[:3, 0].reshape((3, 1)), radius, end_pose.item(3))
                tree.add(ned=end_pose[:3, 0].reshape((3, 1)), airspeed=Va, course=end_pose.item(3), cost=node_cost, parent=node_parent, connect_to_goal=1)

        # find path with minimum cost to end_node
        waypoints_not_smooth = findMinimumPath(tree)
        
        # smooth path
        waypoints = self.smoothPath(waypoints_not_smooth, world_map, radius)
        if self.planner_viewer:
            self.planner_viewer.draw_tree_and_map(world_map, tree, waypoints_not_smooth, waypoints, radius)
            self.planner_viewer.process_app()
        return waypoints

    def extendTree(self, tree, end_pose, Va, world_map, radius):
        ##### TODO #####
        # extend tree by randomly selecting pose and extending tree toward that pose
        p = randomPose(world_map, end_pose.item(2)).reshape((3, 1))
        # findClosestConfiguration
        min_idx = -1
        min_distance = 5e9
        for i in range(tree.ned.shape[1]):
            v = tree.ned[:, i].reshape((3, 1))
            chi = tree.course[i]
            d = self.distance(v, chi, p, radius)
            if d < min_distance:
                min_idx = i
                min_distance = d
        v_star = tree.ned[:, min_idx].reshape((3, 1))
        chi_star = tree.course[min_idx] 
        # planPath
        direction = p - v_star
        norm_direction = np.linalg.norm(direction)
        direction /= float(norm_direction)
        v_plus = v_star + self.segment_length * direction
        chi_plus = np.arctan2(direction.item(1), direction.item(0))
        cost = self.distance(v_star, chi_star, v_plus, radius, chie=chi_plus)
        # existFeasiblePath
        connect_to_goal = 0
        if not self.collision(v_star, chi_star, v_plus, chi_plus, world_map, radius):
            node_cost = tree.cost[min_idx] + cost
            node_parent = min_idx
            d = self.distance(v_plus, chi_plus, end_pose[:3, 0].reshape((3, 1)), radius, chie=end_pose.item(3))
            
            if d < self.segment_length and not self.collision(v_plus, chi_plus, end_pose[:3, 0].reshape((3, 1)), end_pose.item(3), world_map, radius):
                connect_to_goal = 1
            else:
                connect_to_goal = 0
            tree.add(ned=v_plus, airspeed=Va, course=chi_plus, cost=node_cost, parent=node_parent, connect_to_goal=connect_to_goal)

        return (connect_to_goal == 1)

    def collision(self, start_pose, start_chi, end_pose, end_chi, world_map, radius):
        # check to see of path from start_pose to end_pose colliding with map
        ###### TODO ######
        # get distance and heading between start and end
        veh_alt = -end_pose.item(2)  # assumed == -start_pose.item(2)

        self.dubins_path.update(start_pose, start_chi, end_pose, end_chi, radius)
        # model each building as a cylinder to make collision checking easier
        R = np.sqrt(2) * world_map.building_width + SAFETY_RADIUS
        for i, n in enumerate(world_map.building_north.flatten().tolist()):
            for j, e in enumerate(world_map.building_east.flatten().tolist()):
                if veh_alt - world_map.building_height[i, j] > SAFETY_HEIGHT:
                    continue
                start_arc_collision = self.arc_collision(self.dubins_path.center_s, self.dubins_path.start_arc_angles, radius, [n, e], R)
                line_collision = self.line_collision(self.dubins_path.r1, self.dubins_path.r2, [n, e], R)
                end_arc_collision = self.arc_collision(self.dubins_path.center_e, self.dubins_path.end_arc_angles, radius, [n, e], R)
                if start_arc_collision or line_collision or end_arc_collision:
                    return True
        return False

    def arc_collision(self, center, angle_range, radius, obstacle_center, obstacle_radius):
        n, e = obstacle_center

        v = np.array([[n - center.item(0)], [e - center.item(1)]]).reshape((2, 1))
        d = np.linalg.norm(v)

        if d > radius + obstacle_radius:
            # the circles are separate
            return False
        if d < obstacle_radius - radius:
            # the obstacle contains the turning circle
            return True
        if d < radius - obstacle_radius:
            # the turning circle contains the obstacle
            return False
        if np.isclose(d, 0) and np.isclose(obstacle_radius, radius):
            # the circles are coincident
            return True

        # circles intersect at two points
        alpha = d - obstacle_radius
        beta = 0.5 * (obstacle_radius + radius - d)
        #gamma = d - radius
        delta = np.sqrt(radius**2 - (alpha + beta)**2)
        v_perp =  np.array([[v.item(1)], [-v.item(0)]]).reshape((2, 1)) / d  # perpendicular bisector of line connecting two centers
        p1 = center[:2, 0].reshape((2, 1)) + (alpha + beta) * v + delta * v_perp 
        p2 = center[:2, 0].reshape((2, 1)) + (alpha + beta) * v - delta * v_perp 

        # find angles at intersection points
        v1 = p1 - center[:2, 0].reshape((2, 1))
        v2 = p2 - center[:2, 0].reshape((2, 1))
        angle1 = mod(np.arctan2(v1.item(1), v1.item(0)))
        angle2 = mod(np.arctan2(v2.item(1), v2.item(0)))
        intersection_arc_range = [min(angle1, angle2), max(angle1, angle2)]

        # if the arc between intersection points intersects the Dubins arc,
        # the arc hits the obstacle
        arc_range = [min(angle_range), max(angle_range)]

        # if an one interval starts before the other ends, then they overlap
        return arc_range[0] <= intersection_arc_range[1] and intersection_arc_range[0] <= arc_range[1] 
    
    def line_collision(self, start_pose, end_pose, obstacle_center, obstacle_radius):
        v = end_pose[:2, 0] - start_pose[:2, 0]
        n, e = obstacle_center

        # get slope-intercept form of line between start and end pose: y = m*x + b
        m = v.item(1) / v.item(0)
        b = end_pose.item(1) - m * end_pose.item(0)
        # check intersection of line with circle
        # x is north, y is east
        # (x - n)**2 + (mx+b - e)**2 = R**2
        # (x-n)**2 + (mx + (b-e))**2 - R**2 = 0
        # (1 + m**2)*x**2 + (-2*n + 2*m*(b-e))*x + n**2 + (b-e)**2 - R**2 = 0
        A = float(1 + m**2)
        B = float(2 * m * (b - e) - 2 * n)
        C = float(n**2 + (b - e)**2 - obstacle_radius**2)
        D = float(B**2 - 4 * A * C)

        if D > 0:
            # found a collision with the infinite line, now check for collision with the segment
            # there are two intersection points in this case, need to check both
            x1 = (-B + np.sqrt(D)) / (2 * A)
            x2 = (-B - np.sqrt(D)) / (2 * A)
            y1 = m * x1 + b
            y2 = m * x2 + b
            qp1 = np.array([[x1], [y1]])
            qp2 = np.array([[x2], [y2]])
            qp1_on_segment = point_is_on_line_segment(start_pose[:2, 0].reshape((2, 1)), end_pose[:2, 0].reshape((2, 1)), qp1)
            qp2_on_segment = point_is_on_line_segment(start_pose[:2, 0].reshape((2, 1)), end_pose[:2, 0].reshape((2, 1)), qp2)
            if  qp1_on_segment or qp2_on_segment:
                # found a collision
                return True
        elif np.isclose(D, 0):
            # there is one intersection point in this case: the line is tangent to the circle
            x = -B / (2 * A)
            y = m * x + b
            qp = np.array([[x], [y]])
            qp_on_segment = point_is_on_line_segment(start_pose[:2, 0].reshape((2, 1)), end_pose[:2, 0].reshape((2, 1)), qp)
            if qp_on_segment:
                # found a collision
                return True
        return False

    def smoothPath(self, waypoints, world_map, radius):
        # smooth the waypoint path
        # add the first waypoint
        smooth_waypoints = MsgWaypoints()
        smooth_waypoints.type = waypoints.type
        smooth_waypoints.add(ned=waypoints.ned[:, 0].reshape((3, 1)), airspeed=waypoints.airspeed[0], course=waypoints.course[0], cost=0, parent=-1, connect_to_goal=1)
        smooth = [0]
        
        i, j = 0, 1
        # construct smooth waypoint path
        while j < waypoints.ned.shape[1] - 1:
            w_s = waypoints.ned[:, smooth[i]].reshape((3, 1))
            chi_s = waypoints.course[smooth[i]]
            w_plus = waypoints.ned[:, j+1].reshape((3, 1))
            chi_e = waypoints.course[j+1]
            if self.collision(w_s, chi_s, w_plus, chi_e, world_map, radius):
                smooth.append(j)
                i += 1
            j += 1
        smooth.append(waypoints.ned.shape[1] - 1)

        for i in range(len(smooth)-1):
            idx = smooth[i+1]
            idxm1 = smooth[i]
            cost = self.distance(waypoints.ned[:, idxm1].reshape((3, 1)), float(waypoints.course[idxm1]), waypoints.ned[:, idx].reshape((3, 1)), radius, chie=float(waypoints.course[idx]))
            smooth_waypoints.add(ned=waypoints.ned[:, idx].reshape((3, 1)), airspeed=waypoints.airspeed[idx], course=float(waypoints.course[idx]), cost=cost, parent=idxm1, connect_to_goal=1) 
        return smooth_waypoints
    
    def distance(self, ps, chis, pe, radius, chie=None):
        # compute distance between start and end pose
        ##### TODO #####
        v = pe - ps
        if not chie:
            chie = np.arctan2(v.item(1), v.item(0))
        self.dubins_path.update(ps, chis, pe, chie, radius)
        return self.dubins_path.length

def findMinimumPath(tree):
    # find the lowest cost path to the end node

    ##### TODO #####
    # find nodes that connect to end_node
    # find parent of end node
    t = int(tree.parent[-1])
    # -1 is parent of start node
    reversed_path = [len(tree.parent)-1]
    t_p = t
    while t != 0: 
        t = int(tree.parent[t_p])
        t_p = t
        reversed_path.append(t)

    # construct lowest cost path order
    path = list(reversed(reversed_path))  # last node that connects to end node
    # construct waypoint path
    waypoints = MsgWaypoints()
    waypoints.type = tree.type
    for idx in path:
        waypoints.add(ned=tree.ned[:, idx].reshape((3,1)), airspeed=float(tree.airspeed[idx]), course=float(tree.course[idx]), cost=float(tree.cost[idx]), parent=int(tree.parent[idx]), connect_to_goal=1) 
    return waypoints


def randomPose(world_map, pd):
    # generate a random pose
    ###### TODO ######
    sn, se = rng.random(2)
    pn = world_map.city_width * sn 
    pe = world_map.city_width * se 
    pose = np.array([[pn], [pe], [pd]])
    return pose


def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x

def point_is_on_line_segment(ps, pe, qp):
    v = (pe - ps).reshape((2,))
    m = float(v[1] / v[0])
    b = float(pe.item(1) - m * pe.item(0))

    vq = (qp - ps).reshape((2,))
    mq = float(vq[1] / vq[0])
    bq = float(qp.item(1) - mq * qp.item(0))

    if not np.isclose(m, mq) or not np.isclose(b, bq):
        # points are not collinear
        return False

    # ps + t * v = any point on the line, only points with t \in [0, 1] are on the segment
    # t * v = qp - ps -> t = (qp - ps).dot(v) / v.dot(v)
    t = float(vq.dot(v) / v.dot(v) )
    return (t >= 0 and t <= 1)

