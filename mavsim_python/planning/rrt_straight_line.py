# rrt straight line path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/3/2019 - Brady Moon
#         4/11/2019 - RWB
#         3/31/2020 - RWB
import numpy as np
from message_types.msg_waypoints import MsgWaypoints
from viewers.planner_viewer import PlannerViewer

SAFETY_RADIUS = 3  # meters
SAFETY_HEIGHT = 3  # meters

SEED = 12021
rng = np.random.default_rng(SEED)

class RRTStraightLine:
    def __init__(self, app, show_planner=True):
        self.segment_length = 300 # standard length of path segments
        self.show_planner = show_planner
        if show_planner:
            self.planner_viewer = PlannerViewer(app)

    def update(self, start_pose, end_pose, Va, world_map, radius):
        tree = MsgWaypoints()
        # edges is a tuple: (start vertex, end vertex, cost)
        
        #tree.type = 'straight_line'
        tree.type = 'fillet'

        ###### TODO ######
        # add the start pose to the tree
        tree.add(ned=start_pose.reshape((3, 1)), airspeed=Va, course=np.inf, cost=0, parent=-1, connect_to_goal=0)
        
        # check to see if start_pose connects directly to end_pose
        finished = False
        while not finished:
            finished = self.extend_tree(tree, end_pose, Va, world_map)
            self.process_app()
            # existFeasiblePath (to goal)
            if finished:
                node_parent = len(tree.cost)-1
                node_cost = tree.cost[-1] + distance(tree.ned[-1], end_pose)
                tree.add(ned=end_pose, airspeed=Va, course=np.inf, cost=node_cost, parent=node_parent, connect_to_goal=1)

        # find path with minimum cost to end_node
        waypoints_not_smooth = find_minimum_path(tree)
        # smooth path
        waypoints = smooth_path(waypoints_not_smooth, world_map)
        return waypoints

    def extend_tree(self, tree, end_pose, Va, world_map):
        # extend tree by randomly selecting pose and extending tree toward that pose
        p = random_pose(world_map, end_pose.item(2)).reshape((3, 1))
        ###### TODO ######
        # findClosestConfiguration
        min_idx = -1
        min_distance = 5e9
        for i in range(tree.ned.shape[1]):
            v = tree.ned[:, i].reshape((3, 1))
            d = distance(v, p)
            if d < min_distance:
                min_idx = i
                min_distance = d
        v_star = tree.ned[:, min_idx].reshape((3, 1))
        # planPath
        direction = p - v_star
        norm_direction = float(np.linalg.norm(direction))
        direction /= norm_direction
        cost = min(self.segment_length, norm_direction)
        v_plus = v_star + cost * direction
        # existFeasiblePath
        connect_to_goal = 0
        if not collision(v_star, v_plus, world_map):
            node_cost = tree.cost[min_idx] + cost
            node_parent = min_idx
            d = distance(v_plus, end_pose)
            
            if d < self.segment_length and not collision(v_plus, end_pose, world_map):
                connect_to_goal = 1
            else:
                connect_to_goal = 0
            tree.add(ned=v_plus, airspeed=Va, course=np.inf, cost=node_cost, parent=node_parent, connect_to_goal=connect_to_goal)

        return (connect_to_goal == 1)
        
    def process_app(self):
        self.planner_viewer.process_app()

def waypoint_is_in_tree(point, vertices):
    for v in vertices:
        if np.allclose(point.ned, v.ned):
            return True
    return False

def smooth_path(waypoints, world_map):

    ##### TODO #####
    # smooth the waypoint path
    #smooth = [0]  # add the first waypoint
    
    # construct smooth waypoint path
    smooth_waypoints = waypoints

    return smooth_waypoints


def find_minimum_path(tree):
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
        waypoints.add(ned=tree.ned[:, idx].reshape((3,1)), airspeed=float(tree.airspeed[idx]), course=np.inf, cost=float(tree.cost[idx]), parent=int(tree.parent[idx]), connect_to_goal=1) 
    return waypoints


def random_pose(world_map, pd):
    # generate a random pose

    ##### TODO ####
    sn, se = rng.random(2)
    pn = world_map.city_width * sn 
    pe = world_map.city_width * se 
    pose = np.array([[pn], [pe], [pd]])
    return pose


def distance(start_pose, end_pose):
    # compute distance between start and end pose

    ##### TODO #####
    v = end_pose - start_pose
    return float(np.linalg.norm(v))


def collision(start_pose, end_pose, world_map):
    # check to see of path from start_pose to end_pose colliding with map
    ###### TODO ######
    # get distance and heading between start and end
    v = end_pose[:2, 0] - start_pose[:2, 0]
    veh_alt = -end_pose.item(2)  # assumed == -start_pose.item(2)

    # get slope-intercept form of line between start and end pose: y = m*x + b
    m = v.item(1) / v.item(0)
    b = end_pose.item(1) - m * end_pose.item(0)

    # model each building as a cylinder to make collision checking easier
    R = np.sqrt(2) * world_map.building_width + SAFETY_RADIUS
    for i, n in enumerate(world_map.building_north.flatten().tolist()):
        for j, e in enumerate(world_map.building_east.flatten().tolist()):
            if veh_alt - world_map.building_height[i, j] > SAFETY_HEIGHT:
                continue
            # check intersection of line with circle
            # x is north, y is east
            # (x - n)**2 + (mx+b - e)**2 = R**2
            # (x-n)**2 + (mx + (b-e))**2 - R**2 = 0
            # (1 + m**2)*x**2 + (-2*n + 2*m*(b-e))*x + n**2 + (b-e)**2 - R**2 = 0
            A = float(1 + m**2)
            B = float(2 * m * (b - e) - 2 * n)
            C = float(n**2 + (b - e)**2 - R**2)
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


def height_above_ground(world_map, point):
    # find the altitude of point above ground level
    
    ##### TODO #####
    h_agl = -point.item(2)
    return h_agl

def points_along_path(start_pose, end_pose, N):
    # returns points along path separated by Del
    points = None
    return points


def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col
