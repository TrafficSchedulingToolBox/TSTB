'''
Interactions with CityFlow, get/set values from CityFlow, pass it to RL agents
'''
import random
import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import cityflow as engine
import time
import threading
from multiprocessing import Process, Pool
from script import get_traffic_volume
from copy import deepcopy

class RoadNet:

    def __init__(self, roadnet_file):
        self.roadnet_dict = json.load(open(roadnet_file,"r"))
        self.net_edge_dict = {}
        self.net_node_dict = {}
        self.net_lane_dict = {}

        self.generate_node_dict()
        self.generate_edge_dict()
        self.generate_lane_dict()

    def generate_node_dict(self):
        '''
        node dict has key as node id, value could be the dict of input nodes and output nodes
        :return:
        '''
        #对于每一个路口node_dict['id']
        for node_dict in self.roadnet_dict['intersections']:
            node_id = node_dict['id']
            road_links = node_dict['roads']
            input_nodes = []
            output_nodes = []
            input_edges = []
            output_edges = {}
            #对于每一个道路
            for road_link_id in road_links:
                #获得该road_link_id的road字典
                road_link_dict = self._get_road_dict(road_link_id)
                if road_link_dict['startIntersection'] == node_id:
                    end_node = road_link_dict['endIntersection']
                    output_nodes.append(end_node)
                    # todo add output edges
                elif road_link_dict['endIntersection'] == node_id:
                    input_edges.append(road_link_id)
                    start_node = road_link_dict['startIntersection']
                    input_nodes.append(start_node)
                    output_edges[road_link_id] = set()
                    pass

            # update roadlinks
            actual_roadlinks = node_dict['roadLinks']
            for actual_roadlink in actual_roadlinks:
                output_edges[actual_roadlink['startRoad']].add(actual_roadlink['endRoad'])

            net_node = {
                'node_id': node_id,
                #其他点导向该节点
                'input_nodes': list(set(input_nodes)),
                #其他边导向该节点
                'input_edges': list(set(input_edges)),
                #该节点导向其他点
                'output_nodes': list(set(output_nodes)),
                #该节点导向其他边（这个信息还与之前不同，是一个字典构成）
                'output_edges': output_edges# should be a dict, with key as an input edge, value as output edges
            }
            if node_id not in self.net_node_dict.keys():
                self.net_node_dict[node_id] = net_node

    #根据road的id来返回这个road的字典
    def _get_road_dict(self, road_id):
        for item in self.roadnet_dict['roads']:
            if item['id'] == road_id:
                return item
        print("Cannot find the road id {0}".format(road_id))
        sys.exit(-1)
        # return None
    
    #根据已有路网文件来生成边的字典
    def generate_edge_dict(self):
        '''
        edge dict has key as edge id, value could be the dict of input edges and output edges
        :return:
        '''
        for edge_dict in self.roadnet_dict['roads']:
            edge_id = edge_dict['id']
            input_node = edge_dict['startIntersection']
            output_node = edge_dict['endIntersection']

            net_edge = {
                'edge_id': edge_id,
                'input_node': input_node,
                'output_node': output_node,
                'input_edges': self.net_node_dict[input_node]['input_edges'],
                'output_edges': self.net_node_dict[output_node]['output_edges'][edge_id],

            }
            if edge_id not in self.net_edge_dict.keys():
                self.net_edge_dict[edge_id] = net_edge


    #生成lane的字典，包括了该字典index，以及驶入车道，驶出车道信息
    def generate_lane_dict(self):
        lane_dict = {}
        for node_dict in self.roadnet_dict['intersections']:
            for road_link in node_dict["roadLinks"]:
                lane_links = road_link["laneLinks"]
                start_road = road_link["startRoad"]
                end_road = road_link["endRoad"]
                for lane_link in lane_links:
                    start_lane = start_road + "_" + str(lane_link['startLaneIndex'])
                    end_lane = end_road + "_" +str(lane_link["endLaneIndex"])
                    if start_lane not in lane_dict:
                        lane_dict[start_lane] = {
                            "output_lanes": [end_lane],                 #start_lane only have end_lane
                            "input_lanes": []
                        }
                    else:
                        lane_dict[start_lane]["output_lanes"].append(end_lane)
                    if end_lane not in lane_dict:
                        lane_dict[end_lane] = {
                            "output_lanes": [],
                            "input_lanes": [start_lane]                 #end_lane pnly have start_lane
                        }
                    else:
                        lane_dict[end_lane]["input_lanes"].append(start_lane)

        self.net_lane_dict = lane_dict


    def hasEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return True
        else:
            return False

    def getEdge(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return edge_id
        else:
            return None

    def getOutgoing(self, edge_id):
        if edge_id in self.net_edge_dict.keys():
            return self.net_edge_dict[edge_id]['output_edges']
        else:
            return []

    def hasInter(self, inter_id):
        if inter_id in self.net_node_dict.keys():
            return True
        else:
            return False


class Intersection:
    DIC_PHASE_MAP = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        -1: 0
    }
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log, roadnet, net):
        self.inter_id = inter_id

        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        print("inter_name:",self.inter_name)
        self.directed_graph = dic_traffic_env_conf['DIRECTED_GRAPH']
        
        self.eng = eng

        #self.roadnet = roadnet
        
        self.index = light_id_dict['inter_id_to_index'][self.inter_name]

        self.fast_compute = dic_traffic_env_conf['FAST_COMPUTE']

        self.controlled_model  = dic_traffic_env_conf['MODEL_NAME']
        self.path_to_log = path_to_log

        # =====  intersection settings =====
        self.list_approachs = ["W", "E", "N", "S"]
        self.dic_approach_to_node = {"W": 2, "E": 0, "S": 3, "N": 1}
        # self.dic_entering_approach_to_edge = {
        #    approach: "road{0}_{1}_{2}".format(self.dic_approach_to_node[approach], light_id) for approach in self.list_approachs}

        #构造驶入该路口的道路字典，此处需要根据路网实际情况进行构建
        #西方向
        
        print("Neighbors:",light_id_dict["neighbor_ENWS"])
        #西方向,"0"表示路网中不存在该路口，"-1"表示为虚拟路口
        if light_id_dict["neighbor_ENWS"][2] not in ["0","-1"]:
            self.dic_entering_approach_to_edge = {"W": light_id_dict["neighbor_ENWS"][2].replace('intersection','road')+"_0"}
        else:
            self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1]) \
                if roadnet.hasEdge("road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])) else None}
        #东方向
        if light_id_dict["neighbor_ENWS"][0] not in ["0","-1"]:
            self.dic_entering_approach_to_edge.update ({"E": light_id_dict["neighbor_ENWS"][0].replace('intersection','road')+"_2"})
        else:  
            self.dic_entering_approach_to_edge.update ({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1]) \
                if roadnet.hasEdge("road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])) else None})
        #南方向
        if light_id_dict["neighbor_ENWS"][3] not in ["0","-1"]:
            self.dic_entering_approach_to_edge.update({"S": light_id_dict["neighbor_ENWS"][3].replace('intersection','road')+"_1"})
        else:  
            self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1) \
                if roadnet.hasEdge("road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1 )) else None})
        #北方向
        if light_id_dict["neighbor_ENWS"][1] not in ["0","-1"]:
            self.dic_entering_approach_to_edge.update({"N": light_id_dict["neighbor_ENWS"][1].replace('intersection','road')+"_3"})
        else: 
            self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1) \
                if roadnet.hasEdge("road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)) else None})


        #print("entering_approach:",self.dic_entering_approach_to_edge)

        #构造驶出该路口的道路字典
        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) \
                if roadnet.hasEdge("road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach])) else None \
                for approach in self.list_approachs
        }
        #print("exiting_approach:",self.dic_exiting_approach_to_edge)
        
        self.dic_entering_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}
        self.dic_exiting_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}

        # grid settings
        self.length_lane = 300 
        self.length_grid = 5
        self.num_grid = int(self.length_lane // self.length_grid)

        self.list_phases = dic_traffic_env_conf["PHASE"][dic_traffic_env_conf['SIMULATOR_TYPE']]


        # generate all lanes
        #驶入该路口的lane的字典
        self.list_entering_lanes = []
        for approach in self.list_approachs:
            #print("entering_road:",self.dic_entering_approach_to_edge[approach])
            if self.dic_entering_approach_to_edge[approach] is not None:
                self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + '_' + str(i) for i in
                                         range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
            else:
                for i in range(sum(list(dic_traffic_env_conf["LANE_NUM"].values()))):
                    self.list_entering_lanes.append(None) 
                                         #self_dic_traffic_env_conf["LANE_NUM"] = {'LEFT': 1, 'RIGHT': 1, 'STRAIGHT': 1} 表示左右直三个方向的车道数
        #print("entering_lanes:",self.list_entering_lanes) 
        
        #驶出该路口的lane的字典
        self.list_exiting_lanes = []
        for approach in self.list_approachs:
            #print("exiting_road:",self.dic_exiting_approach_to_edge[approach])
            if self.dic_exiting_approach_to_edge[approach] is not None:
                self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in
                                        range(sum(list(dic_traffic_env_conf["LANE_NUM"].values())))]
            else:
                for i in range(sum(list(dic_traffic_env_conf["LANE_NUM"].values()))):
                    self.list_exiting_lanes.append(None)

        #print("exiting_lanes:",self.list_exiting_lanes)


        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        #top_k个邻接路口的编号id，该示例中仅仅有3个邻接路口，不够的则用-1填充
        self.adjacency_row = light_id_dict['adjacency_row']
        #邻接路口的id
        self.neighbor_ENWS = light_id_dict['neighbor_ENWS']
        #邻接的驶入该路口的lane的相关属性
        self.neighbor_lanes_ENWS = light_id_dict['entering_lane_ENWS']

        def _get_top_k_lane(lane_id_list, top_k_input):
            top_k_lane_indexes = []
            for i in range(top_k_input):
                lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                top_k_lane_indexes.append(lane_id)
            return top_k_lane_indexes

        self._adjacency_row_lanes = {}
        # _adjacency_row_lanes is the lane id, not index
        #对于该路口的驶入车道，构建其驶入驶出车道信息。即构建邻居的邻居信息
        for lane_id in self.list_entering_lanes:
            #加入存在则添加该路径，不存在则添加None
            if lane_id in light_id_dict['adjacency_matrix_lane']:
                self._adjacency_row_lanes[lane_id] = light_id_dict['adjacency_matrix_lane'][lane_id]
            else:
                self._adjacency_row_lanes[lane_id] = [_get_top_k_lane([], dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]),
                                                 _get_top_k_lane([], dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"])]
        # order is the entering lane order, each element is list of two lists

        #临接lane字典，从id到index的映射
        self.adjacency_row_lane_id_local = {}
        for index, lane_id in enumerate(self.list_entering_lanes):
            self.adjacency_row_lane_id_local[lane_id] = index

        # previous & current state
        #车道车辆，车道等待车辆数，速度，距离的dict
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_count_previous_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}        
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}
        
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_count_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        #车道车辆，车道等待车辆数的list
        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        '''
        新增如下一些属性：包括车道上的特殊车辆，车道正在等待的特殊车辆的等待时间，特殊车辆速度，特殊车辆距离
        '''
        ##############################################################################
        self.dic_lane_sp_vehicle_previous_step = {}
        self.dic_lane_sp_vehicle_count_previous_step = {}
        self.dic_lane_waiting_sp_vehicle_count_previous_step = {}        
        self.dic_sp_vehicle_speed_previous_step = {}
        self.dic_sp_vehicle_distance_previous_step = {}
        
        self.dic_lane_sp_vehicle_current_step = {}
        self.dic_lane_sp_vehicle_count_current_step = {}
        self.dic_lane_waiting_sp_vehicle_count_current_step = {}
        self.dic_sp_vehicle_speed_current_step = {}
        self.dic_sp_vehicle_distance_current_step = {}
        
        
        ##############################################################################

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        
        #设置交通灯状态？？
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        
        path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        #获取当前时间，以及当前阶段索引，存储成csv文件
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)
        
        
        self.next_phase_to_set_index = None
        #用来计算当前阶段的持续时间
        self.current_phase_duration = -1
        
        #新增属性
        
        
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        #车辆最小速度，到达离开时间
        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative
        
        
        #对于一个交通灯的feature
        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second


    #将_adjacency_row_lanes中的id转化为adjacency_row_lanes中的索引
    def build_adjacency_row_lane(self, lane_id_to_global_index_dict):
        #print("lane_id_to_global_index_dict:",lane_id_to_global_index_dict)
        self.adjacency_row_lanes = [] # order is the entering lane order, each element is list of two lists
        for entering_lane_id in self.list_entering_lanes:
            _top_k_entering_lane, _top_k_leaving_lane = self._adjacency_row_lanes[entering_lane_id]
            top_k_entering_lane = []
            top_k_leaving_lane = []
            for lane_id in _top_k_entering_lane:
                top_k_entering_lane.append(lane_id_to_global_index_dict[lane_id] if lane_id is not None else -1)
            for lane_id in _top_k_leaving_lane:
                top_k_leaving_lane.append(lane_id_to_global_index_dict[lane_id]
                                          if (lane_id is not None) and (lane_id in lane_id_to_global_index_dict.keys())  # TODO leaving lanes of system will also have -1
                                          else -1)
            self.adjacency_row_lanes.append([top_k_entering_lane, top_k_leaving_lane])
            
    # set
    #action_pattern分为切换状态或者直接set到某一状态
    def set_signal(self, action, action_pattern, yellow_time, all_red_time):
        #假如当前状态是黄灯
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            #假如黄灯持续时间达到最大值
            if self.current_phase_duration >= yellow_time: # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index) # if multi_phase, need more adjustment
                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)
                self.all_yellow_flag = False
            else:
                pass
        #当前状态不是黄灯
        else:
            # determine phase
            if action_pattern == "switch": # switch by order
                if action == 0: # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1: # change to the next phase，有设置好的4个固定的状态
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases) # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set": # set to certain phase
                self.next_phase_to_set_index = self.DIC_PHASE_MAP[action] # if multi_phase, need more adjustment

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index: # the light phase keeps unchanged
                pass
            else: # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                #这个地方没有看到哪里去切换current_phase_index的状态，这个地方是交通灯置黄
                self.eng.set_tl_phase(self.inter_name, 0) # !!! yellow, tmp
                path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode='a', header=False, index=False)
                #traci.trafficlights.setRedYellowGreenState(
                #    self.node_light, self.all_yellow_phase_str)
                #下面的三个变量是如何影响交通灯变化的？？
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1


    # update inner measurements
    def update_previous_measurements(self):
        #当前阶段状态
        self.previous_phase_index = self.current_phase_index
        #每个车道上的车辆
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_count_previous_step = self.dic_lane_vehicle_count_current_step
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step
        #每个车道上的特殊车辆信息
        self.dic_lane_sp_vehicle_previous_step = self.dic_lane_sp_vehicle_current_step
        self.dic_lane_sp_vehicle_count_previous_step = self.dic_lane_sp_vehicle_count_current_step
        self.dic_lane_waiting_sp_vehicle_count_previous_step = self.dic_lane_waiting_sp_vehicle_count_current_step
        self.dic_sp_vehicle_speed_previous_step = self.dic_sp_vehicle_speed_current_step
        self.dic_sp_vehicle_distance_previous_step = self.dic_sp_vehicle_distance_current_step
        #更新每个当前的动态有向图
        
        
    def update_current_measurements_map(self, list_sp_vehicle, simulator_state, graph_matrix):
        ## need change, debug in seeing format
        # 更新一次动作所带来的系统的整体的内部属性变化，RL仅需要提取其中一部分表层特征作为输入即可，但系统迭代的信息需要全局更新
        
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.values():
                if value is not None:
                    list_lane_vehicle.extend(value)

            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        '''
        更新自身驶出车道和驶入车道的所有车辆信息，等待车辆数，车速，行驶距离四个特征:
        self.current_phase_duration
        self.dic_lane_vehicle_current_step
        self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_current_step
        '''
       
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane] if lane is not None else None
            self.dic_lane_vehicle_count_current_step[lane] = len(self.dic_lane_vehicle_current_step[lane]) if lane is not None else -1
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane] if lane is not None else -1

        for lane in self.list_exiting_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane] if lane is not None else None
            self.dic_lane_vehicle_count_current_step[lane] = len(self.dic_lane_vehicle_current_step[lane]) if lane is not None else -1
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane] if lane is not None else -1

        if not self.fast_compute:
            self.dic_vehicle_speed_current_step = simulator_state['get_vehicle_speed']
            self.dic_vehicle_distance_current_step = simulator_state['get_vehicle_distance']

        #更新自身驶出车道和驶入车道的特殊车辆信息，包括等待车辆数，特殊车辆车速，特殊车辆行驶距离
        #获取所有lane上的车辆的id
        '''
        self.dic_lane_sp_vehicle_current_step                   车道上的特殊车辆信息
        self.dic_lane_waiting_sp_vehicle_count_current_step     正在等待的车辆数
        self.dic_sp_vehicle_speed_current_step                  特殊车辆的速度
        self.dic_sp_vehicle_distance_current_step               特殊车辆在当前这个lane上行驶了多少米，表示的是位置
        '''
################################################list_entering_lanes##############################################           
        #初始化各个lane特殊车辆信息和停止车辆数
        for lane in self.list_entering_lanes:
            self.dic_lane_sp_vehicle_current_step[lane] = []
            self.dic_lane_waiting_sp_vehicle_count_current_step[lane] = 0
            
        #更新lane的特殊车辆信息     
        for lane in self.list_entering_lanes:
            if lane is not None:
                for vehicle in self.dic_lane_vehicle_current_step[lane]:
                    if vehicle in list_sp_vehicle and vehicle not in self.dic_lane_sp_vehicle_current_step[lane]:
                        self.dic_lane_sp_vehicle_current_step[lane].append(vehicle) 
            
            #根据lane上该车辆速度，判断其是否属于等待车辆
                if self.fast_compute:
                    self.dic_lane_sp_vehicle_count_current_step[lane] = len(self.dic_lane_sp_vehicle_current_step[lane])
                else:
                    for vehicle in self.dic_lane_sp_vehicle_current_step[lane]:
                        if self.dic_vehicle_speed_current_step[vehicle] <= 0.1:
                            self.dic_lane_waiting_sp_vehicle_count_current_step[lane] += 1
                    
################################################list_exiting_lanes################################################### 
        #初始化各个lane特殊车辆信息和停止车辆数
        for lane in self.list_exiting_lanes:
            self.dic_lane_sp_vehicle_current_step[lane] = []
            self.dic_lane_waiting_sp_vehicle_count_current_step[lane] = 0
            
        #更新lane的特殊车辆信息     
        for lane in self.list_exiting_lanes:
            if lane is not None:
                for vehicle in self.dic_lane_vehicle_current_step[lane]:
                    if vehicle in list_sp_vehicle and vehicle not in self.dic_lane_sp_vehicle_current_step[lane]:
                        self.dic_lane_sp_vehicle_current_step[lane].append(vehicle) 
            
                #根据lane上该车辆速度，判断其是否属于等待车辆
                if self.fast_compute:
                    self.dic_lane_sp_vehicle_count_current_step[lane] = len(self.dic_lane_sp_vehicle_current_step[lane]) is lane is not None
                else:
                    for vehicle in self.dic_lane_sp_vehicle_current_step[lane]:
                        if self.dic_vehicle_speed_current_step[vehicle] <= 0.1:
                            self.dic_lane_waiting_sp_vehicle_count_current_step[lane] += 1
        

        #将dict转化为list，list为该路口的所有车道上的车辆的合集
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)
        self.list_lane_sp_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_sp_vehicle_current_step)
        self.list_lane_sp_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_sp_vehicle_previous_step)
        
        #获得新驶入车道和新离开车道的车辆
        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        #获得新驶入车道和新离开车道的sp车辆
        list_sp_vehicle_new_arrive = list(set(self.list_lane_sp_vehicle_current_step) - set(self.list_lane_sp_vehicle_previous_step))
        list_sp_vehicle_new_left = list(set(self.list_lane_sp_vehicle_previous_step) - set(self.list_lane_sp_vehicle_current_step))
        
        #get vehicle list，根据车道信息进行区分，详细显示每个车道上的离开车辆的信息
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        #更新所有车辆的到达和离开的时间，特殊车辆也可以通过索引得到
        self._update_arrive_time(list_vehicle_new_arrive, list_sp_vehicle)
        self._update_left_time(list_vehicle_new_left_entering_lane)

        # update vehicle minimum speed in history, # to be implemented
        #############如果使用动态有向图，此处的adjacency不是静态的，而是动态的，所以需要进行更新
        #print("directed_graph:",self.directed_graph)
        if self.directed_graph:
            self.adjacency_row = graph_matrix[self.index]
        
        #print("sp_vehicle_current_step:",self.dic_lane_sp_vehicle_current_step)
        #print("sp_vehicle_speed:",self.dic_sp_vehicle_speed_current_step)
        #print("sp_vehicle_distance:",self.dic_sp_vehicle_distance_current_step)
        #print("waiting_sp_vehicle_count:",self.dic_lane_waiting_sp_vehicle_count_current_step)
        
        #更新全部特征，导入信息是simulator_state,更新经过加工的相关信息
        self._update_feature_map(simulator_state)


    #此函数在哪里调用仍需探究
    def update_current_measurements(self, list_sp_vehicle):
        ## need change, debug in seeing format
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)

            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1


        self.dic_lane_vehicle_current_step =[] # = self.eng.get_lane_vehicles()
        #not implement
        flow_tmp = self.eng.get_lane_vehicles()
        self.dic_lane_vehicle_current_step = {key: None for key in self.list_entering_lanes}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = flow_tmp[lane]

        self.dic_lane_waiting_vehicle_count_current_step = self.eng.get_lane_waiting_vehicle_count()
        self.dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
        self.dic_vehicle_distance_current_step = self.eng.get_vehicle_distance()

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive, list_sp_vehicle)
        self._update_left_time(list_vehicle_new_left_entering_lane)

        # update vehicle minimum speed in history, # to be implemented
        #self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    
    #返回一个驶入车道的离开车辆list
    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:
        #构造每一个lane的list
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                if lane is not None:
                    last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                    current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                    list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
                )

        return list_entering_lane_vehicle_left



    def _update_arrive_time(self, list_vehicle_arrive, list_sp_vehicle):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                if vehicle in list_sp_vehicle:
                    self.dic_vehicle_arrive_leave_time[vehicle] = \
                        {"enter_time": ts, "leave_time": np.nan, "sp_vehicle": 1}
                else:
                    self.dic_vehicle_arrive_leave_time[vehicle] = \
                        {"enter_time": ts, "leave_time": np.nan, "sp_vehicle": 0}                    
            else:
                #print("vehicle: %s already exists in entering lane!"%vehicle)
                #sys.exit(-1)
                pass



    def _update_left_time(self, list_vehicle_left):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)


    #传入的为邻居的class object以及dic_feature
    def update_neighbor_info(self, neighbors, dic_feature):
        # print(dic_feature)
        '''
                inter.dic_feature格式如下
                { 'cur_phase': [1], 
                  'time_this_phase': [0], 
                  'vehicle_position_img': None, 
                  'vehicle_speed_img': None, 
                  'vehicle_acceleration_img': None, 
                  'vehicle_waiting_time_img': None, 
                  'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'lane_num_sp_vehicle':[ , , , , , , , , , ]
                  'pressure': None,
                  'coming_vehicle': None, 
                  'leaving_vehicle': None, 
                  'lane_num_vehicle_been_stopped_thres01': None, 
                  'lane_num_vehicle_been_stopped_thres1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  'lane_queue_length': None, 
                  'lane_num_vehicle_left': None, 
                  'lane_sum_duration_vehicle_left': None, 
                  'lane_sum_waiting_time': None, 
                  'terminal': None, 
                  'adjacency_matrix': [2, 5, -1, -1, 1], 
                  'adjacency_matrix_lane': [[[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], 
                                            [[-1, -1, -1, -1, -1, -1], [60, 61, 62, -1, -1, -1]], 
                                            [[-1, -1, -1, -1, -1, -1], [18, 19, 20, -1, -1, -1]], 
                                            [[69, 64, 68, -1, -1, -1], [18, 19, 20, -1, -1, -1]], 
                                            [[69, 64, 68, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], 
                                            [[69, 64, 68, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], 
                                            [[-1, -1, -1, -1, -1, -1], [60, 61, 62, -1, -1, -1]], 
                                            [[-1, -1, -1, -1, -1, -1], [18, 19, 20, -1, -1, -1]], 
                                            [[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], 
                                            [[12, 22, 17, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], 
                                            [[12, 22, 17, -1, -1, -1], [-1, -1, -1, -1, -1, -1]], 
                                            [[12, 22, 17, -1, -1, -1], [60, 61, 62, -1, -1, -1]]], 
                  'connectivity': [1.0, 0.0, 0.0, 0.0, 0.0]}
        '''
        none_dic_feature = deepcopy(dic_feature)
        for key in none_dic_feature.keys():
            if none_dic_feature[key] is not None:
                if "cur_phase" in key:
                    none_dic_feature[key] = [1] * len(none_dic_feature[key])
                else:
                    none_dic_feature[key] = [0] * len(none_dic_feature[key])
            else:
                none_dic_feature[key] = None
        for i in range(len(neighbors)):
            neighbor = neighbors[i]
            example_dic_feature = {}
            #邻居不存在则赋值为该路口的默认？是否合理，原值是什么
            if neighbor is None:
                example_dic_feature["cur_phase_{0}".format(i)] = none_dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = none_dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = none_dic_feature["lane_num_vehicle"]
                
            #否则获取邻居路口的状态
            else:
                example_dic_feature["cur_phase_{0}".format(i)] = neighbor.dic_feature["cur_phase"]
                example_dic_feature["time_this_phase_{0}".format(i)] = neighbor.dic_feature["time_this_phase"]
                example_dic_feature["lane_num_vehicle_{0}".format(i)] = neighbor.dic_feature["lane_num_vehicle"]
            dic_feature.update(example_dic_feature)
        #增加以下的相关信息，即四个邻居的当前状态，状态持续时间，以及驶入车道上的车辆数
        '''
        'cur_phase_0': [1],
        'time_this_phase_0': [0],
        'lane_num_vehicle_0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lane_num_sp_vehicle_0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lane_num_sp_vehicle_waiting_count_0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        'cur_phase_1': [1],
        'time_this_phase_1': [0], 
        'lane_num_vehicle_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'lane_num_sp_vehicle_1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        'cur_phase_2': [1], 
        'time_this_phase_2': [0], 
        'lane_num_vehicle_2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'lane_num_sp_vehicle_2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        
        'cur_phase_3': [1], 
        'time_this_phase_3': [0], 
        'lane_num_vehicle_3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        'lane_num_sp_vehicle_3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        '''
         
        return dic_feature


    @staticmethod
    def _add_suffix_to_dict_key(target_dict, suffix):
        keys = list(target_dict.keys())
        for key in keys:
            target_dict[key+"_"+suffix] = target_dict.pop(key)
        return target_dict


############################需要在此处更新调度策略可能使用的特征###############################
    def _update_feature_map(self, simulator_state):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None #self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None #self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None #self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes) 

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes) #[0,1,5,4,2,5,7,2,4,5,1,6]
                
        dic_feature['lane_num_sp_vehicle'] = self._get_lane_num_sp_vehicle(self.list_entering_lanes) #[]
        
        dic_feature["pressure"] = [self._get_pressure()]   #None 
        dic_feature["sp_pressure"] = [self._get_sp_pressure()]

        if self.fast_compute:
            dic_feature["coming_vehicle"] = None
            dic_feature["leaving_vehicle"] = None
        else:
            dic_feature["coming_vehicle"] = self._get_coming_vehicles(simulator_state)
            dic_feature["leaving_vehicle"] = self._get_leaving_vehicles(simulator_state)



        dic_feature["lane_num_vehicle_been_stopped_thres01"] = None # self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature["lane_num_sp_vehicle_been_stopped_thres1"] = self._get_lane_num_sp_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature["lane_queue_length"] =  self._get_lane_queue_length(self.list_entering_lanes) #None 
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = None #self._get_lane_sum_waiting_time(self.list_entering_lanes)
        dic_feature["terminal"] = None


        dic_feature["adjacency_matrix"] = self._get_adjacency_row() # TODO this feature should be a dict? or list of lists
        
        dic_feature["adjacency_matrix_lane"] = self._get_adjacency_row_lane() #row: entering_lane # columns: [inputlanes, outputlanes]
        #dic_feature["adjacency_matrix_lane"]规模为12*2*top_k_(12为该路口的驶入lane,2*top_k为该lane的驶入和驶出的top_k个lane)
        dic_feature['connectivity'] = self._get_connectivity(self.neighbor_lanes_ENWS)

        self.dic_feature = dic_feature


    # ================= calculate features from current observations ======================
    def _get_adjacency_row(self):
        return self.adjacency_row



    def _get_adjacency_row_lane(self):
        return self.adjacency_row_lanes



    def lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter


    #在杭州数据集上，coming_vehicle每三个的和不等于lane_num_vehicle的值，推测因为bins只统计300m车辆,路网道路长度较长。
    def _get_coming_vehicles(self, simulator_state):
        ## TODO f vehicle position   eng.get_vehicle_distance()  ||  eng.get_lane_vehicles()

        coming_distribution = []
        ## dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        ## TODO LANE LENGTH = 300，行驶车道分为3段，然后更详细地统计每一段上分布地车辆数        
        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_entering_lanes:
            if lane is not None:
                coming_vehicle_position = []
                vehicle_position_lane = lane_vid_mapping_dict[lane]
                for vehicle in vehicle_position_lane:
                #加入每一个vehicle在道路上的行驶里程？
                    coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
                coming_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))
        
        #shape为shape(list_entering_lanes)*3
        return coming_distribution

    def _get_leaving_vehicles(self, simulator_state):
        leaving_distribution = []
        ## dimension = num_lane*3*num_list_entering_lanes

        lane_vid_mapping_dict = simulator_state['get_lane_vehicles']
        vid_distance_mapping_dict = simulator_state['get_vehicle_distance']

        ## TODO LANE LENGTH = 300
        bins = np.linspace(0, 300, 4).tolist()

        for lane in self.list_exiting_lanes:
            if lane is not None:
                coming_vehicle_position = []
                vehicle_position_lane = lane_vid_mapping_dict[lane]
                for vehicle in vehicle_position_lane:
                    coming_vehicle_position.append(vid_distance_mapping_dict[vehicle])
                leaving_distribution.extend(self.lane_position_mapper(coming_vehicle_position, bins))

        return leaving_distribution



    def _get_pressure(self):
        ##TODO eng.get_vehicle_distance(), another way to calculate pressure & queue length

        pressure = 0
        all_enter_car_queue = 0
        for lane in self.list_entering_lanes:
            if lane is not None:
                if self.fast_compute:
                    all_enter_car_queue += self.dic_lane_vehicle_count_current_step[lane]
                else:
                    all_enter_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]

        all_leaving_car_queue = 0
        for lane in self.list_exiting_lanes:
            if lane is not None:
                if self.fast_compute:
                    all_leaving_car_queue += self.dic_lane_vehicle_count_current_step[lane]
                else:
                    all_leaving_car_queue += self.dic_lane_waiting_vehicle_count_current_step[lane]

        p = all_enter_car_queue - all_leaving_car_queue
        
        if p < 0:
            p = -p
        #p = all_enter_car_queue

        return p


    def _get_sp_pressure(self):
        ##TODO eng.get_vehicle_distance(), another way to calculate pressure & queue length

        pressure = 0
        all_enter_car_queue = 0
        for lane in self.list_entering_lanes:
            if lane is not None:
                if self.fast_compute:              
                    all_enter_car_queue += self.dic_lane_sp_vehicle_count_current_step[lane]
                else:
                    all_enter_car_queue += self.dic_lane_waiting_sp_vehicle_count_current_step[lane]

        all_leaving_car_queue = 0
        for lane in self.list_exiting_lanes:
            if lane is not None:
                if self.fast_compute:
                    all_leaving_car_queue += self.dic_lane_sp_vehicle_count_current_step[lane]
                else:
                    all_leaving_car_queue += self.dic_lane_waiting_sp_vehicle_count_current_step[lane]

        #p = all_enter_car_queue - all_leaving_car_queue
        
        #if p < 0:
        #    p = -p
            
        p = all_enter_car_queue
        
        return p


    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] if lane is not None else -1 for lane in list_lanes]



    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [len(self.dic_lane_vehicle_current_step[lane]) if lane is not None else -1 for lane in list_lanes]


    def _get_lane_num_sp_vehicle(self, list_lanes):
        '''
        special vehicle number for each lane
        '''
        return [len(self.dic_lane_sp_vehicle_current_step[lane]) if lane is not None else -1 for lane in list_lanes]
        
        #result = []
        #for lane in list_lanes:
        #    if lane is not None:
        #        result.append(len(self.dic_lane_sp_vehicle_current_step[lane]))
        #    else:
        #        resilt.append(-1)
        #return result


    def _get_connectivity(self, dic_of_list_lanes):
        '''
        vehicle number for each lane
        '''
        result = []
        for i in range(len(dic_of_list_lanes['lane_ids'])):
            num_of_vehicles_on_road = sum([len(self.dic_lane_vehicle_current_step[lane]) for lane in dic_of_list_lanes['lane_ids'][i]])
            result.append(num_of_vehicles_on_road)

        lane_length = [0] + dic_of_list_lanes['lane_length']
        if np.sum(result)==0:
            result=[1]+result 
        else:
            result = [np.sum(result)]+ result
        connectivity = list(np.array(result * np.exp(-np.array(lane_length)/(self.length_lane*4))))
        # print(connectivity)
        # sys.exit()
        return connectivity



    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        raise NotImplementedError



    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''

        raise NotImplementedError


    # non temporary
    def _get_lane_num_vehicle_left(self, list_lanes):
        
        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left



    def _get_lane_sum_duration_vehicle_left(self, list_lanes):

        ## not implemented error
        raise NotImplementedError



    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] if lane is not None else -1 for lane in list_lanes]

    def _get_lane_num_sp_vehicle_been_stopped(self, thres, list_lanes):
        return [self.dic_lane_waiting_sp_vehicle_count_current_step[lane] if lane is not None else -1 for lane in list_lanes]


    def _get_position_grid_along_lane(self, vec):
        pos = int(self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_LANEPOSITION")])
        return min(pos//self.length_grid, self.num_grid)



    def _get_lane_vehicle_position(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_vehicle_current_step[lane]
            for vec in list_vec_id:
                pos = int(self.dic_vehicle_distance_current_step[vec])
                pos_grid = min(pos//self.length_grid, self.num_grid)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)
    
    
    # debug
    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_distance_current_step[veh_id]
            speed = self.dic_vehicle_speed_current_step[veh_id]
            return pos, speed
        except:
            return None, None



    def _get_lane_vehicle_speed(self, list_lanes):
        return [self.dic_vehicle_speed_current_step[lane] for lane in list_lanes]



    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):

        raise NotImplementedError


    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()



    def get_dic_vehicle_arrive_leave_time(self):

        return self.dic_vehicle_arrive_leave_time



    def get_feature(self):

        return self.dic_feature



    def get_state(self, list_state_features):
        # customize your own state
        # print(list_state_features)
        # print(self.dic_feature)
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in list_state_features}

        return dic_state



    def get_reward(self, dic_reward_info):
        # customize your own reward
        # 此处对于普通车辆就用pressure进行计算，对于特殊车辆，采用sp_pressure以及被阻塞停止数量作为reward反馈，分别测试
        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] =  np.sum(self.dic_feature["lane_queue_length"])    #None
        dic_reward["sum_lane_wait_time"] = None #np.sum(self.dic_feature['sum_lane_wait_time'])
        dic_reward["sum_lane_sp_vehicle_waiting_time"] = None #np.sum(self.dic_feature['sum_lane_sp_vehicle_waiting_time'])
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])
        dic_reward["lane_num_sp_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_sp_vehicle_been_stopped_thres1"])  #None  
        dic_reward["pressure"] =  np.sum(self.dic_feature["pressure"]) #None
        dic_reward["sp_pressure"] = np.sum(self.dic_feature["sp_pressure"]) #None

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


class Vehicle:
    def __init__(self, vehicle_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log, flow_file):
        self.vehicle_id = vehicle_id

        self.vehicle_name = "sp_vehicle_{0}".format(inter_id,)

        self.eng = eng

        #self.fast_compute = dic_traffic_env_conf['FAST_COMPUTE']

        #self.controlled_model  = dic_traffic_env_conf['MODEL_NAME']
        
        self.path_to_log = path_to_log
        
        #获取车辆自身的相关信息
        self.vehicle_info = self.eng.get_vehicle_info(vehicle_id)
        '''
        返回的是一个字典，包含以下信息：
        running: whether the vehicle is running.

        speed: The speed of the vehicle.
        distance: The distance the vehicle has travelled on the current lane or lanelink.
        drivable: The id of the current drivable(lane or lanelink)
        road: The id of the current road if the vehicle is running on a lane.
        intersection: The next intersection if the vehicle is running on a lane.
        route: A string contains ids of following roads in the vehicle’s route which are separated by ' '.
        '''
        #当前自身所在的edge
        self.loc_road = self.vehicle_info['road']
        #当前自身所在的lane
        self.loc_lane = self.vehicle_info['']
        #自身预设的规划路径
        self.travel_path
        

class AnonEnv:
    #list_intersection_id = [
    #    "intersection_1_1"
    #]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.simulator_type = self.dic_traffic_env_conf["SIMULATOR_TYPE"]

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.feature_name_for_neighbor = self._reduce_duplicates(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            #raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):

        print("# self.eng.reset() to be implemented")

        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],               #1
            "seed": 0,
            "laneChange": False,
            "dir": self.path_to_work_directory+"/",                          #"/colight/Mylight/records"
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],        #这个和dir组合，需要写相对路径 "roadnet_3_4.json"
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],           #应该也是相对路径？ "anon_3_4_jinan_real.json"
            "rlTrafficLight": True,#self.dic_traffic_env_conf["RLTRAFFICLIGHT"],   #True
            "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],           #False
            "roadnetLogFile": "frontend/web/roadnetLogFile.json",            #"roadnetLogFile.json"
            "replayLogFile": "frontend/web/replayLogFile.txt"                #"replayLogFile.txt"
        }
        print("=========================")
        #print(cityflow_config)

        with open(os.path.join(self.path_to_work_directory,"cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)
        self.eng = engine.Engine(os.path.join(self.path_to_work_directory,"cityflow.config"), thread_num=1)
        road_file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])

        roadnet = RoadNet('{0}'.format(road_file))
        
        #构建traffic_light_node_dict字典，判断根据是否需要lane的邻接信息，在邻接信息构造中，也自动调用创建了Roadnet类来存储静态的路网构建信息
        if self.dic_traffic_env_conf["USE_LANE_ADJACENCY"]:
            #构造lane与其他lane的连接关系
            self.traffic_light_node_dict = self._adjacency_extraction_lane()
        else:
            #仅仅构造road之间的连接关系
            self.traffic_light_node_dict = self._adjacency_extraction()
        
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
        
        
        #for i in self.traffic_light_node_dict:
        #    if i == 'intersection_1_4':
        #        for key in self.traffic_light_node_dict[i]:
        #            print("key:",key)
        #            print("value:",self.traffic_light_node_dict[i][key])
        
        
        
        #返回list,存储了所有在路的vehicle的id
        
        self.list_all_vehicle_previous_step = []                          #上一状态所有在路车辆
        self.list_all_vehicle_current_step = self.eng.get_vehicles()      #本状态所有在路车辆
        self.list_all_vehicle_entering = list(set(self.list_all_vehicle_current_step)-set(self.list_all_vehicle_previous_step))  #新驶入车辆
        self.list_all_vehicle_exiting = list(set(self.list_all_vehicle_previous_step)-set(self.list_all_vehicle_current_step))   #新驶出车辆
        
        #构造特殊车辆list,主要存储特殊车辆的id
        self.list_sp_vehicle = []
        for vehicle in self.list_all_vehicle_entering:
            if random.random()<self.dic_traffic_env_conf["SP_VEHICLE_PROPORTION"]:                                       #特殊车辆在路网中的占比情况
                self.list_sp_vehicle.append(vehicle)
                #添加特殊车辆后，为其即时规划一条最通畅地路径
                route = self.eng.get_vehicle_info(vehicle)["route"]
                print(route)
                start = route[1]
                end = route[-1]
                
                new_route = self.Dijkstra(start,end)
                
                set_vehicle_route(vehicle_id, new_route)
                
        for vehicle in self.list_all_vehicle_exiting:
            if vehicle in self.list_sp_vehicle:
                self.list_sp_vehicle.remove(vehicle)
        
        #构造所有车辆驶入驶出环境的相关时间路程信息
        self.dic_all_vehicle = dict()
        self.update_all_vehicle()
        
        
        #构建交通灯intersection类
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],self.path_to_log,roadnet,net)
                                  for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"]) if roadnet.hasInter("intersection_{0}_{1}".format(i+1, j+1))]

        self.list_inter_log = [[] for i in range(self.dic_traffic_env_conf["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_conf["NUM_COL"]) if roadnet.hasInter("intersection_{0}_{1}".format(i+1, j+1))]

        #构建交通灯和车道的全局索引，为每一个intersection,road,lane进行id到索引的映射
        self.id_to_index = {}
        count_inter = 0
        for i in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for j in range(self.dic_traffic_env_conf["NUM_COL"]):
                self.id_to_index['intersection_{0}_{1}'.format(i+1, j+1)] = count_inter
                count_inter += 1

        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)): # TODO
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                if self.list_intersection[i].list_entering_lanes[j] is not None:
                    lane_id = self.list_intersection[i].list_entering_lanes[j] 
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

        # build adjacency_matrix_lane in index from _adjacency_matrix_lane
        #构造每个路口驶入车道的邻接车道矩阵，规模为（驶入车道数：lane_number）*(驶入和驶出方向：2)*(最邻近：top_k),
        #system_state_start_time = time.time()
        for inter in self.list_intersection:
            inter.build_adjacency_row_lane(self.lane_id_to_index)
        #print("Get adjacency matrix time: ", time.time()-system_state_start_time)


        #构建获取此刻系统状态信息信息
        system_state_start_time = time.time()
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
                                
            if not self.dic_traffic_env_conf["SP_VEHICLE"]:                    
                self.system_states = {
                              
                              "get_lane_vehicles": self.eng.get_lane_vehicles(),     #返回字典,lane id作为key,vehicle id 作为value
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),   #返回字典,lane id作为key,vehicle 数量作为value
                              "get_vehicle_speed": None,       #返回字典,vehicle id作为key,speed 作为value
                              "get_vehicle_distance": None     #返回字典,vehicle id作为key,对应在当前lane上已经行驶的distance作为值) 
                              }
            
            else:
                dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
                dic_sp_vehicle_speed_current_step = {}
                for vehicle in self.list_sp_vehicle:
                    dic_sp_vehicle_speed_current_step[vehicle] = dic_vehicle_speed_current_step[vehicle]
        
                self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_sp_vehicle_speed": dic_sp_vehicle_speed_current_step,
                              "get_vehicle_distance": None           #self.eng.get_vehicle_distance()
                              }
            
        else:
        
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }
                              
        #print("Get system state time: ", time.time()-system_state_start_time)
        #初始化当前环境的图矩阵表示，相连矩阵的距离为1
        self.graph_matrix = np.zeros(self.dic_traffic_env_conf["NUM_ROW"]**2*self.dic_traffic_env_conf["NUM_COL"]**2).reshape \
                                               (self.dic_traffic_env_conf["NUM_ROW"]*self.dic_traffic_env_conf["NUM_COL"],self.dic_traffic_env_conf["NUM_ROW"]*self.dic_traffic_env_conf["NUM_COL"])




        if self.dic_traffic_env_conf['DIRECTED_GRAPH']:
            self.init_graph_matrix()
            self.topk_matrix = self.get_topk_from_graph(self.graph_matrix)
        else:
            self.topk_matrix = []
           
       
        #根据system_state更新此刻的measure_map
        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.list_sp_vehicle, self.system_states, self.topk_matrix)
        print("Update_current_measurements_map time: ", time.time()-update_start_time)

        #update neighbor's info,
        neighbor_start_time = time.time()
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            #遍历每一个inter为其增加邻居信息
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters,deepcopy(inter.dic_feature))
                
                
        print("Update_neighbor time: ", time.time()-neighbor_start_time)
        
        #state为list，长度为inter_num，包含了 每个路口强化学习所需要的特征, 以及各个路口之间的邻接矩阵adjacency_matrix_lane
        #这些是强化学习的输入特征
        state, done = self.get_state()
        # print(state)
        return state

    def step(self, action):
        step_start_time = time.time()
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        #在MIN_ACTION_TIME这个时间段内，交通灯全部置为同一状态action
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            #switch将动作全部添加为0，表明接下来的时间内不改变状态
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            #复制每秒的相位状态
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        #list存储每一个动作的reward
        average_reward_action_list = [0]*len(action)
        #在这里对于每一秒的状态都要计算一个reward
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i] 

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            #调用inter类里的获得每一个路口的feature
            before_action_feature = self.get_feature()
            # state = self.get_state()

            if self.dic_traffic_env_conf['DEBUG']:
                print("time: {0}".format(instant_time))
            else:
                if i == 0:
                    print("time: {0}".format(instant_time))

            #内部step
            self._inner_step(action_in_sec)

            # get reward
            if self.dic_traffic_env_conf['DEBUG']:
                start_time = time.time()

            reward = self.get_reward()

            if self.dic_traffic_env_conf['DEBUG']:
                print("Reward time: {}".format(time.time()-start_time))

            #求在一个动作interval中的均值
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)

            # average_reward_action = (average_reward_action*i + reward[0])/(i+1)

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):

        # copy current measurements to previous measurements
########针对每一个路口交通灯执行，将五个关键状态的当前值赋值给前一时刻值,
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        
        # set signals
        # multi_intersection decided by action {inter_id: phase}
########针对每一个路口交通灯，设置信号更新
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step
########cityflow调用状态转移一步
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()
########重新获取system_state
        system_state_start_time = time.time()
        
########以及更新env中的list_all_vehicle
        self.list_all_vehicle_previous_step = self.list_all_vehicle_current_step       #辅助给之前的在路车辆
        self.list_all_vehicle_current_step = self.eng.get_vehicles()                   #本状态所有在路车辆
        self.list_all_vehicle_entering = list(set(self.list_all_vehicle_current_step)-set(self.list_all_vehicle_previous_step))        #新驶入车辆
        self.list_all_vehicle_exiting = list(set(self.list_all_vehicle_previous_step)-set(self.list_all_vehicle_current_step))   #新驶出车辆        
        
        #增加新驶入特殊车辆以及删除已经到达终点的特殊车辆
        for vehicle in self.list_all_vehicle_entering:
            #随机抽取一定比例路网中普通车辆作为特殊车辆
            if random.random() < self.dic_traffic_env_conf["SP_VEHICLE_PROPORTION"]:           #特殊车辆在路网中的占比情况
                self.list_sp_vehicle.append(vehicle)
                #添加特殊车辆后，为其即时规划一条最通畅地路径
                route = self.eng.get_vehicle_info(vehicle)["route"]
                print(route)
                start = route[1]
                end = route[-1]
                
                new_route = self.Dijkstra(start,end)
                
                set_vehicle_route(vehicle_id, new_route)
        for vehicle in self.list_all_vehicle_exiting:
            if vehicle in self.list_sp_vehicle:
                self.list_sp_vehicle.remove(vehicle)
        
        #更新所有车辆的驶入路网时间，以及驶出路网时间
        self.update_all_vehicle()
        
        #此处针对需要使用MyLight将FAST_COMPUTE设置为False,更新特殊车辆速度
        if self.dic_traffic_env_conf["FAST_COMPUTE"]:
                                
            if not self.dic_traffic_env_conf["SP_VEHICLE"]:                    
                self.system_states = {
                              
                              "get_lane_vehicles": self.eng.get_lane_vehicles(),     #返回字典,lane id作为key,vehicle id 作为value
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),   #返回字典,lane id作为key,vehicle 数量作为value
                              "get_vehicle_speed": None,       #返回字典,vehicle id作为key,speed 作为value
                              "get_vehicle_distance": None     #返回字典,vehicle id作为key,对应在当前lane上已经行驶的distance作为值) 
                              }
            
            else:
                dic_vehicle_speed_current_step = self.eng.get_vehicle_speed()
                dic_sp_vehicle_speed_current_step = {}
                for vehicle in self.list_sp_vehicle:
                    dic_sp_vehicle_speed_current_step[vehicle] = dic_vehicle_speed_current_step[vehicle]
        
                self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_sp_vehicle_speed": dic_sp_vehicle_speed_current_step,
                              "get_vehicle_distance": None           #self.eng.get_vehicle_distance()
                              }
            
        else:
        
            self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        
        if self.dic_traffic_env_conf['DEBUG']:
            start_time_2 = time.time()
        #更新由于特殊车辆而导致的动态有向图的变化,以及top-k邻居路口的变化
        if self.dic_traffic_env_conf["DIRECTED_GRAPH"]:
            self.update_graph_matrix()
            self.topk_matrix = self.get_topk_from_graph(self.graph_matrix)
                
        # print("Get system state time: ", time.time()-system_state_start_time)

        if self.dic_traffic_env_conf['DEBUG']:
            print("Get system state all time: {}".format(time.time()-start_time))
            print("Get topk_matrix time: {}".format(time.time()-start_time_2))
        # get new measurements

        if self.dic_traffic_env_conf['DEBUG']:
            start_time = time.time()


########根据system_state来更新每一个inter的measure_map
        update_start_time = time.time()
        for inter in self.list_intersection:
            inter.update_current_measurements_map(self.list_sp_vehicle, self.system_states, self.topk_matrix)

        #update neighbor's info
########为inter.dic_feature增加邻居节点的信息，包括了phase,phase_time,lane_vehicle
        if self.dic_traffic_env_conf["NEIGHBOR"]:
            for inter in self.list_intersection:
                neighbor_inter_ids = inter.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:    #这里append的是几个邻居类
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                inter.dic_feature = inter.update_neighbor_info(neighbor_inters, deepcopy(inter.dic_feature))



        if self.dic_traffic_env_conf['DEBUG']:
            print("Update measurements time: {}".format(time.time()-start_time))

        #self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        #self.log_phase()

    def update_all_vehicle(self):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in self.list_all_vehicle_entering:
            vehicle_state = self.eng.get_vehicle_info(vehicle)
            
            if vehicle not in self.dic_all_vehicle:
                if vehicle in self.list_sp_vehicle:               
                    self.dic_all_vehicle[vehicle] = \
                        {"enter_time": ts, "leave_time": np.nan, "sp_vehicle": 1, \
                         "enter_intersection": vehicle_state["intersection"], "leave_intersection": np.nan, \
                         "travel_trajectory": [vehicle_state["intersection"]], "first_road_dis":0, "travel_distance": 0 }
                else:
                    self.dic_all_vehicle[vehicle] = \
                        {"enter_time": ts, "leave_time": np.nan, "sp_vehicle": 0, \
                         "enter_intersection": vehicle_state["intersection"], "leave_intersection": np.nan, \
                         "travel_trajectory": [vehicle_state["intersection"]], "first_road_dis":0, "travel_distance": 0 }                    
            #else:
                #print("vehicle: %s already exists in entering lane!"%vehicle)
                #sys.exit(-1)
                #pass
        
        #对于所有在路网的车辆，实时地更新他们的当前位置，行驶的时间，路程
        for vehicle in self.eng.get_vehicles():
            vehicle_state = self.eng.get_vehicle_info(vehicle)
     
            if "intersection" in vehicle_state.keys():
                #print("vehicle_state:",vehicle_state)
                
                self.dic_all_vehicle[vehicle]["leave_time"] = ts
                #print("leave_time:",ts)
                
                self.dic_all_vehicle[vehicle]["leave_intersection"] = vehicle_state["intersection"]
                #print("leave_intersection:",vehicle_state["intersection"])  
                
                if self.dic_all_vehicle[vehicle]["leave_intersection"] not in self.dic_all_vehicle[vehicle]["travel_trajectory"]:
                    self.dic_all_vehicle[vehicle]["travel_trajectory"].append(self.dic_all_vehicle[vehicle]["leave_intersection"])
                
                start_loc = self.traffic_light_node_dict[self.dic_all_vehicle[vehicle]["enter_intersection"]]["location"]
                #print("start_loc:",start_loc)
                
                leave_loc = None
                if len(self.dic_all_vehicle[vehicle]["travel_trajectory"]) == 1:
                    self.dic_all_vehicle[vehicle]["first_road_dis"] = float(vehicle_state["distance"])
                    self.dic_all_vehicle[vehicle]["travel_distance"] = float(vehicle_state["distance"])
                    
                
                elif len(self.dic_all_vehicle[vehicle]["travel_trajectory"]) == 2:
                    self.dic_all_vehicle[vehicle]["travel_distance"] = self.dic_all_vehicle[vehicle]["first_road_dis"] + float(vehicle_state["distance"])
                    
                elif len(self.dic_all_vehicle[vehicle]["travel_trajectory"]) > 2:
                    leave_loc = self.traffic_light_node_dict[self.dic_all_vehicle[vehicle]["travel_trajectory"][-2]]["location"]
                    self.dic_all_vehicle[vehicle]["travel_distance"] = self._cal_mht_distance(start_loc, leave_loc) + float(vehicle_state["distance"])
                    
                #print("get the dis")
                
        #except KeyError:
            #print("vehicle not recorded when entering in updating all vehicle")
            #sys.exit(-1)

    def Dijkstra(self, start, end ):
        _ = float('inf')
        points = self.dic_traffic_env_conf["NUM_INTERSECTIONS"]
        if points == 11:
            edges = 15
        else:
            row = self.dic_traffic_env_conf["NUM_ROW"]
            col = self.dic_traffic_env_conf["NUM_COL"]
            edges = (row-1)*col + (clo-1)*row
        
        map = np.ones((self.dic_traffic_env_conf["NUM_INTERSECTIONS"],self.dic_traffic_env_conf["NUM_INTERSECTIONS"]))* _
        
        lane_count =  self.eng.get_lane_vehicle_count()
        print(lane_count)
        
        
        
        map = map.tolist()
        pre = [0]*(points+1) #记录前驱
        vis = [0]*(points+1) #记录节点遍历状态
        dis = [_ for i in range(points + 1)] #保存最短距离
        road = [0]*(points+1) #保存最短路径
        roads = []
        map = graph

        for i in range(points+1):#初始化起点到其他点的距离
            if i == start :
                dis[i] = 0
            else :
                dis[i] = map[start][i]
            if map[start][i] != _ :
                pre[i] = start
            else :
                pre[i] = -1
        vis[start] = 1
        for i in range(points+1):#每循环一次确定一条最短路
            min = _
            for j in range(points+1):#寻找当前最短路
                if vis[j] == 0 and dis[j] < min :
                    t = j
                    min = dis[j]
            vis[t] = 1 #找到最短的一条路径 ,标记
            for j in range(points+1):
                if vis[j] == 0 and dis[j] > dis[t]+ map[t][j]:
                    dis[j] = dis[t] + map[t][j]
                    pre[j] = t
        p = end
        len = 0
        while p >= 1 and len < points:
            road[len] = p
            p = pre[p]
            len += 1
        mark = 0
        len -= 1
        while len >= 0:
            roads.append(road[len])
            len -= 1
            
        #返回距离，以及路径上的点的索引    
        return dis[end],roads

        
        


    #从用矩阵表示的各个路口间的有向连接图中提取每一个路口的top-k路网
    def get_topk_from_graph(self, graph_matrix):
    
    
        graph = deepcopy(graph_matrix)
        m = self.dic_traffic_env_conf['NUM_ROW']   #4
        n = self.dic_traffic_env_conf['NUM_COL']   #3
        k = self.dic_traffic_env_conf['TOP_K_ADJACENCY']
        crossroad_number = m*n
        result=np.zeros(m*n*k).reshape(m*n,k)
        
        def refresh(index,value,current_distance):
            if(current_distance[index]>value): #新值更小
                current_distance[index]=value        
                
        def sort (current_distance,index):
            sorted_index=np.argsort(current_distance)
            for i in range(k):
                result[index][i]=sorted_index[i]        
        
        def find_shortest(crossroads,current_distance):
            shortest_distance=crossroad_number
            shortest_index=0
            # for row in range(crossroad_number):
                # for column in range(crossroad_number):
                    # if(graph[row][column]<shortest_distance and crossroads[column]==1 and crossroads[row]==0 and graph[row][column]!=0): #如果距离短且该目的地已被访问且该出发地没有被访问
                        # shortest_distance=graph[row][column]
                        # shortest_index=row  #最小距离的点是出发点
            for crossroad in range(crossroad_number):
                if crossroads[crossroad] == 1: # 遍历已经访问过的路口
                    if crossroad % n > 0 and crossroads[crossroad - 1] == 0: # 
                        if(0 < graph[crossroad - 1][crossroad] < shortest_distance):
                            shortest_distance = graph[crossroad-1][crossroad]
                            shortest_index = crossroad - 1

                    if crossroad % n < n - 1 and crossroads[crossroad + 1] == 0:  
                        if (0 < graph[crossroad + 1][crossroad] < shortest_distance):
                            shortest_distance = graph[crossroad + 1][crossroad]
                            shortest_index = crossroad + 1

                    if crossroad // n > 0 and crossroads[crossroad - n] == 0: 
                        if(0 < graph[crossroad - n][crossroad] < shortest_distance):
                            shortest_distance = graph[crossroad-n][crossroad]
                            shortest_index = crossroad - n

                    if crossroad // n < m - 1 and crossroads[crossroad + n] == 0:  
                        if (0 < graph[crossroad + n][crossroad] < shortest_distance):
                            shortest_distance = graph[crossroad + n][crossroad]
                            shortest_index = crossroad + n

            crossroads[shortest_index]=1 #标记已访问
            if(shortest_index%n>0):
                refresh(shortest_index-1,current_distance[shortest_index]+graph[shortest_index-1][shortest_index],current_distance)
            if(shortest_index%n<n-1):
                refresh(shortest_index+1,current_distance[shortest_index]+graph[shortest_index+1][shortest_index],current_distance)
            if(shortest_index//n>0):
                refresh(shortest_index-n,current_distance[shortest_index]+graph[shortest_index-n][shortest_index],current_distance)
            if(shortest_index//n<m-1):
                refresh(shortest_index+n,current_distance[shortest_index]+graph[shortest_index+n][shortest_index],current_distance)        

        def calculate_one(crossroad_index):
            current_distance=np.ones(crossroad_number)*crossroad_number
            current_distance[crossroad_index]=0
            crossroads=np.zeros(crossroad_number) #原始状态所有路口没有访问
            crossroads[crossroad_index]=1
            counter=0
            if (crossroad_index%n>0):
                refresh(crossroad_index-1,current_distance[crossroad_index]+graph[crossroad_index-1][crossroad_index],current_distance)
            if (crossroad_index%n<n-1):
                refresh(crossroad_index+1,current_distance[crossroad_index]+graph[crossroad_index+1][crossroad_index],current_distance)
            if (crossroad_index//n>0):
                refresh(crossroad_index-n,current_distance[crossroad_index]+graph[crossroad_index-n][crossroad_index],current_distance)
            if (crossroad_index//n<m-1):
                refresh(crossroad_index+n,current_distance[crossroad_index]+graph[crossroad_index+n][crossroad_index],current_distance)

            while(counter<min(crossroad_number-1,20)):
                find_shortest(crossroads,current_distance)
                counter=counter+1
            #print("路口",crossroad_index,"各路口到它的最短距离",current_distance)
            sort(current_distance,crossroad_index)        
        
        
        for index in range(crossroad_number):
            calculate_one(index)
        return result


    def update_graph_matrix(self):
        for inter in self.list_intersection:
            row_index = inter.index//self.dic_traffic_env_conf["NUM_COL"]
            col_index = inter.index%self.dic_traffic_env_conf["NUM_COL"]
            for index,lane in enumerate(inter.list_entering_lanes):
                
                #print(inter.dic_lane_sp_vehicle_current_step)
                #有经过该路口即将右行的特殊车辆且该路口不是最右侧路口
                if(index==1 or index==6 or index==11):
                    if(len(inter.dic_lane_sp_vehicle_current_step[lane])>0 and row_index<self.dic_traffic_env_conf["NUM_ROW"]-1):
                        #print('inter.index:',inter.index)
                        #print('right_index:',inter.index+self.dic_traffic_env_conf["NUM_COL"])
                        #使用地理距离信息
                        #if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']:
                        #使用关系距离，1或者0.5：
                        #else:   
                        self.graph_matrix[inter.index][inter.index+self.dic_traffic_env_conf["NUM_COL"]]=0.5
                #有经过该路口即将上行的特殊车辆且该路口不是最上侧路口
                if(index==0 or index==5 or index==10):
                    if(len(inter.dic_lane_sp_vehicle_current_step[lane])>0 and col_index<self.dic_traffic_env_conf["NUM_COL"]-1):
                        self.graph_matrix[inter.index][inter.index+1]=0.5
                #有经过该路口即将左行的特殊车辆且该路口不是最左侧路口
                if(index==4 or index==8 or index==9 ):
                    if(len(inter.dic_lane_sp_vehicle_current_step[lane])>0 and row_index>0):
                        self.graph_matrix[inter.index][inter.index-self.dic_traffic_env_conf["NUM_COL"]]=0.5
                #有经过该路口即将下行的特殊车辆且该路口不是最下侧路口
                if(index==2 or index==3 or index==7):
                    if(len(inter.dic_lane_sp_vehicle_current_step[lane])>0 and col_index>0):
                        self.graph_matrix[inter.index][inter.index-1]=0.5        


    def init_graph_matrix(self):
    
        for row in range(self.dic_traffic_env_conf["NUM_ROW"]):
            for column in range(self.dic_traffic_env_conf["NUM_COL"]):
                crossroad_index=row*self.dic_traffic_env_conf["NUM_COL"]+column
                self.graph_matrix[crossroad_index][crossroad_index]=0
                #初始化每个路口之间的距离都是1
                random_numbers=np.random.randint(1,2,4)
                if(row>0): #不是最左边，即左边有路口
                    self.graph_matrix[crossroad_index][crossroad_index-self.dic_traffic_env_conf["NUM_COL"]]=random_numbers[0]
                    self.graph_matrix[crossroad_index-self.dic_traffic_env_conf["NUM_COL"]][crossroad_index]=random_numbers[0]
                if(row<self.dic_traffic_env_conf["NUM_ROW"]-1): #不是最右边，即右边有路口
                    self.graph_matrix[crossroad_index][crossroad_index+self.dic_traffic_env_conf["NUM_COL"]]=random_numbers[1]
                    self.graph_matrix[crossroad_index+self.dic_traffic_env_conf["NUM_COL"]][crossroad_index]=random_numbers[1]
                if(column>0): #不是最下边，即下边有路口
                    self.graph_matrix[crossroad_index][crossroad_index-1]=random_numbers[2]
                    self.graph_matrix[crossroad_index-1][crossroad_index]=random_numbers[2]
                if(column<self.dic_traffic_env_conf["NUM_COL"]-1):  # 不是最上边，即上边有路口
                    self.graph_matrix[crossroad_index][crossroad_index+1] = random_numbers[3]
                    self.graph_matrix[crossroad_index+1][crossroad_index] = random_numbers[3]


    def load_roadnet(self, roadnetFile=None):
        print("Start load roadnet")
        start_time = time.time()
        if not roadnetFile:
            roadnetFile = "roadnet_1_1.json"
        #print("/n/n", os.path.join(self.path_to_work_directory, roadnetFile))
        self.eng.load_roadnet(os.path.join(self.path_to_work_directory, roadnetFile))
        print("successfully load roadnet:{0}, time: {1}".format(roadnetFile,time.time()-start_time))

    def load_flow(self, flowFile=None):
        print("Start load flowFile")
        start_time = time.time()
        if not flowFile:
            flowFile = "flow_1_1.json"
        self.eng.load_flow(os.path.join(self.path_to_work_directory, flowFile))
        print("successfully load flowFile: {0}, time: {1}".format(flowFile, time.time()-start_time))

    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):
        # consider neighbor info
        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = self._check_episode_done(list_state)

        # print(list_state)

        return list_state, done

    @staticmethod
    def _reduce_duplicates(feature_name_list):
        new_list = set()
        for feature_name in feature_name_list:
            if feature_name[-1] in ["0","1","2","3"]:
                new_list.add(feature_name[:-2])
        return list(new_list)

    def get_reward(self):

        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]

        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({ "time": cur_time,
                                                    "state": before_action_feature[inter_ind],
                                                    "action": action[inter_ind]})

    def batch_log(self, start, stop):
        #存储各个路口的信息到csv以及pkl文件
        for inter_ind in range(start, stop):
            if int(inter_ind)%100 == 0:
                print("Batch log for inter ",inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            #获取该路口每个车辆的到达时间，离开时间，以及sp类型，存储进csv
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle,orient='index')
            df.to_csv(path_to_log_file, na_rep="nan")
            #获得pkl文件
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()
        
        path_to_log_file = os.path.join(self.path_to_log, "all_vehicle.csv")
        dic_all_vehicle = self.dic_all_vehicle
        df = pd.DataFrame.from_dict(dic_all_vehicle,orient='index')
        df.to_csv(path_to_log_file, na_rep="nan")
        

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        #选取的步长batch_size_run是batch_size和len(inter)中的较小值
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start,stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()

        print("end join")

    def bulk_log(self):

        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            df.to_csv(path_to_log_file, na_rep="nan")

        for inter_ind in range(len(self.list_inter_log)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

        self.eng.print_log(os.path.join(self.path_to_log, self.dic_traffic_env_conf["ROADNET_FILE"]),
                           os.path.join(self.path_to_log, "replay_1_1.txt"))

        #print("log files:", os.path.join("data", "frontend", "roadnet_1_1_test.json"))

    def log_attention(self, attention_dict):
        path_to_log_file = os.path.join(self.path_to_log, "attention.pkl")
        f = open(path_to_log_file, "wb")
        pickle.dump(attention_dict, f)
        f.close()

    def log_hidden_state(self, hidden_states):
        path_to_log_file = os.path.join(self.path_to_log, "hidden_states.pkl")

        with open(path_to_log_file, "wb") as f:
            pickle.dump(hidden_states, f)

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str
        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt"%dic_lane_map[lane]), "a"))

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str
        dic_lane_map = {
            "road_0_1_0_0": "w",
            "road_2_1_2_0": "e",
            "road_1_0_1_0": "s",
            "road_1_2_3_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt"%dic_lane_map[lane]), "a"))

    def log_first_vehicle(self):
        _veh_id = "flow_0_"
        _veh_id_2 = "flow_2_"
        _veh_id_3 = "flow_4_"
        _veh_id_4 = "flow_6_"

        for inter in self.list_intersection:
            for i in range(100):
                veh_id = _veh_id + str(i)
                veh_id_2 = _veh_id_2 + str(i)
                pos, speed = inter._get_vehicle_info(veh_id)
                pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_a")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_a"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_b")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_b"))

                if pos and speed:
                    print("%f, %f, %f" % (self.get_current_time(), pos, speed),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_a", "first_vehicle_info_a_%d.txt" % i), "a"))
                if pos_2 and speed_2:
                    print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_b", "first_vehicle_info_b_%d.txt" % i), "a"))

                veh_id_3 = _veh_id_3 + str(i)
                veh_id_4 = _veh_id_4 + str(i)
                pos_3, speed_3 = inter._get_vehicle_info(veh_id_3)
                pos_4, speed_4 = inter._get_vehicle_info(veh_id_4)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_c")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_c"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_d")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_d"))

                if pos_3 and speed_3:
                    print("%f, %f, %f" % (self.get_current_time(), pos_3, speed_3),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_c", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_4 and speed_4:
                    print("%f, %f, %f" % (self.get_current_time(), pos_4, speed_4),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_d", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

    def log_phase(self):
        for inter in self.list_intersection:
            print("%f, %f" % (self.get_current_time(), inter.current_phase_index),
                  file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))

    #邻接信息提取
    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            #生成存储交通灯节点信息的字典
            for inter in net['intersections']:
                if not inter['virtual']:
                    #包含交通灯坐标位置，总路口数，
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                       'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,}

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            
            #每一个红绿灯的命名编号对应的数字编号
            inter_id_to_index = {}
            
            #从json文件提取路网信息
            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()


            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            #对每一个交通灯节点i，设置其id，邻居以及驶入的lanes的相关数据
            for i in traffic_light_node_dict.keys():
                #每一个路口都存储有inter_id_to_index的值
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                #对4个不同的方向j的道路road_id
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    #获取道路终点
                    
                    if not roadnet.hasEdge(road_id):
                        traffic_light_node_dict[i]['neighbor_ENWS'].append("0")
                    else:
                        neighboring_node = edge_id_dict[road_id]['to']
                    
                    # calculate the neighboring intersections，添加道路终点到邻居节点
                        if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                            traffic_light_node_dict[i]['neighbor_ENWS'].append("-1")
                        else:
                            traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                        
                    # calculate the entering lanes ENWS
                    for key, value in edge_id_dict.items():
                        #从道路中筛查，是从邻居过来且终点是路口i
                        if value['from'] == neighboring_node and value['to'] == i:
                            neighboring_road = key

                            neighboring_lanes = []
                            #添加车道数
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road+"_{0}".format(k))
                            #添加该车道的land_id以及lane的长度
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])

            #返回交通灯i的top k个最近的路口
            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO        #先添加自身？
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num

        return traffic_light_node_dict


    def _adjacency_extraction_lane(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])

        roadnet = RoadNet('{0}'.format(file))
        with open('{0}'.format(file)) as json_data:
            net = json.load(json_data)
            # print(net)
            for inter in net['intersections']:
                if not inter['virtual']:
                    traffic_light_node_dict[inter['id']] = {'location': {'x': float(inter['point']['x']),
                                                                       'y': float(inter['point']['y'])},
                                                            "total_inter_num": None, 'adjacency_row': None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None,
                                                            "entering_lane_ENWS": None,
                                                            "total_lane_num": None, 'adjacency_matrix_lane': None,
                                                            "lane_id_to_index": None,
                                                            "lane_ids_in_intersction": []
                                                            }

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            top_k_lane = self.dic_traffic_env_conf["TOP_K_ADJACENCY_LANE"]
            total_inter_num = len(traffic_light_node_dict.keys())

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()


            # set inter id to index dict
            inter_id_to_index = {}
            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            # set the neighbor_ENWS nodes and entring_lane_ENWS for intersections
            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['inter_id_to_index'] = inter_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    #新增判断
                    
                    if not roadnet.hasEdge(road_id):
                        traffic_light_node_dict[i]['neighbor_ENWS'].append("0")
                    else:
                        neighboring_node = edge_id_dict[road_id]['to']
                        # calculate the neighboring intersections
                        if neighboring_node not in traffic_light_node_dict.keys(): # virtual node
                            traffic_light_node_dict[i]['neighbor_ENWS'].append("-1")
                        else:
                            traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_node)
                        # calculate the entering lanes ENWS
                        for key, value in edge_id_dict.items():
                            if value['from'] == neighboring_node and value['to'] == i:
                                neighboring_road = key

                                neighboring_lanes = []
                                for k in range(value['num_of_lane']):
                                    neighboring_lanes.append(neighboring_road+"_{0}".format(k))

                                traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                                traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])

            #打印traffic_light_node_dict的内容
            
            #for items in traffic_light_node_dict.items():
            #    print(items[0])
            #    for items in items[1].items():
            #        print(items)
            #    print('\n')
            '''
            具体内容如下：
            intersection_2_3
            ('location', {'x': 800.0, 'y': 800.0})
            ('total_inter_num', 12)
            ('adjacency_row', 5,8,-1,2,4)
            ('inter_id_to_index', {'intersection_1_1': 0, 'intersection_1_2': 1, 'intersection_1_3': 2, 'intersection_2_1': 3, 'intersection_2_2': 4, 'intersection_2_3': 5, 'intersection_3_1': 6, 'intersection_3_2': 7, 'intersection_3_3': 8, 'intersection_4_1': 9, 'intersection_4_2': 10, 'intersection_4_3': 11})
            ('neighbor_ENWS', ['intersection_4_2', 'intersection_3_3', 'intersection_2_2', 'intersection_3_1'])
            ('entering_lane_ENWS', {'lane_ids': [['road_4_2_2_0', 'road_4_2_2_1', 'road_4_2_2_2'], ['road_3_3_3_0', 'road_3_3_3_1', 'road_3_3_3_2'], ['road_2_2_0_0', 'road_2_2_0_1', 'road_2_2_0_2'], ['road_3_1_1_0', 'road_3_1_1_1', 'road_3_1_1_2']], 'lane_length': [2573.5913600840718, 2920.225231898308, 2025.798040898392, 1931.370849898476]})
            ('total_lane_num', None)
            ('adjacency_matrix_lane', None)
            ('lane_id_to_index', None)
            ('lane_ids_in_intersction', [])
            '''

           #假如加入了车道信息，还需要建立车道的字典来进行相关设置
            lane_id_dict = roadnet.net_lane_dict
            total_lane_num = len(lane_id_dict.keys())
            # output an adjacentcy matrix for all the intersections
            # each row stands for a lane id,
            # each column is a list with two elements: first is the lane's entering_lane_LSR, second is the lane's leaving_lane_LSR
            #这是已经排序过的list吗，直接取前k个
            def _get_top_k_lane(lane_id_list, top_k_input):
                top_k_lane_indexes = []
                for i in range(top_k_input):
                    lane_id = lane_id_list[i] if i < len(lane_id_list) else None
                    top_k_lane_indexes.append(lane_id)
                return top_k_lane_indexes

            adjacency_matrix_lane = {}
            for i in lane_id_dict.keys(): # Todo lane_ids should be in an order
                adjacency_matrix_lane[i] = [_get_top_k_lane(lane_id_dict[i]['input_lanes'], top_k_lane),
                                            _get_top_k_lane(lane_id_dict[i]['output_lanes'], top_k_lane)]


            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                # TODO return with Top K results
                if not self.dic_traffic_env_conf['ADJACENCY_BY_CONNECTION_OR_GEO']: # use geo-distance
                    row = np.array([0]*total_inter_num)
                    # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                    #计算该路口到其他路口的地理距离
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = AnonEnv._cal_distance(location_1,location_2)
                        row[inter_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                    elif len(row) > top_k:
                        #距离最近的top_k个元素
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_inter_num)]
                    #去除自身的索引
                    adjacency_row_unsorted.remove(inter_id_to_index[i])
                    #再添加自身索引在最前边？？
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]] + adjacency_row_unsorted
                else: # use connection infomation，添加东北西南四个方向的路口的index
                    traffic_light_node_dict[i]['adjacency_row'] = [inter_id_to_index[i]]    
                    for j in traffic_light_node_dict[i]['neighbor_ENWS']: ## TODO
                        if j is not None:
                            traffic_light_node_dict[i]['adjacency_row'].append(inter_id_to_index[j])
                        else:
                            traffic_light_node_dict[i]['adjacency_row'].append(-1)


                traffic_light_node_dict[i]['total_inter_num'] = total_inter_num
                traffic_light_node_dict[i]['total_lane_num'] = total_lane_num
                traffic_light_node_dict[i]['adjacency_matrix_lane'] = adjacency_matrix_lane



        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a-b)**2))
        
    def _cal_mht_distance(self, loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return abs(a[0]-b[0])+abs(a[1]-b[1])
        
    def end_sumo(self):
        print("anon process end")
        pass

if __name__ == '__main__':

    dic_agent_conf = {
    "PRIORITY": True,
    "nan_code":True,
    "att_regularization":False,
    "rularization_rate":0.03,
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    #special care for pretrain
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    #"TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
        }

    dic_exp_conf = {
        "RUN_COUNTS": 3600,
        "MODEL_NAME": "STGAT",


        "ROADNET_FILE": "roadnet_{0}.json".format("3_4"),

        "NUM_ROUNDS": 200,
        "NUM_GENERATORS": 4,

        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 3,

        "PRETRAIN_NUM_ROUNDS": 0,
        "PRETRAIN_NUM_GENERATORS": 15,

        "AGGREGATE": False,
        "PRETRAIN": False,
        "DEBUG": False,
        "EARLY_STOP": True
    }

    dic_traffic_env_conf  = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_3_4_jinan_real.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 9,
        "ACTION_PATTERN": "set",
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 10,
        "YELLOW_TIME": 5,
        "DEBUG": False,
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": False,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY":9,
        "TOP_K_ADJACENCY_LANE": 6,
        
        
        "SAVEREPLAY": False,
        "NUM_ROW": 4,
        "NUM_COL": 3,

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("3_4"),

        "LIST_STATE_FEATURE": [
            "cur_phase",
            # "time_this_phase",
            # "vehicle_position_img",
            # "vehicle_speed_img",
            # "vehicle_acceleration_img",
            # "vehicle_waiting_time_img",
            "lane_num_vehicle",
            # "lane_num_vehicle_been_stopped_thres01",
            # "lane_num_vehicle_been_stopped_thres1",
            # "lane_queue_length",
            # "lane_num_vehicle_left",
            # "lane_sum_duration_vehicle_left",
            # "lane_sum_waiting_time",
            # "terminal",
            # "coming_vehicle",
            # "leaving_vehicle",
            # "pressure"
            # "adjacency_matrix",
            # "lane_queue_length",
            "adjacency_matrix_lane",
            ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),
            D_COMING_VEHICLE = (12,),
            D_LEAVING_VEHICLE = (12,),
            D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
            D_CUR_PHASE=(1,),
            D_NEXT_PHASE=(1,),
            D_TIME_THIS_PHASE=(1,),
            D_TERMINAL=(1,),
            D_LANE_SUM_WAITING_TIME=(4,),
            D_VEHICLE_POSITION_IMG=(4, 60,),            
            D_VEHICLE_SPEED_IMG=(4, 60,),
            D_VEHICLE_WAITING_TIME_IMG=(4, 60,),
            D_PRESSURE=(1,),
            D_ADJACENCY_MATRIX=(2,),
            D_ADJACENCY_MATRIX_LANE=(6,)
             ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
            "sum_duration_vehicle_left": 0,
            "sum_num_vehicle_been_stopped_thres01": 0,
            "sum_num_vehicle_been_stopped_thres1": -0.25,
            "pressure": 0  # -0.25
            },

        "LANE_NUM": {
            "LEFT": 1,
            "RIGHT": 1,
            "STRAIGHT": 1
            },

        "PHASE": {
            "sumo": {
                0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                },
            "anon": {
                    # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                    # 'WSWL',
                    # 'ESEL',
                    # 'WSES',
                    # 'NSSS',
                    # 'NSNL',
                    # 'SSSL',
                    },
                }
        }

    dic_path= {
            "PATH_TO_MODEL": "/colight/Mylight/records",
            "PATH_TO_WORK_DIRECTORY": "/colight/Mylight/records",

            "PATH_TO_DATA": "data/test/",
            "PATH_TO_ERROR": "error/test/"
        }
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                    "round_" + str(0), "generator_" + str(0))

    env = AnonEnv(path_to_log, dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf)
    env.reset()
    print("finish")






