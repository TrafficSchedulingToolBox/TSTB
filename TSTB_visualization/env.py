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



class AnonEnv:
    list_intersection_id = [
        "intersection_1_1"
    ]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):
        self_path_to_log = path_to_log
        self_path_to_work_directory = path_to_work_directory
        self_dic_traffic_env_conf = dic_traffic_env_conf
        self_simulator_type = self_dic_traffic_env_conf["SIMULATOR_TYPE"]

        self_list_intersection = None
        self_list_inter_log = None
        self_list_lanes = None
        self_system_states = None
        self_feature_name_for_neighbor = self._reduce_duplicates(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])

        # check min action time
        if self_dic_traffic_env_conf["MIN_ACTION_TIME"] <= self_dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            #raise ValueError


        os.makedirs(self_path_to_log)

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self_dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self_path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        
        print("# self.eng.reset() to be implemented")
        cityflow_config = {
            "interval": self_dic_traffic_env_conf["INTERVAL"],
            "seed": 0,
            "laneChange": False,
            "dir": self_path_to_work_directory+"/",
            "roadnetFile": "roadnet_3_4.json",
            "flowFile": "anon_3_4_jinan_real.json",
            "rlTrafficLight": self_dic_traffic_env_conf["RLTRAFFICLIGHT"],
            "saveReplay": self_dic_traffic_env_conf["SAVEREPLAY"],
            "roadnetLogFile": "roadnetLogFile.json",
            "replayLogFile": "replayLogFile.txt"
        }
            
        with open(os.path.join(self_path_to_work_directory,"cityflow_config.json"), "w") as json_file:
            json.dump(cityflow_config, json_file)
        self_eng = engine.Engine(os.path.join(self_path_to_work_directory,"cityflow_config.json"), thread_num=1)   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
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
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
        }

    dic_exp_conf = {
        "RUN_COUNTS": 3600,
        "MODEL_NAME": "STGAT",


        "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),

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
        "TRAFFIC_FILE": "/mnt/RLSignal_general/records/test/anon_3_3_test/anon_3_3_700_1.0.json",
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
        "FAST_COMPUTE": True,
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
            
            "PATH_TO_MODEL": "/colight/Mylight/model",
            "PATH_TO_WORK_DIRECTORY": "/colight/Mylight/records",
            "PATH_TO_DATA": "colight/Mylight/data/",
            "PATH_TO_ERROR": "/colight/Mylight/errors"
        }
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                    "round_" + str(0), "generator_" + str(0))

    env = AnonEnv(path_to_log, dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf)
    env.reset()
    print("finish")