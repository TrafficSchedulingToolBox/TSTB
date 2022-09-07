import config
import copy
from pipeline import Pipeline
import os
import time
from multiprocessing import Process
import argparse
import matplotlib
# matplotlib.use('TkAgg')

from script import get_traffic_volume

multi_process = False
TOP_K_ADJACENCY=-1
TOP_K_ADJACENCY_LANE=-1
PRETRAIN=False
NUM_ROUNDS=15
EARLY_STOP=False 
NEIGHBOR=False
SAVEREPLAY=False
ADJACENCY_BY_CONNECTION_OR_GEO=False
hangzhou_archive=True
ANON_PHASE_REPRE=[]

def parse_args():
    parser = argparse.ArgumentParser()
    # The file folder to create/log in
    parser.add_argument("--memo", type=str, default='0525_SimpleDQNOne_hefei_40per')#1_3,2_2,3_3,4_4
    parser.add_argument("--env", type=int, default=1)  #env=1 means you will run CityFlow
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--road_net", type=str, default='4_4')#'3_4') # which road net you are going to run
    
    #volume决定了后边template文件夹的取值设置，可以取data中各个文件夹的数据，具体设置见line 394
    parser.add_argument("--volume", type=str, default='hangzhou')#'300',
    
    #后缀信息，比如data中的Jinan数据集，后缀为real或者real_2000，对于template_lsr文件夹，后缀为0.3_bi或者0.3_uni
    parser.add_argument("--suffix", type=str, default="real_5816")#'0.3_bi'
    
    #parser.add_argument("--mod", type=str, default='CoLight')   #SimpleDQN,SimpleDQNOne,CoLight,Lit,MyLight
    parser.add_argument("--mod", type=str, default='MyLight')
    
    #cnt为每一个训练周期迭代的秒数
    parser.add_argument("--cnt",type=int, default=3600) #3600
    #generator的数目
    
    parser.add_argument("--gen",type=int, default=4)#4
    #all是针对虚拟生成数据template_ls中的数据，设置为all则对其进行全部测试
    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--workers",type=int, default=7)
    parser.add_argument("--onemodel",type=bool, default=False)
    #sp_vehicle是否考虑特殊车辆，True表示在state中加入相关信息
    parser.add_argument("--sp_vehicle",type=bool,default=False)
    parser.add_argument("--sp_vehicle_proportion",type=float,default=0.01)
    
#######################所有地与实验参数设置相关的global parameter都在line200中并入traffic_env_conf_extra作补充
    global hangzhou_archive
    hangzhou_archive=False
    
    #top-k = 5,top-k = 6
    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY=5
    
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE=5
    #训练周期数，设置为100
    global NUM_ROUNDS
    NUM_ROUNDS=100  #100
    
    global EARLY_STOP
    EARLY_STOP=True
    
    global NEIGHBOR
    # TAKE CARE，这个是指是否将邻居完全加入进来，我们的方法在构造自身状态时就加入了邻居状态，因此设置为False
    NEIGHBOR=False
    
    global SAVEREPLAY # if you want to relay your simulation, set it to be True
    SAVEREPLAY=False
    
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE,这里使用False是使用地理距离信息，使用True是仅仅基于图的连接信息
    ADJACENCY_BY_CONNECTION_OR_GEO=False
    
    global DIRECTED_GRAPH
    #当使用我们的Mylight+GAT模型时，该值置为True，因为基于有向图生成相关连接信息
    DIRECTED_GRAPH=True
    
    #modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN=False
    
    
    #################  gpu是否可用
    parser.add_argument("--visible_gpu", type=str, default="-1")
    
    global ANON_PHASE_REPRE
    tt=parser.parse_args()
    if 'CoLight_Signal' in tt.mod:
        #12dim
        ANON_PHASE_REPRE={
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            #0-2:西边路口 3-5:东边路口 6-8:北边路口 9-11：南边路口 ,右转常绿
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES', West straight East straight
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS', North straight South straight
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL', West left East left
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL', North left South left
        }
    else:
        #12dim
        ANON_PHASE_REPRE={
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0]
        }
    print('agent_name:%s',tt.mod)
    print('ANON_PHASE_REPRE:',ANON_PHASE_REPRE)
    

    return parser.parse_args()


def memo_rename(traffic_file_list):
    new_name = ""
    for traffic_file in traffic_file_list:
        if "synthetic" in traffic_file:
            #返回字符串最后一次出现的位置(从右向左查询)
            sta = traffic_file.rfind("-") + 1
            print(traffic_file, int(traffic_file[sta:-4]))
            new_name = new_name + "syn" + traffic_file[sta:-4] + "_"
        elif "cross" in traffic_file:
            sta = traffic_file.find("equal_") + len("equal_")
            end = traffic_file.find(".xml")
            new_name = new_name + "uniform" + traffic_file[sta:end] + "_"
        elif "flow" in traffic_file:
            new_name = traffic_file[:-4]
    new_name = new_name[:-1]
    return new_name

#对字典已有内容进行更新，若无则添加dic_to_change中的新内容,并返回添加后的内容
def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


#对此处进行调用，进行程序运行
def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf, # experiment config
                   dic_agent_conf=dic_agent_conf, # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_conf, # the simolation configuration
                   dic_path=dic_path # where should I save the logs?
                   )
    global multi_process
    ppl.run(multi_process=multi_process)

    print("pipeline_wrapper end")
    return


#环境（sumo或者anon，路网，GUI，流量设置，后缀，模型名称，运行回合，generator的数目，是否测试全部，workers数目，是否全局统一模型）
def main(memo, env, road_net, gui, volume, suffix, mod, cnt, gen, r_all, workers, onemodel, sp_vehicle, sp_vehicle_proportion):

    # main(args.memo, args.env, args.road_net, args.gui, args.volume, args.ratio, args.mod, args.cnt, args.gen)
    if volume=='hefei' and suffix in ['all','20per','40per']:
        NUM_COL = 4
        NUM_ROW = 3
        num_intersections = 11
    else:
        NUM_COL = int(road_net.split('_')[0])          #对路网文件字符串按照_进行分割并取第一个
        NUM_ROW = int(road_net.split('_')[1])
        num_intersections = NUM_ROW * NUM_COL
    

    ENVIRONMENT = ["sumo", "anon"][env]  
    
    #r_all设置为1，则会测试100-400的不同流量情况
    #这里的设置主要是针对 data\template_lsr里的3_3,6_6,10_10的模拟生成数据，假如需要使用data里的其他数据集，修改volume请自行设置
    #此处仅仅进行文件名的设置，文件路径设置，在dic_path_extra中
    if r_all:
        traffic_file_list = [ENVIRONMENT+"_"+road_net+"_%d_%s" %(v,suffix) for v in range(100,400,100)]
    else:
        traffic_file_list=["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)]


    if env:
        traffic_file_list = [i+ ".json" for i in traffic_file_list ]
    else:
        traffic_file_list = [i+ ".xml" for i in traffic_file_list ]

    process_list = []
    n_workers = workers     #len(traffic_file_list)
    multi_process = False

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    for traffic_file in traffic_file_list:
        dic_exp_conf_extra = {
    #根据main里的输入数据，更改相关的dic_exp_conf_extra字典信息
            "RUN_COUNTS": cnt,
            "MODEL_NAME": mod,
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic

            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            
            
            "NUM_ROUNDS": NUM_ROUNDS,
            "NUM_GENERATORS": gen,

            "MODEL_POOL": False,
            "NUM_BEST_MODEL": 3,     #在best_model中最多存储多少个模型
            "PRETRAIN": PRETRAIN,#
            "PRETRAIN_MODEL_NAME":mod,
            "PRETRAIN_NUM_ROUNDS": 0,
            "PRETRAIN_NUM_GENERATORS": 15,

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": EARLY_STOP,
        }
        
        if volume=='hefei':
            dic_exp_conf_extra['ROADNET_FILE'] = 'roadnet_{0}.json'.format(11)

        dic_agent_conf_extra = {
            "EPOCHS": 100,
            "SAMPLE_SIZE": 1000,
            "MAX_MEMORY_LEN": 10000,
            "UPDATE_Q_BAR_EVERY_C_ROUND": False,
            "UPDATE_Q_BAR_FREQ": 5,
            # network

            "N_LAYER": 2,
            "TRAFFIC_FILE": traffic_file,
        }

        global TOP_K_ADJACENCY
        global TOP_K_ADJACENCY_LANE
        global NEIGHBOR
        global SAVEREPLAY
        global ADJACENCY_BY_CONNECTION_OR_GEO
        global ANON_PHASE_REPRE
        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            "ONE_MODEL": onemodel,
            "NUM_AGENTS": num_intersections,
            "SP_VEHICLE": sp_vehicle,
            "SP_VEHICLE_PROPORTION": sp_vehicle_proportion,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            #measure time 具体在哪个地方调用？
            "MEASURE_TIME": 10,
            "IF_GUI": gui,
            "DEBUG": True,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "DIRECTED_GRAPH": DIRECTED_GRAPH,
            
            #这个二进制阶段扩展是什么意思
            "BINARY_PHASE_EXPANSION": True,
            
            "FAST_COMPUTE": True,

            "NEIGHBOR": NEIGHBOR,
            "MODEL_NAME": mod,

            "SAVEREPLAY": SAVEREPLAY,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "VOLUME": volume,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "phase_expansion": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },

            #用于强化学习的特征的提取
            "LIST_STATE_FEATURE": [
                "cur_phase",
                #"time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_sp_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_num_sp_vehicle_waiting_count",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure",
                # "sp_pressure",

                # "adjacency_matrix",
                # "lane_queue_length",
                # "connectivity",

                # "adjacency_matrix_lane"
            ],

    ########################字典存储特征维度
                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),

                    D_COMING_VEHICLE = (12,),
                    D_LEAVING_VEHICLE = (12,),
                    
                    #新增,第一个待定
                    D_COMING_SP_VEHICLE = (12,),
                    D_LEAVING_SP_VEHICLE = (12,),
                    
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

                    D_ADJACENCY_MATRIX_LANE=(6,),
                    
                    #sp_vehicle相关特征维度
                    D_LANE_NUM_SP_VEHICLE=(4,),
                    D_LANE_NUM_SP_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_SP_PRESSURE=(1,),
                    
                ),

#字典存储计算reward相关权重
            "DIC_REWARD_INFO": {
                "flickering": 0,#-5,#
                "sum_lane_queue_length": 0, #-1,
                "sum_lane_wait_time": 0,
                "sum_lane_sp_vehicle_wait_time": 0, #-1,
                "sum_lane_num_vehicle_left": 0,#-1,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1":  0,#-0.25,
                "lane_num_sp_vehicle_been_stopped_thres1": 0,#-2.5,
                "pressure":   0, #-1/(1-sp_vehicle_proportion)
                "sp_pressure": 0, #-1/sp_vehicle_proportion,                        #根据model_pool的229行左右的评价指标，来做决定
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

                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                #     2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                #     3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                #     4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
                "anon":ANON_PHASE_REPRE,
                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                #     2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                #     3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                #     4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
            }
        }

        if volume=='hefei':
            dic_traffic_env_conf_extra['ROADNET_FILE'] = 'roadnet_{0}.json'.format(11)

        if dic_exp_conf_extra ["MODEL_NAME"] == 'Lit':
            dic_traffic_env_conf_extra["DIC_REWARD_INFO"]["sum_lane_queue_length"] = -1
        elif dic_exp_conf_extra ["MODEL_NAME"] == 'CoLight':
            dic_traffic_env_conf_extra["DIC_REWARD_INFO"]["pressure"] = -1
        elif dic_exp_conf_extra ["MODEL_NAME"] == 'MyLight':
            dic_traffic_env_conf_extra["DIC_REWARD_INFO"]["pressure"] = -1/(1-sp_vehicle_proportion)
            dic_traffic_env_conf_extra["DIC_REWARD_INFO"]["sp_pressure"] = -1/sp_vehicle_proportion
        elif dic_exp_conf_extra ["MODEL_NAME"] in ['SimpleDQNOne','SimpleDQN']:
            dic_traffic_env_conf_extra["DIC_REWARD_INFO"]["sum_num_vehicle_been_stopped_thres1"] = -0.25

        ## ==================== multi_phase ====================
        global hangzhou_archive
        if hangzhou_archive:
            template='Archive+2'
        elif volume=='jinan':
            template='Jinan'
        elif volume=='hefei':
            template="Hefei"
        elif volume=='hangzhou':
            template='Hangzhou'
        elif volume=='newyork':
            template='NewYork'
        elif volume=='chacha':
            template='Chacha'
        elif volume=='dynamic_attention':
            template='dynamic_attention'
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
            template = "template_ls"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._S:
            template = "template_s"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LSR:
            template = "template_lsr"
        else:
            raise ValueError

#以下内容是根据智能体的模型选择不同，来修改对应的feature以及feature dim
#################################################################################################################################
        #添加邻居节点的相关状态，包括每一个邻居节点的当前状态，每个车道上的车辆数,MyLight,Colight为false
        if dic_traffic_env_conf_extra['NEIGHBOR']:
            list_feature = dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].copy()
            for feature in list_feature:
                for i in range(4):
                    dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].append(feature+"_"+str(i))

        #针对Colight,GCN,SimpleDQN,Mylight的模型添加相关feature
        
        if mod in ['CoLight','GCN','SimpleDQNOne','MyLight']:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = 1
            dic_traffic_env_conf_extra['ONE_MODEL'] = False
            #添加adjacency_matrix和adjacency_matrix_lane属性
            
            if "adjacency_matrix" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                "adjacency_matrix_lane" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                mod not in ['SimpleDQNOne']:
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix")
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix_lane")
                
                #假如采用邻接地理关系，则是东西南北和自身五个路口的信息，top_k为5
                #Mylight,Colight不使用该信息，设置为false
                if dic_traffic_env_conf_extra['ADJACENCY_BY_CONNECTION_OR_GEO']:
                    TOP_K_ADJACENCY = 5
                    dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("connectivity")
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CONNECTIVITY'] = \
                        (5,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (5,)
                    
                #不使用地理信息，更改D_ADJACENCY_MATRIX的维度为top_k_road
                else:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY'],)

                #使用邻接车道，更改D_ADJACENCY_MATRIX_LANE的数值为top_k_lane，Mylight,Colight需要添加，数值也为5
                if dic_traffic_env_conf_extra['USE_LANE_ADJACENCY']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX_LANE'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY_LANE'],)
                        
            #添加sp_vehicle所需的相关特征         
            if sp_vehicle: 
                if "lane_num_sp_vehicle" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                "lane_num_sp_vehicle_waiting_count" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                "sp_pressure" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE']:
                        #车道上的特殊车辆数
                        dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("lane_num_sp_vehicle")
                        #处于拥堵或者停止状态的特殊车辆
                        dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("lane_num_sp_vehicle_been_stopped_thres1")
                        #特殊车辆压力
                        #dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("sp_pressure")
                                                                        
                                                
        #其他方式使用每个路口作为一个智能体进行训练,智能体数等于路口数
        else:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        #重新定义相关状态空间的维度，Colight和Mylight有八种状态组合情况
        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
            
            if mod in ['MyLight']:
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_BEEN_STOPPED_THRES1_0'] = (4,)
                
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_BEEN_STOPPED_THRES1_1'] = (4,)
                
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_BEEN_STOPPED_THRES1_2'] = (4,)
                
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_3'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_SP_VEHICLE_BEEN_STOPPED_THRES1_3'] = (4,)                
            
            else:
                #这当设置了邻居信息时，还需要在状态当中
                if dic_traffic_env_conf_extra['NEIGHBOR']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
                else:
                #这个地方还需要设置，为什么时4而不是12，另外1和8的设置是什么，为什么有8个维度
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
      
        print(traffic_file)
        
        
        prefix_intersections = str(road_net)
        
        #设置相关的调用文件路径，mome是环境名，基于sumo还是anon
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),

            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", memo)
        }
###########################################################################################################################
        #将dic_exp_conf_extra的参数更新到config.DIC_EXP_CONF中
        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        #将dic_agent_conf_extra的参数更新到config.DIC_MOD_AGENT_CONF中
        deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(mod.upper())),
                                      dic_agent_conf_extra)
        #将dic_traffic_env_conf_extra的参数更新到config.dic_traffic_env_conf                        
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
###########################################################################################################################
        # TODO add agent_conf for different agents
        # deploy_dic_agent_conf_all = [deploy_dic_agent_conf for i in range(deploy_dic_traffic_env_conf["NUM_AGENTS"])]

        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)



        #在该段调用多线程执行的准备环节，为每个进行相关模型训练，主要存在四个关键参数
        if multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                             dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)


    return memo


if __name__ == "__main__":
    args = parse_args()
    #memo = "multi_phase/optimal_search_new/new_headway_anon"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    main(args.memo, args.env, args.road_net, args.gui, args.volume,
         args.suffix, args.mod, args.cnt, args.gen, args.all, args.workers,
         args.onemodel, args.sp_vehicle, args.sp_vehicle_proportion)



