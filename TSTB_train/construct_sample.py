import numpy as np
import pickle
from multiprocessing import Pool, Process
import os
import traceback
import pandas as pd

class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        #传入的构建sample的文件夹path
        self.parent_dir = path_to_samples
        #构建该轮的文件夹path
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        #交通环境信息
        self.dic_traffic_env_conf = dic_traffic_env_conf
        #存储全部log数据
        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples = []
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']

    #加载文件夹中的内容，加载成功则会返回flag=1，以及相关数据logging_data
    def load_data(self, folder, i):

        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data

        except Exception as e:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None



    #记录全部log数据进入logging_data_list_per_gen
    def load_data_for_system(self, folder):
        '''
        Load data for all intersections in one folder
        :param folder:
        :return: a list of logging data of one intersection for one folder
        '''
        self.logging_data_list_per_gen = []
        # load settings
        print("Load data for system in ", folder)
        #measure_time是进行长期reward评测时的测量时长，设置为10，及计算在接下来10s内的10个reward的平均值
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)
            '''
            维度为i*time，i为路口编号（0-36），time为系统运行期间的时间间隔数标号（0-360）
            load结束的logging_data数据内容以字典形式存储如下：
            {'time': 3590.0, 
             'state': {'cur_phase': [1], 
                       'time_this_phase': [35], 
                       'vehicle_position_img': None, 
                       'vehicle_speed_img': None, 
                       'vehicle_acceleration_img': None, 
                       'vehicle_waiting_time_img': None, 
                       'lane_num_vehicle': [0, 3, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], 
                       'pressure': None, 
                       'coming_vehicle': None, 
                       'leaving_vehicle': None, 
                       'lane_num_vehicle_been_stopped_thres01': None, 
                       'lane_num_vehicle_been_stopped_thres1': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], 
                       'lane_queue_length': None, 
                       'lane_num_vehicle_left': None, 
                       'lane_sum_duration_vehicle_left': None, 
                       'lane_sum_waiting_time': None, 
                       'terminal': None, 
                       'adjacency_matrix': [1, 2, 7, 0, 8], 
                       'adjacency_matrix_lane': [[[-1, -1, -1, -1, -1], [33, 34, 35, -1, -1]], 
                                                 [[-1, -1, -1, -1, -1], [84, 85, 86, -1, -1]], 
                                                 [[-1, -1, -1, -1, -1], [6, 7, 8, -1, -1]], 
                                                 [[93, 88, 92, -1, -1], [6, 7, 8, -1, -1]], 
                                                 [[93, 88, 92, -1, -1], [-1, -1, -1, -1, -1]], 
                                                 [[93, 88, 92, -1, -1], [33, 34, 35, -1, -1]], 
                                                 [[26, 27, 31, -1, -1], [84, 85, 86, -1, -1]], 
                                                 [[26, 27, 31, -1, -1], [6, 7, 8, -1, -1]], 
                                                 [[26, 27, 31, -1, -1], [-1, -1, -1, -1, -1]], 
                                                 [[0, 10, 5, -1, -1], [-1, -1, -1, -1, -1]], 
                                                 [[0, 10, 5, -1, -1], [33, 34, 35, -1, -1]], 
                                                 [[0, 10, 5, -1, -1], [84, 85, 86, -1, -1]]], 
                       'connectivity': [6.0, 0.5468649546968608, 0.4723665527410147, 1.6405948640905823, 0.7788007830714049]
                      }, 
             'action': 0}
            
         
            '''
            
            if pass_code == 0:
                print('Loading data failed')
                return 0
            #size(logging_data_list_per_gen): 路口数*360
            self.logging_data_list_per_gen.append(logging_data)
        return 1


    def load_hidden_state_for_system(self, folder):
        print("loading hidden states: {0}".format(os.path.join(self.path_to_samples, folder, "hidden_states.pkl")))
        # load settings
        if self.hidden_states_list is None:
            self.hidden_states_list = []

        try:
            f_hidden_state_data = open(os.path.join(self.path_to_samples, folder, "hidden_states.pkl"), "rb")
            hidden_state_data = pickle.load(f_hidden_state_data) # hidden state_data is a list of numpy array
            # print(hidden_state_data)
            print(len(hidden_state_data))
            hidden_state_data_h_c = np.stack(hidden_state_data, axis=2)
            hidden_state_data_h_c = pd.Series(list(hidden_state_data_h_c))
            next_hidden_state_data_h_c = hidden_state_data_h_c.shift(-1)
            hidden_state_data_h_c_with_next = pd.concat([hidden_state_data_h_c,next_hidden_state_data_h_c], axis=1)
            hidden_state_data_h_c_with_next.columns = ['cur_hidden','next_hidden']
            self.hidden_states_list.append(hidden_state_data_h_c_with_next[:-1].values)
            return 1
        except Exception as e:
            print("Error occurs when loading hidden states in ", folder)
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    #接受features,时间和路口编号来进行state构建
    def construct_state(self,features,time,i):
        '''
        :param features:
        :param time:
        :param i:  intersection id
        :return:
        '''

        state = self.logging_data_list_per_gen[i][time]
        assert time == state["time"]
        #BINARY_PHASE_EXPANSION默认为true
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        # print(state_after_selection)
        return state_after_selection


    def construct_state_process(self, features, time, state, i):
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection, i

    #返回的是一个reward字典，存储了不同的影响因素对应的reward
    def get_reward_from_features(self, rs):
        #print("rs which will be np.sum:",rs)
        reward = {}
        reward["sum_lane_queue_length"] = np.sum(rs["lane_queue_length"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_sum_waiting_time"])
        reward["sum_lane_num_vehicle_left"] = np.sum(rs["lane_num_vehicle_left"])
        reward["sum_duration_vehicle_left"] = np.sum(rs["lane_sum_duration_vehicle_left"])
        reward["sum_num_vehicle_been_stopped_thres01"] = np.sum(rs["lane_num_vehicle_been_stopped_thres01"])
        reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(rs["lane_num_vehicle_been_stopped_thres1"])
        ##TODO pressure
        reward["pressure"] = np.sum(rs["pressure"])
        reward["lane_num_sp_vehicle_been_stopped_thres1"] = np.sum(rs["lane_num_sp_vehicle_been_stopped_thres1"])
        reward["sp_pressure"] = np.sum(rs["sp_pressure"])
               
        return reward


    def cal_reward(self, rs, rewards_components):
        r = 0
        #print("rs values:",rs)
        #print("rs components:",rewards_components)
        #rewards_compomemts是reward各个部分的权重占比参数
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        return r

    #reward_components是该算法计算reward时所需要考虑的元素，返回动作在下一秒的即时reward，以及之后10s的平均reward（注意不是累计收益）
    def construct_reward(self,rewards_components,time, i):

        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])
        r_instant = self.cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):
            #print("t is ", t)
            rs = self.logging_data_list_per_gen[i][t]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average


    def judge_action(self,time,i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']


    def make_reward(self, folder, i):
        '''
        make reward for one folder and one intersection,
        add the samples of one intersection into the list.samples_all_intersection[i]
        :param i: intersection id
        :return:
        '''
        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = []

        if i % 5 == 0:
            print("make reward for inter {0} in folder {1}".format(i, folder))

        list_samples = []

        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)
            # construct samples
            time_count = 0
            #measure_time默认值设置为10
            for time in range(0, total_time - self.measure_time + 1, self.interval):
                
                #logging_data_list_per_gen记录了全部转移数据，根据LIST_STATE_FEATURE的特征，来选择构建专属state
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                reward_instant, reward_average = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"],
                                                                       time, i)
                #判断action是否为-1，判断合法性
                action = self.judge_action(time, i)

                #如果time + self.interval > total_time 呢
                if time + self.interval == total_time:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval - 1, i)

                else:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval, i)
                                                      
                #这里的sample中，转移的时间间隔是自己设定的，默认为10s，reward分为下一秒的即时reward和时间间隔内的平均reward
                sample = [state, action, next_state, reward_average, reward_instant, time,
                          folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)


            # list_samples = self.evaluate_sample(list_samples)
            self.samples_all_intersection[i].extend(list_samples)
            return 1
        except Exception as e:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0


    def make_reward_for_system(self):
        '''
        Iterate all the generator folders, and load all the logging data for all intersections for that folder
        At last, save all the logging data for that intersection [all the generators]
        :return:
        '''
        for folder in os.listdir(self.path_to_samples):
            print(folder)
            #未找到generator就继续
            if "generator" not in folder:
                print('cannot find generator')
                continue
                
            #该条sample不符合要求 or 加载数据不成功
            if not self.evaluate_sample(folder) or not self.load_data_for_system(folder):
                print('sample is not satisfied')
                continue

            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                
            #make reward成功则返回1
                pass_code = self.make_reward(folder, i)
                if pass_code == 0:
                    continue

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.dump_sample(self.samples_all_intersection[i],"inter_{0}".format(i))


    def dump_hidden_states(self, folder):
        total_hidden_states = np.vstack(self.hidden_states_list)
        print("dump_hidden_states shape:",total_hidden_states.shape)
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_hidden_states.pkl"),"ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_hidden_states_{0}.pkl".format(folder)),"ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "hidden_states_{0}.pkl".format(folder)),'wb') as f:
                pickle.dump(total_hidden_states, f, -1)


    # def evaluate_sample(self,list_samples):
    #     return list_samples

    #评测一条sample是否能够用来进行训练
    def evaluate_sample(self, generator_folder):
        return True
        print("Evaluate samples")
        #查找到sample路径下，几个generator文件夹
        list_files = os.listdir(os.path.join(self.path_to_samples, generator_folder, ""))
        df = []
        # print(list_files)
        for file in list_files:
            if ".csv" not in file:
                continue
            data = pd.read_csv(os.path.join(self.path_to_samples, generator_folder, file))
            df.append(data)
        df = pd.concat(df)
        #df.isna()判断是否为空
        #前一部分应该是总的驶入车辆数，后一部分是还未驶出的车辆数
        num_vehicles = len(df['Unnamed: 0'].unique()) -len(df[df['leave_time'].isna()]['leave_time'])
        #volume设置为300，当真实流量小于额定设置流量数*行数且训练回合数>40时，丢弃该条样本，因为通过的流量数较小
        if num_vehicles < self.dic_traffic_env_conf['VOLUME']* self.dic_traffic_env_conf['NUM_ROW'] and self.cnt_round > 40: # Todo Heuristic
            print("Dumpping samples from ",generator_folder)
            return False
        else:
            return True


    def dump_sample(self, samples, folder):
        if folder == "":
        #ab+以二进制方式打开，self.parent_dir是传入construct_sample函数的path_to_sample
            with open(os.path.join(self.parent_dir, "total_samples.pkl"),"ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)),"ab+") as f:
                pickle.dump(samples, f, -1)
        #self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)),'wb') as f:
                pickle.dump(samples, f, -1)


if __name__=="__main__":
    path_to_samples = "/Users/Wingslet/PycharmProjects/RLSignal/records/test/anon_3_3_test/train_round"
    generator_folder = "generator_0"

    dic_traffic_env_conf  = {

            "NUM_INTERSECTIONS": 9,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "MIN_ACTION_TIME": 10,
            "DEBUG": False,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,

            "NEIGHBOR": False,
            "MODEL_NAME": "STGAT",
            "SIMULATOR_TYPE": "anon",



            "SAVEREPLAY": False,
            "NUM_ROW": 3,
            "NUM_COL": 3,

            "VOLUME": 300,
            "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),

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
                    1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                    # 'WSWL',
                    # 'ESEL',
                    # 'WSES',
                    # 'NSSS',
                    # 'NSNL',
                    # 'SSSL',
                },
            }
        }

    cs = ConstructSample(path_to_samples, 0, dic_traffic_env_conf)
    cs.make_reward_for_system()

