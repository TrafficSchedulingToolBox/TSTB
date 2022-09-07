import os
import copy
from config import DIC_AGENTS, DIC_ENVS
import time
import sys
from multiprocessing import Process, Pool

class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, best_round=None):
        
        
        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']

        if self.dic_exp_conf["PRETRAIN"]:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
        else:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        #生成器是用来生成环境以及相关的智能体。 多线程调用生成器，平行环境进行训练？
        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log = self.path_to_log,
                              path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = self.dic_traffic_env_conf)        
        self.env.reset()

        # every generator's output
        # generator for pretraining
        # Todo pretrain with intersection_id
        if self.dic_exp_conf["PRETRAIN"]:
            #使用默认预训练模型：random生成初始数据
            self.agent_name = self.dic_exp_conf["PRETRAIN_MODEL_NAME"]
            
            
            #调用config中的DIC_AGENTS, 而config调用Colight_agent.py 进行智能体网络的初始化
            self.agent = DIC_AGENTS[self.agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                best_round=best_round,
            )

        else:

            start_time = time.time()

            for i in range(dic_traffic_env_conf['NUM_AGENTS']):
                #调用需要进行训练的模型
                agent_name = self.dic_exp_conf["MODEL_NAME"]
                #the CoLight_Signal needs to know the lane adj in advance, from environment's intersection list
                #Colight相比于其他方法，调用增加环境当中定义的智能体list
                if agent_name=='CoLight_Signal':
                #这个地方用的是CoLight_Signal,这个模型和Colight是不一样的
                #if agent_name=='Mylight':
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round, 
                        best_round=best_round,
                        inter_info=self.env.list_intersection,         #colight add the intersection information,
                        intersection_id=str(i)
                    )      
                else:              
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=self.dic_agent_conf,
                        dic_traffic_env_conf=self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        cnt_round=self.cnt_round,
                        best_round=best_round,
                        intersection_id=str(i)
                    )
                self.agents[i] = agent
                
            #调用创建智能体，以及环境所花费的时间
            print("Create intersection agent time: ", time.time()-start_time)



    def generate(self):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time

        running_start_time = time.time()
        ###################################################################################################################
        #在此处加入对于car的相关调度
        '''
        
        '''
        while not done and step_num < int(self.dic_exp_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            step_start_time = time.time()

            #对于每一个生成的智能体
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                #假如智能体是MyLight,Colight,GCN,SimpleDQNOne, state使用全局状态吗？？
                #目前来看，所有的agent都会接受一个全局信息来进行动作选择
                if self.dic_exp_conf["MODEL_NAME"] in ["CoLight","GCN", "SimpleDQNOne", "MyLight"]:
                    one_state = state
                    
                    if self.dic_exp_conf["MODEL_NAME"] == 'MyLight':
                        action, _ = self.agents[i].choose_action(step_num, one_state)
                        
                    elif self.dic_exp_conf["MODEL_NAME"] == 'CoLight':
                        action, _ = self.agents[i].choose_action(step_num, one_state)
                        
                    elif self.dic_exp_conf["MODEL_NAME"] == 'GCN':
                        action = self.agents[i].choose_action(step_num, one_state)
                    else: # simpleDQNOne
                        if True:
                            action = self.agents[i].choose_action(step_num, one_state)
                        else:
                            action = self.agents[i].choose_action_separate(step_num, one_state)
                    #这三种方法应该是全局公用一个智能体，输出的action直接是所有的路口的调控动作的集合
                    action_list = action
                #其他智能体，以及固定控制规则，仅仅提取单个路口的局部状态？？
                else:
                    one_state = state[i]
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list.append(action)
            #env.step需要收集所有agent的动作，然后进行下一步的状态迭代变化
            next_state, reward, done, _ = self.env.step(action_list)
            #running time 是指获得状态信息后，进行动作选择，已经state更新所花费的时间
            print("time: {0}, step_num: {1} running_time: {2}".format(self.env.get_current_time()-self.dic_traffic_env_conf["MIN_ACTION_TIME"], 
                                                    step_num, time.time()-step_start_time))
            state = next_state
            step_num += 1
        
        
        running_time = time.time() - running_start_time

        log_start_time = time.time()
        print("start logging")
        #？？？
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time

        self.env.end_sumo()
        #generator开始时，重启设置环境的时间
        print("reset_env_time: ", reset_env_time)
        #整个step过程花费的时间
        print("running_time: ", running_time)
        #生成记录文件所花费的时间
        print("log_time: ", log_time)
