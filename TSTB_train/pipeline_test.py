import json
import os
import shutil
import xml.etree.ElementTree as ET
from generator import Generator
from construct_sample import ConstructSample
from updater import Updater
from multiprocessing import Process, Pool
from model_pool import ModelPool
import random
import pickle
import model_test_compare as model_test
import pandas as pd
import numpy as np
from math import isnan
import sys
import time
import traceback

class Pipeline:
    _LIST_SUMO_FILES = [
        "cross.tll.xml",
        "cross.car.type.xml",
        "cross.con.xml",
        "cross.edg.xml",
        "cross.net.xml",
        "cross.netccfg",
        "cross.nod.xml",
        "cross.sumocfg",
        "cross.typ.xml"
    ]

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    #检查是否存在路径，不存在则创建
    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            #if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            #    raise FileExistsError
            #else:
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            #if self.dic_path["PATH_TO_MODEL"] != "model/default":
            #    raise FileExistsError
            #else:
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_MODEL"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

        if os.path.exists(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
            pass
        else:
            os.makedirs(self.dic_path["PATH_TO_PRETRAIN_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_sumo_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files
        for file_name in self._LIST_SUMO_FILES:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(path, file_name))
        for file_name in self.dic_exp_conf["TRAFFIC_FILE"]:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(path, file_name))

    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files

        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                        os.path.join(path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_exp_conf["ROADNET_FILE"]))

    def _modify_sumo_file(self, path=None):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # modify sumo files
        self._set_traffic_file(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "cross.sumocfg"),
                               os.path.join(path, "cross.sumocfg"),
                               self.dic_exp_conf["TRAFFIC_FILE"])

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, road_net):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.road_net = road_net

        # do file operations
        self._path_check()
        self._copy_conf_file()
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'sumo':
            self._copy_sumo_file()
            self._modify_sumo_file()
        elif self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
            self._copy_anon_file()
        # test_duration
        self.test_duration = []

        sample_num = 10 if self.dic_traffic_env_conf["NUM_INTERSECTIONS"]>=10 else min(self.dic_traffic_env_conf["NUM_INTERSECTIONS"], 9)
        print("sample_num for early stopping:", sample_num)
        
        #随机选择其中所有路口中的sample_num个
        self.sample_inter_id = random.sample(range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]), sample_num)


    def early_stopping(self, dic_path, cnt_round): # Todo multi-process
        print("decide whether to stop")
        #earlystopping过程开始的时间
        early_stopping_start_time = time.time()
        record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "round_"+str(cnt_round))

        ave_duration_all = []
        # compute duration
        for inter_id in self.sample_inter_id:
            try:
                df_vehicle_inter_0 = pd.read_csv(os.path.join(record_dir, "vehicle_inter_{0}.csv".format(inter_id)),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])
                duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
                #计算不同的交通流在该路口的平均通行时间
                ave_duration = np.mean([time for time in duration if not isnan(time)])
                ave_duration_all.append(ave_duration)
            except FileNotFoundError:
                error_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
                if not os.path.exists(error_dir):
                    os.makedirs(error_dir)
                f = open(os.path.join(error_dir, "error_info.txt"), "a")
                f.write("Fail to read csv of inter {0} in early stopping of round {1}\n".format(inter_id, cnt_round))
                f.close()
                pass
                
        #计算所选择sample的路口平均通行时间
        ave_duration = np.mean(ave_duration_all)
        self.test_duration.append(ave_duration)
        #earlystopping停止的时间
        early_stopping_end_time = time.time()
        
        print("early_stopping time: {0}".format(early_stopping_end_time - early_stopping_start_time) )
        
        
        if len(self.test_duration) < 30:
            return 0
        else:
        
            #抽取test_duration后15个回合的平均通行时间(reward,或者loss)，并计算均值，标准差，最大值
            duration_under_exam = np.array(self.test_duration[-15:])
            mean_duration = np.mean(duration_under_exam)
            std_duration = np.std(duration_under_exam)
            max_duration = np.max(duration_under_exam)
            #根据相关标准差小于均值的0.1且最大值小于1.5倍均值
            if std_duration/mean_duration < 0.1 and max_duration < 1.5 * mean_duration:
                return 1
            else:
                return 0

    #进行与环境的交互，采取动作，产生reward
    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          best_round=None):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return

    
    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            best_round=best_round,
            bar_round=bar_round
        ) 

        updater.load_sample_for_agents()
        updater.update_network_for_agents()
        print("updater_wrapper end")
        return

    def model_pool_wrapper(self, dic_path, dic_exp_conf, cnt_round):
        model_pool = ModelPool(dic_path, dic_exp_conf)
        model_pool.model_compare(cnt_round)
        model_pool.dump_model_pool()


        return
        #self.best_round = model_pool.get()
        #print("self.best_round", self.best_round)

    def downsample(self, path_to_log, i):

        path_to_pkl = os.path.join(path_to_log, "inter_{0}.pkl".format(i))
        with open(path_to_pkl, "rb") as f_logging_data:
            try:
                #pickle.load()反序列化对象，将文件中的数据解析为一个python对象。file中有read()接口和readline()接口
                logging_data = pickle.load(f_logging_data)
                subset_data = logging_data[::10]
                print(subset_data)
                os.remove(path_to_pkl)
                with open(path_to_pkl, "wb") as f_subset:
                    try:
                        pickle.dump(subset_data, f_subset)
                    except Exception as e:
                        print("----------------------------")
                        print("Error occurs when WRITING pickles when down sampling for inter {0}".format(i))
                        print('traceback.format_exc():\n%s' % traceback.format_exc())
                        print("----------------------------")

            except Exception as e:
                # print("CANNOT READ %s"%path_to_pkl)
                print("----------------------------")
                print("Error occurs when READING pickles when down sampling for inter {0}, {1}".format(i, f_logging_data))
                print('traceback.format_exc():\n%s' % traceback.format_exc())
                print("----------------------------")


    def downsample_for_system(self, path_to_log, dic_traffic_env_conf):
        for i in range(dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.downsample(path_to_log, i)

    def construct_sample_multi_process(self, train_round, cnt_round, batch_size=200):
        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                             dic_traffic_env_conf=self.dic_traffic_env_conf)
        if batch_size > self.dic_traffic_env_conf['NUM_INTERSECTIONS']:
            batch_size_run = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, self.dic_traffic_env_conf['NUM_INTERSECTIONS'], batch_size_run):
            start = batch
            stop = min(batch + batch_size, self.dic_traffic_env_conf['NUM_INTERSECTIONS'])
            process_list.append(Process(target=self.construct_sample_batch, args=(cs, start, stop)))

        for t in process_list:
            t.start()
        for t in process_list:
            t.join()

    def construct_sample_batch(self, cs, start,stop):
        for inter_id in range(start, stop):
            print("make construct_sample_wrapper for ", inter_id)
            cs.make_reward(inter_id)
        

    def run(self, multi_process=False):

        best_round, bar_round = None, None
        #"PATH_TO_WORK_DIRECTORY": "records/default",
        f_time = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],"{0}_running_time.csv".format(self.dic_traffic_env_conf["MODEL_NAME"])),"w")
        #记录各个阶段的运行时间来帮助进行运行时间分析
        f_time.write("generator_time,\tmaking_samples_time,\tupdate_network_time,\ttest_evaluation_times,\tall_times\n")
        f_time.close()


        #假如是预训练阶段,与训练阶段使用random模型
        if self.dic_exp_conf["PRETRAIN"]:
            #用于返回指定的文件夹包含的文件或文件夹的名字的列表。
            if os.listdir(self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
                for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
                    #TODO:only suitable for CoLight
                    #将前者的文件内容复制到后者中
                    shutil.copy(os.path.join(self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                            "round_0_inter_%d.h5" % i),
                                os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_%d.h5"%i))
            else:
                #没有PATH_TO_PRETRAIN_MODEL也没有PATH_TO_PRETRAIN_WORK_DIRECTORY
                if not os.listdir(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
                    for cnt_round in range(self.dic_exp_conf["PRETRAIN_NUM_ROUNDS"]):
                        print("round %d starts" % cnt_round)

                        process_list = []

                        # ==============  generator =============
                        if multi_process:
                            #根据预训练的generator数来进行generator_wrapper的生成，这个是干什么用的，和sample有什么区别？？
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):
                                p = Process(target=self.generator_wrapper, 
                                            args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                                  self.dic_agent_conf, self.dic_traffic_env_conf, best_round)
                                            )
                                print("before start")
                                p.start()
                                print("end start")
                                process_list.append(p)
                            print("before join")
                            for p in process_list:
                                p.join()
                            print("end join")
                        else:
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):
                                self.generator_wrapper(cnt_round=cnt_round,
                                                       cnt_gen=cnt_gen,
                                                       dic_path=self.dic_path,
                                                       dic_exp_conf=self.dic_exp_conf,
                                                       dic_agent_conf=self.dic_agent_conf,
                                                       dic_traffic_env_conf=self.dic_traffic_env_conf,
                                                       best_round=best_round)

                        # ==============  make samples =============
                        # make samples and determine which samples are good

                        train_round = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round")
                        if not os.path.exists(train_round):
                            os.makedirs(train_round)
                        #构建转移，存储s,a,r,s'的转移
                        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                             dic_traffic_env_conf=self.dic_traffic_env_conf)
                        #评测相关reward
                        cs.make_reward()


                #假如该模型需要更新，就进行相关的参数更新
                if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                    if multi_process:
                        p = Process(target=self.updater_wrapper,
                                    args=(0,
                                          self.dic_agent_conf,
                                          self.dic_exp_conf,
                                          self.dic_traffic_env_conf,
                                          self.dic_path,
                                          best_round))
                        p.start()
                        p.join()
                    else:
                        self.updater_wrapper(cnt_round=0,
                                             dic_agent_conf=self.dic_agent_conf,
                                             dic_exp_conf=self.dic_exp_conf,
                                             dic_traffic_env_conf=self.dic_traffic_env_conf,
                                             dic_path=self.dic_path,
                                             best_round=best_round)
        
        
        # train with aggregate samples  假如使用总样本进行训练
        if self.dic_exp_conf["AGGREGATE"]:
            if "aggregate.h5" in os.listdir("model/initial"):
                shutil.copy("model/initial/aggregate.h5",
                            os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(0,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round))
                    p.start()
                    p.join()
                else:
                    self.updater_wrapper(cnt_round=0,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round)

        self.dic_exp_conf["PRETRAIN"] = False
        self.dic_exp_conf["AGGREGATE"] = False


        #这里for每一个round
        for cnt_round in range(1):
        
            #在每一个round都会进行generator生成，makesample,update,test,early-stopping,model pool evaluation
            print("round %d starts" % cnt_round)
            round_start_time = time.time()
            process_list = []
            print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            if multi_process:
                p = Process(target=model_test.test,
                            args=(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, False, self.road_net))
                p.start()
                if self.dic_exp_conf["EARLY_STOP"]:
                    p.join()
            else:
                model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"], self.dic_traffic_env_conf, False, self.road_net)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time


            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))
            


