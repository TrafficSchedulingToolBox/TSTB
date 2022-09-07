import numpy as np
from matplotlib import pyplot as plt
import csv

class Flow:
    def __init__(self, enter_time, leave_time, sp, distance):
        self.time = leave_time - enter_time

        if sp == 1:
            self.sp = True
        else:
            self.sp = False

        self.distance = distance

        # print("time:", self.time)
        # print("distance: ", self.distance)
        self.time_per_km = self.time / (self.distance / 1000.0)



class Round:
    def __init__(self, time_no_sp, time_sp, distance_no_sp, distance_sp, time_per_km_sp, time_per_km_no_sp):
        self.time_no_sp = time_no_sp
        self.time_sp = time_sp
        self.distance_no_sp = distance_no_sp
        self.distance_sp = distance_sp
        self.time_per_km_sp = time_per_km_sp
        self.time_per_km_no_sp = time_per_km_no_sp
        self.model = ""

    def show(self):
        # pass
        # print("Model: ", self.model.split("/")[2])
        print("Average Travel Time for Normal Vechile: ", self.time_no_sp)
        print("Average Travel Time for Special Vechile: ", self.time_sp)
        print("Average Travel Distance for Normal Vechile: ", self.distance_no_sp)
        print("Average Travel Distance for Special Vechile: ", self.distance_sp)
        print("Average Travel Time per Kilometer for Normal Vechile: ", self.time_per_km_no_sp)
        print("Average Travel Time per Kilometer for Special Vechile: ", self.time_per_km_sp)
        print()


def process_a_model(model, round_number):
    rounds = []
    for i in range(round_number):
        filename = model + 'round_' + str(i) + "/all_vehicle.csv"
        # print("---------- Round ", i, " ----------")
        round = process_a_file(filename)
        rounds.append(round)

    x = np.arange(0, round_number)
    time_sps = []
    time_no_sps = []
    distance_sps = []
    distance_no_sps = []
    time_per_km_sp = []
    time_per_km_no_sp = []
    for round in rounds:
        time_sps.append(round.time_sp)
        time_no_sps.append(round.time_no_sp)
        distance_sps.append(round.distance_sp)
        distance_no_sps.append(round.distance_no_sp)
        time_per_km_sp.append(round.time_per_km_sp)
        time_per_km_no_sp.append(round.time_per_km_no_sp)

    model_name = round.model.split('/')[2]
    model_ = model_name.split('_')[0]

    last_ten_sp = time_sps[-10:]
    last_ten_no_sp = time_no_sps[-10:]

    print("-----------------------------")
    print(model_name)
    print("Last 10 Average for Special: ", np.mean(last_ten_sp))
    print("Last 10 Average for Normal: ", np.mean(last_ten_no_sp))
    
    print("Overall Min Time for Special: ", np.min(time_sps))
    print("Round: ", np.argmin(time_sps))
    print("Corresponding Time for Common: ", time_no_sps[np.argmin(time_sps)])
    
    print("Last 20 min for special: ", np.min(time_sps[-20:]))
    print("Round: ", np.argmin(time_sps[-20:])+80)
    print("Corresponding Time for Common: ", time_no_sps[np.argmin(time_sps[-20:])+80])
    
    print(np.argsort(time_sps))
    # print("Overall Min for Common: ", np.min(last_ten_no_sp))
    print("-----------------------------")




    #
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time for Special Vehicle")
    # plt.plot(x, time_sps)
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Time_sp.png")
    # plt.show()
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time for Normal Vehicle")
    # plt.plot(x, time_no_sps)
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Time_no_sp.png")
    # plt.show()
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Distance for Special Vehicle")
    # plt.plot(x, distance_sps)
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Distance_sp.png")
    # plt.show()
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Distance for Normal Vehicle")
    # plt.plot(x, distance_no_sps)
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Distance_no_sp.png")
    # plt.show()
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilometer for Special Vechile")
    # plt.plot(x, time_per_km_sp)
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Time_per_km_sp.png")
    # plt.show()
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilogram for Special Vehicle (Smooth)")
    # plt.plot(np_move_avg(time_per_km_sp, 5))
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Time_per_km_sp_smooth.png")
    # plt.show()
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilometer for Normal Vechile")
    # plt.plot(x, time_per_km_no_sp)
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Time_per_km_no_sp.png")
    # plt.show()
    #
    # plt.title(model_name)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilogram for Normal Vehicle (Smooth)")
    # plt.plot(np_move_avg(time_per_km_no_sp, 5))
    # plt.savefig("./result/" + model_ + "/" + model_name + "/" + model_name + "_Time_per_km_no_sp_smooth.png")
    # plt.show()


def compare_model(dataset, round_number):
    rounds_MylightWithGAT = []
    rounds_MylightWithoutGAT = []
    rounds_Colight = []
    rounds_Lit = []
    rounds_SimpleDQNOne = []

    model = "./data/Mylight+GAT_" + dataset + "/"
    for i in range(round_number):
        print("Round ", i)
        filename = model + 'round_' + str(i) + "/all_vehicle.csv"
        round = process_a_file(filename)
        rounds_MylightWithGAT.append(round)

    model = "./data/Mylight-GAT_" + dataset + "/"
    for i in range(round_number):
        print("Round ", i)
        filename = model + 'round_' + str(i) + "/all_vehicle.csv"
        round = process_a_file(filename)
        rounds_MylightWithoutGAT.append(round)

    model = "./data/Colight_" + dataset + "/"
    for i in range(round_number):
        filename = model + 'round_' + str(i) + "/all_vehicle.csv"
        round = process_a_file(filename)
        rounds_Colight.append(round)

    # model = "./data/Lit_" + dataset + "/"
    # for i in range(round_number):
    #     filename = model + 'round_' + str(i) + "/all_vehicle.csv"
    #     round = process_a_file(filename)
    #     rounds_Lit.append(round)

    model = "./data/SimpleDQNOne_" + dataset + "/"
    for i in range(round_number):
        filename = model + 'round_' + str(i) + "/all_vehicle.csv"
        round = process_a_file(filename)
        rounds_SimpleDQNOne.append(round)


    x = np.arange(0, round_number)

    time_sps_MylightWithGAT = []
    time_no_sps_MylightWithGAT = []
    distance_sps_MylightWithGAT = []
    distance_no_sps_MylightWithGAT = []
    time_per_km_sps_MylightWithGAT = []
    time_per_km_no_sps_MylightWithGAT = []
    for round in rounds_MylightWithGAT:
        time_sps_MylightWithGAT.append(round.time_sp)
        time_no_sps_MylightWithGAT.append(round.time_no_sp)
        distance_sps_MylightWithGAT.append(round.distance_sp)
        distance_no_sps_MylightWithGAT.append(round.distance_no_sp)
        time_per_km_sps_MylightWithGAT.append(round.time_per_km_sp)
        time_per_km_no_sps_MylightWithGAT.append(round.time_per_km_no_sp)

    time_sps_MylightWithoutGAT = []
    time_no_sps_MylightWithoutGAT = []
    distance_sps_MylightWithoutGAT = []
    distance_no_sps_MylightWithoutGAT = []
    time_per_km_sps_MylightWithoutGAT = []
    time_per_km_no_sps_MylightWithoutGAT = []
    for round in rounds_MylightWithoutGAT:
        time_sps_MylightWithoutGAT.append(round.time_sp)
        time_no_sps_MylightWithoutGAT.append(round.time_no_sp)
        distance_sps_MylightWithoutGAT.append(round.distance_sp)
        distance_no_sps_MylightWithoutGAT.append(round.distance_no_sp)
        time_per_km_sps_MylightWithoutGAT.append(round.time_per_km_sp)
        time_per_km_no_sps_MylightWithoutGAT.append(round.time_per_km_no_sp)

    time_sps_Colight = []
    time_no_sps_Colight = []
    distance_sps_Colight = []
    distance_no_sps_Colight = []
    time_per_km_sps_Colight = []
    time_per_km_no_sps_Colight = []
    for round in rounds_Colight:
        time_sps_Colight.append(round.time_sp)
        time_no_sps_Colight.append(round.time_no_sp)
        distance_sps_Colight.append(round.distance_sp)
        distance_no_sps_Colight.append(round.distance_no_sp)
        time_per_km_sps_Colight.append(round.time_per_km_sp)
        time_per_km_no_sps_Colight.append(round.time_per_km_no_sp)

    # time_sps_Lit = []
    # time_no_sps_Lit = []
    # distance_sps_Lit = []
    # distance_no_sps_Lit = []
    # time_per_km_sps_Lit = []
    # for round in rounds_Lit:
    #     time_sps_Lit.append(round.time_sp)
    #     time_no_sps_Lit.append(round.time_no_sp)
    #     distance_sps_Lit.append(round.distance_sp)
    #     distance_no_sps_Lit.append(round.distance_no_sp)
    #     time_per_km_sps_Lit.append(round.time_per_km_sp)

    time_sps_SimpleDQNOne = []
    time_no_sps_SimpleDQNOne = []
    distance_sps_SimpleDQNOne = []
    distance_no_sps_SimpleDQNOne = []
    time_per_km_sps_SimpleDQNOne = []
    time_per_km_no_sps_SimpleDQNOne = []
    for round in rounds_SimpleDQNOne:
        time_sps_SimpleDQNOne.append(round.time_sp)
        time_no_sps_SimpleDQNOne.append(round.time_no_sp)
        distance_sps_SimpleDQNOne.append(round.distance_sp)
        distance_no_sps_SimpleDQNOne.append(round.distance_no_sp)
        time_per_km_sps_SimpleDQNOne.append(round.time_per_km_sp)
        time_per_km_no_sps_SimpleDQNOne.append(round.time_per_km_no_sp)
    #
    plt.title(dataset)
    plt.xlabel("Round")
    plt.ylabel("Average Travel Time for Special Vehicle")
    plt.plot(x, time_sps_MylightWithGAT, color="r", label="MylightWithGAT")
    plt.plot(x, time_sps_MylightWithoutGAT, color="b", linestyle="--", label="MylightWithoutGAT")
    plt.plot(x, time_sps_Colight, color="g", linestyle="--", label="Colight")
    # plt.plot(x, time_sps_Lit, color="k", linestyle="--", label="Lit")
    plt.plot(x, time_sps_SimpleDQNOne, color="y", linestyle="--", label="SimpleDQNOne")
    plt.legend(loc="best", ncol=2)
    plt.savefig("./result/" + dataset + "/" + dataset + "_Time_sp.png")
    plt.show()
    #
    plt.title(dataset)
    plt.xlabel("Round")
    plt.ylabel("Average Travel Time for Normal Vehicle")
    plt.plot(x, time_no_sps_MylightWithGAT, color="r", label="MylightWithGAT")
    plt.plot(x, time_no_sps_MylightWithoutGAT, color="b", linestyle="--", label="MylightWithoutGAT")
    plt.plot(x, time_no_sps_Colight, color="g", linestyle="--", label="Colight")
    # plt.plot(x, time_no_sps_Lit, color="k", linestyle="--", label="Lit")
    plt.plot(x, time_no_sps_SimpleDQNOne, color="y", linestyle="--", label="SimpleDQNOne")
    plt.legend(loc="best", ncol=2)
    plt.savefig("./result/" + dataset +  "/" + dataset + "_Time_no_sp.png")
    plt.show()
    #
    # plt.title(dataset)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Distance for Special Vehicle")
    # plt.plot(x, distance_sps_MylightWithGAT, color="r", label="MylightWithGAT")
    # plt.plot(x, distance_sps_MylightWithoutGAT, color="b", linestyle="--", label="MylightWithoutGAT")
    # plt.plot(x, distance_sps_Colight, color="g", linestyle="--", label="Colight")
    # # plt.plot(np.arange(0, 30), distance_sps_Lit, color="k", linestyle="--", label="Lit")
    # plt.plot(x, distance_sps_SimpleDQNOne, color="y", linestyle="--", label="SimpleDQNOne")
    # plt.legend(loc="best", ncol=2)
    # plt.savefig("./result/" + dataset + "/" + dataset + "_Distance_sp.png")
    # plt.show()
    #
    # plt.title(dataset)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Distance for Normal Vehicle")
    # plt.plot(x, distance_no_sps_MylightWithGAT, color="r", label="MylightWithGAT")
    # plt.plot(x, distance_no_sps_MylightWithoutGAT, color="b", linestyle="--", label="MylightWithoutGAT")
    # plt.plot(x, distance_no_sps_Colight, color="g", linestyle="--", label="Colight")
    # # plt.plot(np.arange(0, 30), distance_no_sps_Lit, color="k", linestyle="--", label="Lit")
    # plt.plot(x, distance_no_sps_SimpleDQNOne, color="y", linestyle="--", label="SimpleDQNOne")
    # plt.legend(loc="best", ncol=2)
    # plt.savefig("./result/" + dataset + "/" + dataset + "_Distance_no_sp.png")
    # plt.show()

    # plt.title(dataset)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilogram for Special Vehicle")
    # plt.plot(x, time_per_km_sps_MylightWithGAT, color="r", label="MylightWithGAT")
    # plt.plot(x, time_per_km_sps_MylightWithoutGAT, color="b", linestyle="--", label="MylightWithoutGAT")
    # # plt.plot(x, time_per_km_sps_Colight, color="g", linestyle="--", label="Colight")
    # # plt.plot(np.arange(0, 30), time_per_km_sps_Lit, color="k", linestyle="--", label="Lit")
    # # plt.plot(x, time_per_km_sps_SimpleDQNOne, color="y", linestyle="--", label="SimpleDQNOne")
    # plt.legend(loc="best", ncol=2)
    # # plt.savefig("./result/" + dataset + "/" + dataset + "_Time_per_km_sp.png")
    # plt.show()

    # plt.title(dataset)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilogram for Special Vehicle (Smooth)")
    # plt.plot(np_move_avg(time_per_km_sps_MylightWithGAT, 5), color="r", label="MylightWithGAT")
    # plt.plot(np_move_avg(time_per_km_sps_MylightWithoutGAT, 5), color="b", linestyle="--", label="MylightWithoutGAT")
    # plt.plot(np_move_avg(time_per_km_sps_Colight, 5), color="g", linestyle="--", label="Colight")
    # # plt.plot(np_move_avg(time_per_km_sps_Lit, 5), color="k", linestyle="--", label="Lit")
    # plt.plot(np_move_avg(time_per_km_sps_SimpleDQNOne, 5), color="y", linestyle="--", label="SimpleDQNOne")
    # plt.legend(loc="best", ncol=2)
    # plt.savefig("./result/" + dataset + "/" + dataset + "_Time_per_km_sp_smooth.png")
    # plt.show()
    #
    # plt.title(dataset)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilogram for Normal Vehicle")
    # plt.plot(x, time_per_km_no_sps_MylightWithGAT, color="r", label="MylightWithGAT")
    # plt.plot(x, time_per_km_no_sps_MylightWithoutGAT, color="b", linestyle="--", label="MylightWithoutGAT")
    # plt.plot(x, time_per_km_no_sps_Colight, color="g", linestyle="--", label="Colight")
    # # plt.plot(np.arange(0, 30), time_per_km_sps_Lit, color="k", linestyle="--", label="Lit")
    # plt.plot(x, time_per_km_no_sps_SimpleDQNOne, color="y", linestyle="--", label="SimpleDQNOne")
    # plt.legend(loc="best", ncol=2)
    # plt.savefig("./result/" + dataset + "/" + dataset + "_Time_per_km_no_sp.png")
    # plt.show()
    #
    # plt.title(dataset)
    # plt.xlabel("Round")
    # plt.ylabel("Average Travel Time per Kilogram for Normal Vehicle (Smooth)")
    # plt.plot(np_move_avg(time_per_km_no_sps_MylightWithGAT, 5), color="r", label="MylightWithGAT")
    # plt.plot(np_move_avg(time_per_km_no_sps_MylightWithoutGAT, 5), color="b", linestyle="--", label="MylightWithoutGAT")
    # plt.plot(np_move_avg(time_per_km_no_sps_Colight, 5), color="g", linestyle="--", label="Colight")
    # # plt.plot(np_move_avg(time_per_km_sps_Lit, 5), color="k", linestyle="--", label="Lit")
    # plt.plot(np_move_avg(time_per_km_no_sps_SimpleDQNOne, 5), color="y", linestyle="--", label="SimpleDQNOne")
    # plt.legend(loc="best", ncol=2)
    # plt.savefig("./result/" + dataset + "/" + dataset + "_Time_per_km_no_sp_smooth.png")
    # plt.show()



def np_move_avg(a,n):
    return (np.convolve(a, np.ones((n,))/n, mode="same"))

def process_a_file(filename):
    records = read_file(filename)
    round = calculate(records)
    round.model = filename
    round.show()
    return round

def read_file(filename):
    file = csv.reader(open(filename, 'r'))
    records = []
    next(file)

    for row in file:
        if float(row[8]) != 0:
            flow = Flow(float(row[1]), float(row[2]), int(row[3]), float(row[8]))
            # if flow.time_per_km <= 10000:
            records.append(flow)

    return records

def calculate(records):
    times_no_sp = []
    times_sp = []
    distances_no_sp = []
    distances_sp = []
    # times_per_km_sp = []
    # times_per_km_no_sp = []

    for record in records:
        if record.sp == True:
            times_sp.append(record.time)
            distances_sp.append(record.distance)
            # times_per_km_sp.append(record.time_per_km)
        else:
            times_no_sp.append(record.time)
            distances_no_sp.append(record.distance)
            # times_per_km_no_sp.append(record.time_per_km)

    time_no_sp = np.mean(times_no_sp)
    time_sp = np.mean(times_sp)
    distance_no_sp = np.mean(distances_no_sp)
    distance_sp = np.mean(distances_sp)

    time_per_km_sp = time_sp / (distance_sp / 1000.0)
    time_per_km_no_sp = time_no_sp / (distance_no_sp / 1000.0)


    return Round(time_no_sp, time_sp, distance_no_sp, distance_sp, time_per_km_sp, time_per_km_no_sp)


# process_a_model("./data/Mylight+GAT_Jinan_4/", 100)
# process_a_model("./data/Mylight-GAT_Jinan_4/", 100)
# process_a_model("./data/Colight_Jinan_4/", 100)
# process_a_model("./data/Lit_Jinan_4/", 100)
# process_a_model("./data/SimpleDQNOne_Jinan_4/", 100)

# compare_model("Jinan_4", 100)

# process_a_model("./data/Mylight+GAT_Hefei_4/", 100)
# process_a_model("./data/Mylight-GAT_Hefei_4/", 100)
# process_a_model("./data/Colight_Hefei_4/", 100)
# process_a_model("./data/Lit_Hefei_4/", 100)
# process_a_model("./data/SimpleDQNOne_Hefei_4/", 100)
# compare_model("Hefei_4", 100)

# process_a_model("./data/Mylight+GAT_Hangzhou_4/", 100)
# process_a_model("./data/Mylight-GAT_Hangzhou_4/", 100)
# process_a_model("./data/Colight_Hangzhou_4/", 100)
# process_a_model("./data/Lit_Hangzhou_4/", 100)
# process_a_model("./data/SimpleDQNOne_Hangzhou_4/", 100)
# compare_model("Hangzhou_4", 100)


# process_a_model("./data/Mylight+GAT_Newyork_4/", 100)
# process_a_model("./data/Mylight-GAT_Newyork_4/", 100)
# process_a_model("./data/Colight_Newyork_4/", 100)
# process_a_model("./data/Lit_Newyork_4/", 100)
# process_a_model("./data/SimpleDQNOne_Newyork_4/", 100)
#
# process_a_model("./records/0329_SimpleDQNOne_Hefei_40per/anon_32_19_hefei_40_per.json_03_29_13_33_53/test_round/", 100)
# process_a_file("all_vehicle.csv")

# process_a_model("./records/0827_CoLight_jinan/anon_3_4_jinan_real_2500.json_08_27_03_47_36/test_round/", 100)
# process_a_model("./records/0827_CoLight_hangzhou/anon_4_4_hangzhou_real_5816.json_08_27_03_48_07/test_round/", 100)
# process_a_model("./records/0827_CoLight_hefei/anon_3_3_hefei_20per.json_08_27_03_48_58/test_round/", 100)
# process_a_model("./records/0827_CoLight_6_6/anon_6_6_300_0.3_bi.json_08_27_03_45_40/test_round/", 100)
# process_a_model("./records/0827_CoLight_newyork/anon_28_7_newyork_real_triple.json_08_27_03_47_00/test_round/", 100)

# process_a_model("./records/0827_MyLight+GAT_jinan/anon_3_4_jinan_real_2500.json_08_27_12_57_02/test_round/", 100)
# process_a_model("./records/0827_MyLight+GAT_hangzhou/anon_4_4_hangzhou_real_5816.json_08_27_12_59_24/test_round/", 100)
# process_a_model("./records/0827_MyLight+GAT_6_6/anon_6_6_300_0.3_bi.json_08_27_13_00_47/test_round/", 100)

# process_a_model("./records/0827_MyLight-GAT_jinan/anon_3_4_jinan_real_2500.json_08_27_12_58_03/test_round/", 100)
# process_a_model("./records/0827_MyLight-GAT_hangzhou/anon_4_4_hangzhou_real_5816.json_08_27_12_58_50/test_round/", 100)
# process_a_model("./records/0827_MyLight-GAT_hefei/anon_3_3_hefei_20per.json_08_27_13_03_56/test_round/", 100)
# process_a_model("./records/0827_MyLight-GAT_6_6/anon_6_6_300_0.3_bi.json_08_27_13_01_17/test_round/", 100)
# process_a_model("./records/0827_MyLight-GAT_newyork/anon_28_7_newyork_real_triple.json_08_27_13_02_11/test_round/", 100)

# process_a_model("./records/0828_SimpleDQNOne_jinan/anon_3_4_jinan_real_2500.json_08_28_07_28_25/test_round/", 100)
# process_a_model("./records/0828_SimpleDQNOne_hangzhou/anon_4_4_hangzhou_real_5816.json_08_28_07_29_01/test_round/", 100)
# process_a_model("./records/0828_SimpleDQNOne_hefei/anon_3_3_hefei_20per.json_08_28_07_26_53/test_round/", 100)
# process_a_model("./records/0830_SimpleDQNOne_hefei/anon_3_3_hefei_20per.json_08_30_06_39_36/test_round/", 100)
# process_a_model("./records/0828_SimpleDQNOne_6_6/anon_6_6_300_0.3_bi.json_08_28_07_30_31/test_round/", 100)
# process_a_model("./records/0828_SimpleDQNOne_newyork/anon_28_7_newyork_real_triple.json_08_28_07_29_47/test_round/", 100)


# process_a_model("./records/0829_Lit_jinan/anon_3_4_jinan_real_2500.json_08_29_08_15_09/test_round/", 100)
# process_a_model("./records/0829_Lit_hangzhou/anon_4_4_hangzhou_real_5816.json_08_29_08_16_01/test_round/", 100)
# process_a_model("./records/0829_Lit_hefei/anon_3_3_hefei_20per.json_08_29_08_16_50/test_round/", 100)
# process_a_model("./records/0829_Lit_6_6/anon_6_6_300_0.3_bi.json_08_29_08_12_48/test_round/", 100)


# process_a_model("./records/0830_SimpleDQN_jinan/anon_3_4_jinan_real_2500.json_08_30_02_54_49/test_round/", 100)
# process_a_model("./records/0830_SimpleDQN_hangzhou/anon_4_4_hangzhou_real_5816.json_08_30_02_55_51/test_round/", 100)
# process_a_model("./records/0830_SimpleDQN_6_6/anon_6_6_300_0.3_bi.json_08_30_02_56_57/test_round/", 100)


# process_a_model("./records/0830_SimpleDQN_hefei/anon_3_3_hefei_20per.json_08_30_02_56_26/test_round/", 100)

print("Lit Jinan")
process_a_file("./records/0830_Lit_jinan/anon_3_4_jinan_real_2500.json_08_30_03_18_14/test_round/round_0/all_vehicle.csv")
print("Lit Jinan Route Baseline")
process_a_file("./records/0903_Lit_jinan_route_baseline/anon_3_4_jinan_real_2500.json_09_03_07_05_19/test_round/round_0/all_vehicle.csv")
print("Lit Jinan Route Proposed")
process_a_file("./records/0903_Lit_jinan_route_proposed/anon_3_4_jinan_real_2500.json_09_03_14_06_49/test_round/round_0/all_vehicle.csv")









