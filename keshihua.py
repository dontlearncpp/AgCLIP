import json
from  tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    writer = SummaryWriter('./runs/exp')  # 实例化writer
    log = r"巴拉巴拉.log"

    i = 0

    for line in open('/media/test/run/count/countx/CounTX-main-arg-2/results/log.txt', 'r'):

        # print(line) # 按照每一行分开
        # print(type(line)) # <class 'str'>

        # 把str转化为字典
        dic_line = json.loads(line)
        # print(dic_line)
        # print(type(dic_line))

        # 取 键data的值
        TrainMAE = dic_line.get("Current Train MAE") # 获取
        TrainRMSE = dic_line.get("Current Train RMSE") # 获取
        ValMAE = dic_line.get("Current Val MAE") # 获取
        ValRMS = dic_line.get("Current Val RMSE") # 获取

        i += 1
        writer.add_scalar("Current Train MAE", TrainMAE, i)
        writer.add_scalar("Current Train RMSE", TrainRMSE, i)
        writer.add_scalar("Current Val MAE", ValMAE, i)
        writer.add_scalar("Current Val RMSE", ValRMS, i)

    writer.close()



