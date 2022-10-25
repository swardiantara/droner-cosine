# 设置随机数种子
from fastNLP import MetricBase

def set_rng_seed(rng_seed:int = None, random:bool = True, numpy:bool = True,
                 pytorch:bool=True, deterministic:bool=True):
    """
    设置模块的随机数种子。由于pytorch还存在cudnn导致的非deterministic的运行，所以一些情况下可能即使seed一样，结果也不一致
        需要在fitlog.commit()或fitlog.set_log_dir()之后运行才会记录该rng_seed到log中
    :param int rng_seed: 将这些模块的随机数设置到多少，默认为随机生成一个。
    :param bool, random: 是否将python自带的random模块的seed设置为rng_seed.
    :param bool, numpy: 是否将numpy的seed设置为rng_seed.
    :param bool, pytorch: 是否将pytorch的seed设置为rng_seed(设置torch.manual_seed和torch.cuda.manual_seed_all).
    :param bool, deterministic: 是否将pytorch的torch.backends.cudnn.deterministic设置为True
    """
    if rng_seed is None:
        import time
        rng_seed = int(time.time()%1000000)
    if random:
        import random
        random.seed(rng_seed)
    if numpy:
        try:
            import numpy
            numpy.random.seed(rng_seed)
        except:
            pass
    if pytorch:
        try:
            import torch
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
        except:
            pass
    return rng_seed


class EvaluateNER(MetricBase):
    def __init__(self, pred=None, target=None):
        super().__init__()
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.total_tp = 0
        self_total_fp = 0
        self.total_fn = 0
        self.total_token = 0
        self.acc_count = 0
        self.total = 0


    def evaluate(self, pred, target):
        print('Preds: ')
        print(pred)
        print('Target: ')
        print(target)
        self.total += target.size(0)
        self.acc_count += target.eq(pred).sum().item()

    def get_metric(self, reset=True): # 在这里定义如何计算metric
        acc = self.acc_count/self.total
        if reset: # 是否清零以便重新计算
            self.acc_count = 0
            self.total = 0
        return {'acc': acc}