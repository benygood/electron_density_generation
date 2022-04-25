r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
from torch._six import container_abcs, string_classes, int_classes
from  torch.utils.data._utils.collate import default_collate
np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_ignore_none(batch):
    '''
         collate_fn (callable, optional): merges a list of samples to form a mini-batch.
         该函数参考touch的default_collate函数，也是DataLoader的默认的校对方法，当batch中含有None等数据时，
         默认的default_collate校队方法会出现错误
         一种的解决方法是：
         判断batch中image是否为None,如果为None，则在原来的batch中清除掉，这样就可以在迭代中避免出错了
        :param batch:
        :return:
        '''
    r"""Puts each data field into a tensor with outer dimension batch size"""

    batch = list(filter(lambda x: (x[0] is not None) and (x[1] is not None) and (x[2] is not None), batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)
