'''pth'''
# import torch
# file = '/data/ys/mmaction/work_dirs/monkey/switch_swBB_group_2class/best_mAP@0.5IOU_epoch_2.pth'
# file2 = '/data/ys/mmaction/work_dirs/monkey/switch_swBB_group_2class/after_change.pth.tar'
# checkpoint = torch.load(file, map_location=lambda storage, loc: storage.cuda())
# print(type(checkpoint))#<class 'dict'>
# print(len(checkpoint))#3
# # 1. filter out unnecessary keys
# pretrained_dict = {k.replace('backbone', 'module.switch_backbone'): v for k, v in checkpoint['state_dict'].items() if 'backbone' in k}
# torch.save(pretrained_dict, file2)
# print()
'''pkl'''
# import pickle

# file = '/data/ys/mmaction/work_dirs/monkey/switch_swBB_group_2class/best_mAP@0.5IOU_epoch_2.pth'
# with open(file, 'rb') as f:
#     for line in f.readlines():
#         pass

'''class'''
# class F1:
#     def __init__(self):
#         self.a = 1
#     def forward(self, x):
#         x = x + self.a
#         return x

# class son(F1):
#     def my_f(self):
#         print(333)

# feat = 0
# now = F1()
# out = now.forward(feat)
# print(out)

'''expand expand_as'''
# import torch
# a = torch.randn([2,1,1])
# b = torch.randn([102,8,8])
# c1 = a.expand(2,8,8)
# H,W = c1.size()[-2:]
# c2 = a.expand(10,8,8) #error--> expand（）函数只能将size=1的维度扩展到更大的尺寸，如果扩展其他size（）的维度会报错。
# c3 = a.expand(11,8,8)
# d1 = a.expand_as(b)
# print(a)
# print(c1)

'''list'''
# a = [1,2,3]
# b = [1,2,4]
# s = [a, b]
# aa, bb = s
# for i in range(1, len(a)):
#     print(i)
# for i in range(len(a)-1, 0, -1):
#     print(i)
# pass

# '''tensor'''
import torch
mm = [2.]*32
m = torch.tensor(mm).resize(2,16)
# m = torch.tensor(mm).reshape(1,14).repeat(2,1)
mm2 = [1.]*32
m2 = torch.tensor(mm2).resize(2,16)
#status = [0.98]

out = m*0.98 + m2*0.02
pass