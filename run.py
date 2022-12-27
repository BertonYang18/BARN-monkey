import torch
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# inference the demo video
top5_label = inference_recognizer(model, 'demo/demo.mp4')
print(top5_label)



# ####使用@语法糖
# def makeitalic(fun):
#     def wrapped():
#         return "<i>" + fun() + "</i>"
#     return wrapped

# @makeitalic#使用了装饰器可以直接调用，不需要赋值了
# def hello():
#     return "hello world"
# print(hello())#使用了装饰器可以直接调用，不需要赋值了