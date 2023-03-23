import torch

file = "/data/ys/mmaction/work_dirs/monkey_BARN/interaction_2class_v2/best_mAP@0.5IOU_epoch_27.pth"
# file = "/data/ys/mmaction/work_dirs/monkey/mix_switch_swBB_group_2class/best_mAP@0.5IOU_epoch_20.pth"
out_path = "/data/ys/mmaction/work_dirs/monkey_BARN/interaction_2class_v2/swBB_after_change_interaction.pth"
#file = out_path  #用于检验保存成功与否

ckt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
out = ckt["state_dict"]

key = [k.replace("backbone", "module.switch_backbone") for k in out ]
key = [k.replace("roi_head.bbox_head", "module.roi_head.switch_head") for k in key ]
oo = key[-10:]
value = [v for k,v in out.items()]
value[-2] = value[-2][1:, :]  # (3,2304) -> (2,2304)
value[-1] = value[-1][1:]
vv = value[-2:]
out = dict(zip(key, value))

torch.save(out, out_path)

print()

