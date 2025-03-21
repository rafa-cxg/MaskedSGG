import yaml
with open('lavis/projects/sggp/train/finetune.yaml',encoding='utf-8') as file1:
    data = yaml.load(file1,Loader=yaml.FullLoader)#读取yaml文件
    predict_mode=data["model"]["predict_mode"]
    mode = data["model"]["task_mode"] #sgdet or predcls
if predict_mode=="zero-shot":
    from .vg_eval import do_vg_evaluation
else:
    from .vg_eval_2stream import do_vg_evaluation


def vg_evaluation(
    predictions,
    targets,
    iou_types,
):
    return do_vg_evaluation(

        predictions=predictions,
        groundtruths=targets,
        iou_types=iou_types,
        mode = mode
    )
