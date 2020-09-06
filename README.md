# ECO-full
ECO-full用百度飞桨框架复现。

python avi2jpg.py  #把视频处理成图片

python jpg2pkl.py   #划分数据集

python data_list_gener.py  #生成文件列表

python train.py --use_gpu True --epoch  1  --pretrain True   #有模型时候的模型训练命令

python train.py --use_gpu True --epoch  1    #如果是头开始时候的模型训练命令

python eval.py --weights 'checkpoints_models/tsn_model' --use_gpu True  #模型评估
