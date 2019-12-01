# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import sys
import torchreid


def main():
    save_dir = 'log/best_cluster_MNNPL'
    log_name = 'log.txt'
    import os.path as osp
    sys.stdout = torchreid.utils.Logger(osp.join(save_dir, log_name))

    loss = 'triplet'
    # 2.Load data manager
    my_dataloader = torchreid.data.datamanager.MyDataManager(root='C:/Users/Server/Desktop/code/dataset/',
                                                             height=384, width=128,
                                                             # norm_mean=None, norm_std=None, use_gpu=True,
                                                             norm_mean=[0.09721232, 0.18305508, 0.21273703],
                                                             norm_std=[0.17512791, 0.16554857, 0.22157137],
                                                             use_gpu=True,
                                                             batch_size_train=64, batch_size_test=1024,
                                                             workers=4, num_instances=4,
                                                             loss=loss,
                                                             train_sampler='RandomIdentitySampler',
                                                             transforms=['random_flip', 'random_crop', 'random_erase'],
                                                             )
    # print(my_dataloader.num_train_pids)
    # 3. Build model, optimizer and lr_scheduler
    model = torchreid.models.build_model(
        name='se_resnext101_32x4d',
        num_classes=my_dataloader.num_train_pids,
        loss=loss,
        pretrained=True
    )

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.00035
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='multi_step',
        stepsize=[30, 120, 151]
    )

    # load pre-trained model
    fpath = "C:/Users/Server/Desktop/code/dataset/model.pth.tar-150"
    torchreid.utils.torchtools.resume_from_checkpoint(fpath, model, optimizer)
    torchreid.utils.torchtools.load_pretrained_weights(model, fpath)

    model = model.cuda()
    import torch
    gpu_ids = [0, 1]
    torch.cuda.set_device(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()

    # 4.Build engine
    engine = torchreid.engine.ImageTripletEngine(
        my_dataloader,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        use_center=True,
        use_oim=False,
        num_classes=my_dataloader.num_train_pids,
        margin=0.7,
        weight_o=1.5,
        center_lr=0.5,
        weight_c=0.0005,
        oim_dims=2048,
        center_dims=2048,
        use_ranked_loss=True,
        weight_r=2,
        weight_t=1,
        use_focal=False,
        weight_f=10
    )
    # 5.Run training and test
    print(save_dir)

    import numpy as np
    trainloader, valloader, testloader, testdataset = my_dataloader.trainloader, my_dataloader.valloader, my_dataloader.testloader, my_dataloader.testdataset
    queryloader = testloader['query']
    galleryloader = testloader['gallery']
    distmat, gf = engine._evaluate(150,
                               queryloader=queryloader,
                               galleryloader=galleryloader,
                               testdataset=testdataset,
                               dist_metric='cosine',  # euclidean
                               normalize_feature=True,
                               save_dir=save_dir,
                               use_metric_cuhk03=False,
                               ranks=[1, 5, 10, 20],
                               rerank=False,
                               multi_scale_interpolate_mode='bilinear',
                               multi_scale=(1,),
                               return_json=True)
    def get_similarity(tf):
        feature = torch.tensor(tf).cuda()

        score = torch.mm(feature, feature.t()).detach().cpu().numpy()
        indexs = np.argsort(-score, axis=1)
        return indexs

    indexs = get_similarity(gf)
    print(indexs.shape)

    class union_find:
        def __init__(self, length):
            self.length = length
            self.ids = np.arange(length)

        def union(self, i, j):
            id_i = self.ids[i]
            id_j = self.ids[j]
            for i in range(self.length):
                if self.ids[i] == id_i:
                    self.ids[i] = id_j

        def get_set(self):
            keys = []
            result = {}
            for i in range(self.length):
                value = self.ids[i]
                if value in keys:
                    result[value].append(i)
                else:
                    keys.append(value)
                    result[value] = [i]
            return result

    k_ne = 6
    def connect_with_mutual(indexs):
        u = union_find(indexs.shape[0])
        for i in range(indexs.shape[0]):
            for k in indexs[i][:k_ne]:
                if i in indexs[k][:k_ne]:
                    u.union(i, k)
            if i % 1000 == 0:
                print(i)
        return u.get_set()

    connected = connect_with_mutual(indexs)

    reliable_keys = []
    for key in connected.keys():
        if len(connected[key]) >= 5 and len(connected[key]) <= 32:
            reliable_keys.append(key)
    import pandas as pd
    from glob import glob
    from os import path as osp
    train_image_path = "C:/Users/Server/Desktop/code/dataset/初赛A榜测试集/初赛A榜测试集/gallery_a"
    fpaths = sorted(glob(osp.join(train_image_path, '*.png')))
    df = pd.DataFrame()
    import csv
    # csv 写入
    csv_name = 'result.csv'
    out = open(csv_name, 'a', newline='')
    # 设定写入模式
    csv_write = csv.writer(out, dialect='excel')
    for i, key in enumerate(reliable_keys):
        csv_write.writerow(np.array(fpaths)[connected[key]])
    print("write over")
    with open(csv_name, "rb") as f, open("./" + csv_name,'wb') as fw:
        fw.write(f.read())

if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()