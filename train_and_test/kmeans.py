# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import sys
import torchreid


def main():
    save_dir = 'log/best_kmeans_cluster'
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
                                                             batch_size_train=64, batch_size_test=1024,  # 64, 128
                                                             workers=4, num_instances=4,
                                                             loss=loss,
                                                             train_sampler='RandomSampler',
                                                             # if triplet, PK sample is needing
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



    import sklearn.cluster as cluster
    import multiprocessing
    def KMeans(feat, n_clusters=2):
        kmeans = cluster.KMeans(n_clusters=n_clusters,
                                n_jobs=multiprocessing.cpu_count(),
                                random_state=0).fit(feat)
        return kmeans.labels_
    pred = KMeans(gf.numpy(), n_clusters=4000)   # 1348

    print(pred)
    import shutil
    output_dir = "./test_b"
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    test_data_path = 'C:/Users/Server/Desktop/code/dataset/初赛B榜测试集/初赛B榜测试集/gallery_b'
    for ind, image_name in enumerate(os.listdir(test_data_path)):
        new_image_name = '{:04d}_{}'.format(pred[ind], image_name)
        shutil.copy(osp.join(test_data_path, image_name), osp.join(output_dir, new_image_name))



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()