# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import sys
import torchreid
import copy


def main():
    save_dir = 'log_jicheng6'
    log_name = 'log.txt'
    sys.stdout = torchreid.utils.Logger(osp.join(save_dir, log_name))

    loss = 'triplet'
    # 2.Load data manager
    my_dataloader = torchreid.data.datamanager.MyDataManager(root='/project/data',
                                                             height=384, width=128,
                                                             norm_mean=[0.09721232, 0.18305508, 0.21273703],
                                                             norm_std=[0.17512791, 0.16554857, 0.22157137],
                                                             use_gpu=True,
                                                             batch_size_train=64, batch_size_test=512,
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

    model = model.cuda()
    print(model)

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.000035
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='multi_step',
        stepsize=[30, 120, 151]
    )


    #-------------load-------------------
    #'/project/CODE/AIC/code/log/best/' 
    model_1 = copy.deepcopy(model)
    model_2 = copy.deepcopy(model)
    model_3 = copy.deepcopy(model)
    model_4 = copy.deepcopy(model)
    model_5 = copy.deepcopy(model) 
    model_6 = copy.deepcopy(model) 
      
    fpath_1 = "/project/CODE/code3_1/log/best1/model.pth.tar-150"
    torchreid.utils.torchtools.load_pretrained_weights(model_1, fpath_1)

    fpath_2 = "/project/CODE/code3_2/log/best2/model.pth.tar-150"
    torchreid.utils.torchtools.load_pretrained_weights(model_2, fpath_2)
    
    fpath_3 = "/project/CODE/code3_3/log/best3/model.pth.tar-150"
    torchreid.utils.torchtools.load_pretrained_weights(model_3, fpath_3)

    fpath_4 = "/project/CODE/code3_4/log/best4/model.pth.tar-150"
    torchreid.utils.torchtools.load_pretrained_weights(model_4, fpath_4)

    fpath_5 = "/project/CODE/code3_5/log/best5/model.pth.tar-150"
    torchreid.utils.torchtools.load_pretrained_weights(model_5, fpath_5)
    
    fpath_6 = "/project/CODE/code3_6/log/best6/model.pth.tar-150"
    torchreid.utils.torchtools.load_pretrained_weights(model_6, fpath_6)


    # 4.Build engine
    engine = torchreid.engine.ImageTripletEngine(
        my_dataloader,
        model_1,
        model_2,
        model_3,
        model_4,
        model_5,
        model_6,
        optimizer=optimizer,
        scheduler=scheduler,
        use_center=False,
        num_classes=my_dataloader.num_train_pids,
    )
    # 5.Run training and test
    engine.run(
        save_dir=save_dir,
        normalize_feature=True,
        max_epoch=150,
        eval_freq=10,
        rerank=False,
        print_freq=100,
        test_only=True,
        label_smooth_end_epoch=120
    )



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
