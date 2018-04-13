#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time, model, pickle
import logging, pdb


def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                            axis=1, is_random=True) # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, x2, x3, x4, y

def vis_imgs(X, y, path):
    """ show one slice """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y]), size=(1, 5),
        image_path=path)

def vis_imgs2(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:,:,np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:,:,np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:,:,0,np.newaxis],
        X[:,:,1,np.newaxis], X[:,:,2,np.newaxis],
        X[:,:,3,np.newaxis], y_, y]), size=(1, 6),
        image_path=path)

# def reshape_dataset(dataset):
#     dataset_reshape = []
#     set = []
#     for idx, data in enumerate(dataset):
#         if idx > 0 and idx % 155 == 0:
#             dataset_reshape.append(set)
#             set = []
#         else:
#             set.append(data)
#     dataset_reshape = np.asarray(dataset_reshape, dtype=np.float32)
#     return dataset_reshape

def main(task='all', data_size='half'):
    ## Create folder to save trained model and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("model.log")
    formatter = logging.Formatter("%(levelname)s-%(message)s")
    fh.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(console)
    tl.files.exists_or_mkdir("samples/{}".format(task))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    ###======================== LOAD DATA ===================================###
    ## by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    if not os.path.exists("./data/train_dev_all/" + "dataset_{}.npz".format(data_size)):
        raise ValueError("can't find preprocess data, please execute prepare_data_with_valid.py first")
    data_path = "./data/train_dev_all/" + "dataset_{}.npz".format(data_size)
    npzfile = np.load(data_path)
    X_train = np.reshape(npzfile['arr_0'], (-1, 5, 240, 240, 4))
    y_train = np.reshape(npzfile['arr_1'], (-1, 5, 240, 240, 1))
    X_test =  np.reshape(npzfile['arr_2'], (-1, 5, 240, 240, 4))
    y_test =  np.reshape(npzfile['arr_3'], (-1, 5, 240, 240, 1))

    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
    else:
        exit("Unknow task %s" % task)

    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 3
    #steps = 5
    lr = 0.0001
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 100
    print_freq_step = 200
    ###======================== SHOW DATA ===================================###
    ###show one slice
    nw, nh, nz = 240, 240, 4
    nl = 5#steps for lstm
    X = np.asarray(X_train[0])
    y = np.asarray(y_train[0])
    # print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    # print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    vis_imgs(X[0], y[0], 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(10):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[0,:,:,0,np.newaxis], X[0,:,:,1,np.newaxis],
                X[0,:,:,2,np.newaxis], X[0,:,:,3,np.newaxis], y[0]])#[:,:,np.newaxis]])
        # print(x_flair.shape, x_t1.shape, x_t1ce.shape, x_t2.shape, label.shape) # (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        # print(X_dis.shape, X_dis.min(), X_dis.max()) # (240, 240, 4) -0.380588233471 2.62376139209
        vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))

    with tf.device('/gpu:0'): #<- remove it if you train on CPU or other GPU
        ###======================== DEFIINE MODEL =======================###
        ## nz is 4 as we input all Flair, T1, T1c and T2.
        sess = tf.Session(config=run_config)
        t_image = tf.placeholder('float32', [batch_size, nl, nw, nh, nz], name='input_image')
        ## labels are either 0 or 1
        t_seg = tf.placeholder('float32', [batch_size, nl, nw, nh, 1], name='target_segment')
        net = model.cu_net(t_image, is_train=True, reuse=False)
        net_test = model.cu_net(t_image, is_train=False, reuse=True)

        out_seg = net.outputs
        dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3,4])#, 'jaccard', epsilon=1e-5)
        iou_loss = tl.cost.iou_coe(out_seg, t_seg, axis=[0,1,2,3,4])
        dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg, axis=[0,1,2,3,4])
        loss = dice_loss
        ## test losses
        test_out_seg = net_test.outputs
        test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3,4])#, 'jaccard', epsilon=1e-5)
        test_iou_loss = tl.cost.iou_coe(test_out_seg, t_seg, axis=[0,1,2,3,4])
        test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg, axis=[0,1,2,3,4])

        t_vars = tl.layers.get_variables_with_name('cu_net', True, True)
        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)

        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        ## load existing model if possible
        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/cu_net_{}.npz'.format(task), network=net)
        ###======================== TRAINING ================================###
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        ## update decay learning rate at the beginning of a epoch
        # if epoch !=0 and (epoch % decay_every == 0):
        #     new_lr_decay = lr_decay ** (epoch // decay_every)
        #     sess.run(tf.assign(lr_v, lr * new_lr_decay))
        #     log = " ** new learning rate: %f" % (lr * new_lr_decay)
        #     print(log)
        # elif epoch == 0:
        #     sess.run(tf.assign(lr_v, lr))
        #     log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
        #     print(log)

        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                    batch_size=batch_size, shuffle=True):
            images, labels = batch
            step_time = time.time()
            ## data augumentation for a batch of Flair, T1, T1c, T2 images
            # and label maps synchronously.
            slice_images = []
            slice_labels = []
            for i in range(nl):
                data = tl.prepro.threading_data([_ for _ in zip(images[:,i,:,:,0, np.newaxis],
                        images[:,i,:,:,1, np.newaxis], images[:,i,:,:,2, np.newaxis],
                        images[:,i,:,:,3, np.newaxis], labels[:,i,:,:,:])],
                        fn=distort_imgs)
                slice_images.append(data[:,0:4,:,:,:])
                slice_labels.append(data[:,4,:,:,:])
            b_images = np.concatenate(slice_images, axis=4)
            b_labels = np.concatenate(slice_labels, axis=3)
            b_images = b_images.transpose((0,4,2,3,1))
            b_labels = b_labels.transpose((0,3,1,2))
            b_labels = b_labels[:,:,:,:,np.newaxis]
            # for i in range(nl):
            #     vis_imgs(b_images[0][i], b_labels[0][i], 'samples/{}/_train_im_{}.png'.format(task, i))
            #update network

            _, _dice, _iou, _diceh, out = sess.run([train_op,
                    dice_loss, iou_loss, dice_hard, net.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1
            ## you can show the predition here:
            # vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_tmp.png".format(task))
            # exit()

            # if _dice == 1: # DEBUG
            #     print("DEBUG")
            #     vis_imgs2(b_images[0], b_labels[0], out[0], "samples/{}/_debug.png".format(task))
            if n_batch % print_freq_step == 0:
                logger.info("Epoch %d step %d 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)"
                % (epoch, n_batch, _dice, _diceh, _iou, time.time()-step_time))

            ## check model fail
            if np.isnan(_dice):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")

        logger.info(" ** Epoch [%d/%d] train 1-dice: %f hard-dice: %f iou: %f took %fs (2d with distortion)" %
                (epoch, n_epoch, total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch, time.time()-epoch_time))

        ## save a predition of training set
        for i in range(batch_size):
            if np.max(b_images[i][2]) > 0:
                vis_imgs2(b_images[i][2], b_labels[i][2], out[i][2], "samples/{}/train_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i][2], b_labels[i][2], out[i][2], "samples/{}/train_{}.png".format(task, epoch))

        ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                        batch_size=batch_size, shuffle=True):
            b_images, b_labels = batch
            _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                    test_iou_loss, test_dice_hard, net_test.outputs],
                    {t_image: b_images, t_seg: b_labels})
            total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
            n_batch += 1

        logger.info(" **"+" "*17+"test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
                (total_dice/n_batch, total_dice_hard/n_batch, total_iou/n_batch))
        logger.info(" task: {}".format(task))
        ## save a predition of test set
        for i in range(batch_size):
            if np.max(b_images[i][2]) > 0:
                vis_imgs2(b_images[i][2], b_labels[i][2], out[i][2], "samples/{}/test_{}.png".format(task, epoch))
                break
            elif i == batch_size-1:
                vis_imgs2(b_images[i][2], b_labels[i][2], out[i][2], "samples/{}/test_{}.png".format(task, epoch))

        ###======================== SAVE MODEL ==========================###
        tl.files.save_npz(net.all_params, name=save_dir+'/cu_net_{}.npz'.format(task), sess=sess)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='all', help='all, necrotic, edema, enhance')
    parser.add_argument('--data_size', type=str, default='half', help='small, half, all')
    args = parser.parse_args()

    main(args.task, args.data_size)
