""" Training script

This module creates and trains recurrent neural network language model.

Example:

Todo:
    - Support other optimization method
    - Support TensorBoard

"""

import argparse
import logging
import time
import os
import json
import random
# random.seed(1234)

import cPickle
import numpy as np
# np.random.seed(1234)
import tensorflow as tf
# tf.set_random_seed(1234)

import lm
import common_utils
import data_utils
from exp_utils import *

def main(lm_opt):
    model_prefix = ['latest_lm']
    dataset = ['train', 'valid']
    lm_data, lm_vocab = load_datasets(lm_opt, dataset=dataset)
    lm_opt.vocab_size = lm_vocab.vocab_size
    logger.info('Loading data completed')
    init_scale = lm_opt.init_scale
    sess_config = common_utils.get_tf_sess_config(lm_opt)
    logger.info('Starting TF Session...')

    with tf.Session(config=sess_config) as sess:
        logger.debug(
            '- Creating initializer ({} to {})'.format(-init_scale, init_scale))
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        logger.debug('- Creating training LM...')
        with tf.variable_scope('LM', reuse=None, initializer=initializer):
            lm_train = lm.LM(lm_opt)

            if(lm_opt.special_train):
                logger.info("Special train")
                lm_train_op, lm_lr_var = lm.train_op_mod(lm_train, lm_opt)

            elif(lm_opt.freeze_model):
                logger.info("Trainining completely frozen model - TEST")
                lm_train_op, lm_lr_var = lm.train_op_frozen(lm_train, lm_opt)

            else:
                lm_train_op, lm_lr_var = lm.train_op(lm_train, lm_opt)

        logger.debug('- Creating validating LM (reuse params)...')
        with tf.variable_scope('LM', reuse=True, initializer=initializer):
            lm_valid = lm.LM(lm_opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing vairables...')
        sess.run(tf.global_variables_initializer())

        if(lm_opt.special_train):
            for v in tf.trainable_variables():
                if "emb" in v.name:
                    sess.run(tf.assign(v, lm_opt.initialization))
                    lm_opt._embedding_var = v
                    # print sess.run(v)[:, 100] - opt.parameter_masks["LM/emb:0"][:, 100]
                    # v.assign(tf.multiply(v, freeze_tensor))
                    # parameter_tensor = tf.constant(lm_opt.paramter_masks["LM/emb_0:0"])
                    # parameter_tensor = tf.cast(parameter_tensor, tf.float32)
                    # v.assign(tf.add(v, parameter_tensor))
                    # sess.run(v)
                    # exit()
                    break

        if(lm_opt.freeze_model):
            for v in tf.trainable_variables():
                if "emb" in v.name:
                    sess.run(tf.assign(v, lm_opt._emb))
                    lm_opt._emb_var = v
                elif "basic_lstm_cell/weights" in v.name:
                    sess.run(tf.assign(v, lm_opt._lstm_w))
                    lm_opt._lstm_w_var = v
                elif "basic_lstm_cell/biases" in v.name:
                    sess.run(tf.assign(v, lm_opt._lstm_w))
                    lm_opt._lstm_b_var = v
                elif "softmax_w" in v.name:
                    sess.run(tf.assign(v, lm_opt._softmax_w))
                    lm_opt._softmax_w_var = v
                elif "softmax_b" in v.name:
                    sess.run(tf.assign(v, lm_opt._softmax_b))
                    lm_opt._softmax_b_var = v

        saver = tf.train.Saver()
        states = {}
        for p in model_prefix:
            states[p] = common_utils.get_initial_training_state()
        states, _ = resume_many_states(lm_opt.output_dir, sess,
                                       saver, states, model_prefix)
        lm_state = states[model_prefix[0]]
        lm_state.learning_rate = lm_opt.learning_rate

        logger.info('Start training loop:')
        logger.debug('\n' + common_utils.SUN_BRO())

        for epoch in range(lm_state.epoch, lm_opt.max_epochs):
            epoch_time = time.time()
            logger.info("========= Start epoch {} =========".format(epoch+1))
            sess.run(tf.assign(lm_lr_var, lm_state.learning_rate))
            logger.info("- Traning LM with learning rate {}...".format(
                lm_state.learning_rate))
            lm_train_ppl, _ = run_epoch(sess, lm_train, lm_data['train'],
                                        lm_opt, train_op=lm_train_op)
            logger.info('- Validating LM...')
            lm_valid_ppl, _ = run_epoch(sess, lm_valid,
                                        lm_data['valid'], lm_opt)
            logger.info('----------------------------------')
            logger.info('LM post epoch routine...')

            if (lm_opt.special_train):
                shared_indexes = cPickle.load(open("models/r1.0/gen_m1/index_m1_m2_t1_46.6641693115_t2_6.87459030151.pickle", "r"))
                for index in shared_indexes:
                    if(np.array_equal(sess.run(lm_opt._embedding_var)[:, index], opt.parameter_masks["LM/emb_0:0"][:, index]) != True):
                        logger.info("SPECIAL TRAINING - something went horribly wrong")
                        exit()
                logger.info("SPECIAL TRAINING - Successful")

            if (lm_opt.freeze_model):

                if(np.array_equal(sess.run(lm_opt._emb_var), lm_opt._emb) != True):
                    logger.info("MODEL FREEZE - _emb freeze went horribly wrong")
                    exit()

                if(np.array_equal(sess.run(lm_opt._lstm_w_var), lm_opt._lstm_w) != True):
                    logger.info("MODEL FREEZE - _lstm_w freeze went horribly wrong")
                    exit()

                if(np.array_equal(sess.run(lm_opt._lstm_b_var), lm_opt._lstm_b) != True):
                    logger.info("MODEL FREEZE - _lstm_b freeze went horribly wrong")
                    exit()

                if(np.array_equal(sess.run( lm_opt._softmax_w_var), lm_opt._softmax_w) != True):
                    logger.info("MODEL FREEZE - _softmax_w freeze went horribly wrong")
                    exit()

                if(np.array_equal(sess.run( lm_opt._softmax_b_var), lm_opt._softmax_b) != True):
                    logger.info("MODEL FREEZE - _softmax_b freeze went horribly wrong")
                    exit()

                logger.info("MODEL FREEZE - Successful")

            done_training = run_post_epoch(
                lm_train_ppl, lm_valid_ppl, lm_state, lm_opt,
                sess=sess, saver=saver,
                best_prefix="best_lm", latest_prefix="latest_lm")
            logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
            if done_training:
                break
        logger.info('Done training at epoch {}'.format(lm_state.epoch + 1))




if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()

    parser.add_argument('--special_train', action='store_true',
                        help='Trains using masks')

    parser.add_argument('--freeze_model', action='store_true',
                        help='Freeze whole model')

    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)
    logger = common_utils.get_logger(opt.log_file_path)
    opt.logger = logger
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info('Configurations:\n{}'.format(opt.__repr__()))

    if(opt.special_train):
        # print "Hello"
        logger.info('Loading Masks ')
        opt.parameter_masks =  cPickle.load(open("models/r1.0/gen_m1/parameters_m1_m2_t1_46.6641693115_t2_6.87459030151.pickle", "r"))
        opt.freeze_masks = cPickle.load(open("models/r1.0/gen_m1/freeze_m1_m2_t1_46.6641693115_t2_6.87459030151.pickle", "r"))

        temp = np.random.uniform(-.1 , .1, [10000, 300])
        opt.initialization = np.multiply(temp, opt.freeze_masks["LM/emb:0"]) + opt.parameter_masks["LM/emb:0"]

        # print opt.initialization[:, 100] - opt.parameter_masks["LM/emb_0:0"][:, 100]
        # print opt.initialization[:, 46] - opt.parameter_masks["LM/emb_0:0"][:, 46]

        # exit()

        logger.info('Loading Masks completed')
        # exit()

    if(opt.freeze_model):
        opt.parameter_masks = cPickle.load(open("../../data/r1.0/models/m1/params.pickle", "r"))
        opt.freeze_masks = cPickle.load(open("../../data/r1.0/masks/zeros/freeze_m1.pickle", "r"))

        opt._emb = opt.parameter_masks["LM/emb:0"]
        opt._lstm_w = opt.parameter_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights:0"]
        opt._lstm_b = opt.parameter_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases:0"]
        opt._softmax_w = opt.parameter_masks["LM/softmax_w:0"]
        opt._softmax_b = opt.parameter_masks["LM/softmax_b:0"]


    main(opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
