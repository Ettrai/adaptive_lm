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

def find_mapping_index(data, sharing):
   return data["similarity"].index(min(data["similarity"], key=lambda x: abs(x - sharing)))

def main(lm_opt):

    logger = lm_opt.logger

    if(lm_opt.special_train):
        gen_from_models = lm_opt.gen_from_models.split(",")
        gen_prefix = gen_from_models[0] + "_" + gen_from_models[1]

        similarity_data = cPickle.load(open("../data/r1.0/sharing/cosine/" + gen_prefix + "_emb_similarity.pickle", "r"))

        #Retrieve the number of shared neurons and the threshold used
        similarity_index = find_mapping_index(similarity_data, lm_opt.num_shared_neurons)
        similarity_neurons = similarity_data["similarity"][similarity_index]
        similarity_threshold = similarity_data["thresholds"][similarity_index]
        similarity_config = gen_prefix + "_n" + str(similarity_neurons) + "_t" + str(similarity_threshold)

        parameter_mask_file = "../data/r1.0/masks/cosine/parameters_" + similarity_config + ".pickle"
        freeze_mask_file = "../data/r1.0/masks/cosine/freeze_" + similarity_config + ".pickle"
        mapping_indexes_file = "../data/r1.0/masks/cosine/mapping_" + similarity_config + ".pickle"

        logger.info('Loading Masks')
        lm_opt.parameter_masks = cPickle.load(open(parameter_mask_file, "r"))
        lm_opt.freeze_masks = cPickle.load(open(freeze_mask_file, "r"))
        lm_opt.mapping_indexes = cPickle.load(open(mapping_indexes_file, "r"))
        logger.info('Loading Masks completed')

    if(lm_opt.freeze_model):
        logger.info('Loading Masks ')

        # lm_opt.parameter_masks = cPickle.load(open("../data/r1.0/models/m1/params.pickle", "r"))
        # lm_opt.freeze_masks = cPickle.load(open("../data/r1.0/masks/zeros/freeze_m1.pickle", "r"))
        
        file_suffix = "m1_m2_t" + str(lm_opt.threshold) + ".pickle"

        mask_folder = lm_opt.mask_folder

        lm_opt.parameter_masks = cPickle.load(open("../data/r1.0/masks/" + mask_folder + "/parameters_" + file_suffix, "r"))
        lm_opt.freeze_masks = cPickle.load(open("../data/r1.0/masks/" + mask_folder + "/freeze_" + file_suffix , "r"))

        lm_opt._emb = lm_opt.parameter_masks["LM/emb:0"]
        lm_opt._lstm_w = lm_opt.parameter_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights:0"]
        lm_opt._lstm_b = lm_opt.parameter_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases:0"]
        lm_opt._softmax_w = lm_opt.parameter_masks["LM/softmax_w:0"]
        lm_opt._softmax_b = lm_opt.parameter_masks["LM/softmax_b:0"]

        logger.info('Loading Masks completed')

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
                logger.info("Trainining completely frozen model - " + lm_opt.sensitivity)
                lm_train_op, lm_lr_var = lm.train_op_frozen(lm_train, lm_opt)

            else:
                lm_train_op, lm_lr_var = lm.train_op(lm_train, lm_opt)

        logger.debug('- Creating validating LM (reuse params)...')
        with tf.variable_scope('LM', reuse=True, initializer=initializer):
            lm_valid = lm.LM(lm_opt, is_training=False)
        logger.debug('Trainable variables:')
        for v in tf.trainable_variables():
            logger.debug("- {} {} {}".format(v.name, v.get_shape(), v.device))
        logger.info('Initializing variables...')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        states = {}
        for p in model_prefix:
            states[p] = common_utils.get_initial_training_state()
        states, _ = resume_many_states(lm_opt.output_dir, sess,
                                       saver, states, model_prefix)
        lm_state = states[model_prefix[0]]
        lm_state.learning_rate = lm_opt.learning_rate

        if(lm_opt.special_train):
            for v in tf.trainable_variables():
                if "emb" in v.name:
                    sess.run(tf.assign(v, tf.multiply(v,lm_opt.freeze_masks["LM/emb:0"])))
                    sess.run(tf.assign(v, tf.add(v,lm_opt.parameter_masks["LM/emb:0"])))
                    lm_opt._emb_var = v
                    break

        # if(lm_opt.freeze_model):
        #     for v in tf.trainable_variables():
        #         if "emb" in v.name:
        #             sess.run(tf.assign(v, lm_opt._emb))
        #             lm_opt._emb_var = v
        #         elif "basic_lstm_cell/weights" in v.name:
        #             sess.run(tf.assign(v, lm_opt._lstm_w))
        #             lm_opt._lstm_w_var = v
        #         elif "basic_lstm_cell/biases" in v.name:
        #             sess.run(tf.assign(v, lm_opt._lstm_b))
        #             lm_opt._lstm_b_var = v
        #         elif "softmax_w" in v.name:
        #             sess.run(tf.assign(v, lm_opt._softmax_w))
        #             lm_opt._softmax_w_var = v
        #         elif "softmax_b" in v.name:
        #             sess.run(tf.assign(v, lm_opt._softmax_b))
        #             lm_opt._softmax_b_var = v

        if(lm_opt.freeze_model):
            for v in tf.trainable_variables():
                if "emb" in v.name:
                    sess.run(tf.assign(v, tf.multiply(v,lm_opt.freeze_masks["LM/emb:0"])))
                    sess.run(tf.assign(v, tf.add(v,lm_opt.parameter_masks["LM/emb:0"])))
                    lm_opt._emb_var = v
                elif "basic_lstm_cell/weights" in v.name:
                    sess.run(tf.assign(v, tf.multiply(v,lm_opt.freeze_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights:0"])))
                    sess.run(tf.assign(v, tf.add(v,lm_opt.parameter_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights:0"])))
                    lm_opt._lstm_w_var = v
                elif "basic_lstm_cell/biases" in v.name:
                    sess.run(tf.assign(v, tf.multiply(v,lm_opt.freeze_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases:0"])))
                    sess.run(tf.assign(v, tf.add(v,lm_opt.parameter_masks["LM/rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases:0"])))
                    lm_opt._lstm_b_var = v
                elif "softmax_w" in v.name:
                    sess.run(tf.assign(v, tf.multiply(v,lm_opt.freeze_masks["LM/softmax_w:0"])))
                    sess.run(tf.assign(v, tf.add(v,lm_opt.parameter_masks["LM/softmax_w:0"])))
                    lm_opt._softmax_w_var = v
                elif "softmax_b" in v.name:
                    sess.run(tf.assign(v, tf.multiply(v,lm_opt.freeze_masks["LM/softmax_b:0"])))
                    sess.run(tf.assign(v, tf.add(v,lm_opt.parameter_masks["LM/softmax_b:0"])))
                    lm_opt._softmax_b_var = v

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

            done_training = run_post_epoch(
                lm_train_ppl, lm_valid_ppl, lm_state, lm_opt,
                sess=sess, saver=saver,
                best_prefix="best_lm", latest_prefix="latest_lm")
            logger.info('- Epoch time: {}s'.format(time.time() - epoch_time))
            if done_training:
                break

        logger.info('Done training at epoch {}'.format(lm_state.epoch + 1))

        # if (lm_opt.freeze_model):
        #
        #     if (np.array_equal(sess.run(lm_opt._emb_var), lm_opt._emb) != True):
        #         logger.info("MODEL FREEZE - _emb freeze went horribly wrong")
        #         exit()
        #
        #     if (np.array_equal(sess.run(lm_opt._lstm_w_var), lm_opt._lstm_w) != True):
        #         logger.info("MODEL FREEZE - _lstm_w freeze went horribly wrong")
        #         exit()
        #
        #     if (np.array_equal(sess.run(lm_opt._lstm_b_var), lm_opt._lstm_b) != True):
        #         logger.info("MODEL FREEZE - _lstm_b freeze went horribly wrong")
        #         exit()
        #
        #     if (np.array_equal(sess.run(lm_opt._softmax_w_var), lm_opt._softmax_w) != True):
        #         logger.info("MODEL FREEZE - _softmax_w freeze went horribly wrong")
        #         exit()
        #
        #     if (np.array_equal(sess.run(lm_opt._softmax_b_var), lm_opt._softmax_b) != True):
        #         logger.info("MODEL FREEZE - _softmax_b freeze went horribly wrong")
        #         exit()
        #
        #     logger.info("MODEL FREEZE - Successful")
        #
        # if (lm_opt.special_train):
        #     for index in lm_opt.mapping_indexes:
        #         if(np.array_equal(sess.run(lm_opt._emb_var)[:, index], lm_opt.parameter_masks["LM/emb:0"][:, index]) != True):
        #             logger.info("SPECIAL TRAIN - _emb freeze went horribly wrong")
        #             exit()
        #     logger.info("SPECIAL TRAIN - Successful")


if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
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

    main(opt)
    logger.info('Total time: {}s'.format(time.time() - global_time))
