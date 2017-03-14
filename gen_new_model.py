import common_utils
import logging
import time
import csv

import numpy as np

import train
import get_params
import test

import smtplib
import socket
from email.mime.text import MIMEText

import os

def setup_folders():

    directory = opt.output_dir = opt.new_model_path

    if os.path.exists(directory):
        print directory + " exists already, assuming wrong parameters"
        print "Exiting"
        exit()

    if not os.path.exists(directory):
        os.makedirs(directory)


def train_network():
    opt.emb_size = 300
    opt.state_size = 300
    opt.max_grad_nor = 5.0
    opt.gpu=True
    opt.log_file_path = opt.new_model_path + "/training.log"
    opt.output_dir = opt.new_model_path
    train.main(opt)

def dump_parameters():
    opt.output_dir = opt.new_model_path
    get_params()

def test_model():
    opt.num_steps = 1
    opt.batch_size = 1
    opt.emb_size = 300
    opt.state_size = 300
    opt.max_grad_nor = 5.0
    opt.out_token_loss_file = "model_output.tsv"
    opt.gpu=True
    opt.output_dir = opt.new_model_path
    test.main()

def send_email(receiver):

    hostname  = socket.gethostname()

    text = "Generation of a new model on " + hostname + " completed. "
    text+= "The data about model " + opt.new_model_path + " are now available"

    msg = MIMEText(text)

    sender = "noreply@" + hostname

    msg['Subject'] = "Gen Model on " + hostname + " completed"
    msg['From'] = "noreply@" + hostname
    msg['To'] = receiver

    s = smtplib.SMTP('localhost')
    s.sendmail(sender, [receiver], msg.as_string())
    s.quit()

if __name__ == "__main__":
    print

    global_time = time.time()
    parser = common_utils.get_common_argparse()

    parser.add_argument('--new_model_path', type=str,default=None, help='Train and generates output of a new model')

    parser.add_argument('--vocab_ppl_file', type=str,
                        default=None,
                        help='output vocab ppl to a file')

    parser.add_argument('--shared_emb', dest='shared_emb',
                        action='store_true', help='use emb from shared_emb scope')
    parser.set_defaults(shared_emb=False)

    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)

    logger = common_utils.get_logger(opt.log_file_path)
    if opt.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    opt.logger = logger

    if(opt.new_model_path == None):
        print "No model path provided, exiting"
        exit()

    logger.info("Setting up model folder")
    setup_folders()
    logger.info("Training new model")
    train_network()
    logger.info("Dumping new model parameters")
    dump_parameters()
    logger.info("Testing and generating model outputs")
    test_model()

    logger.info("Sending email")
    send_email("ettrai@u.northwestern.edu")

    print
    logger.info('Total time: {}s'.format(time.time() - global_time))
    print

