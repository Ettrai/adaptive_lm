import time
import argparse
import os

# Email utils
import smtplib
import socket
from email.mime.text import MIMEText

# Bunch class utils
import common_utils

def setup_folders():

    directory = opt.output_dir = opt.new_model_path

    if os.path.exists(directory):
        print directory + " exists already, assuming wrong parameters"
        print "Exiting"
        exit()

    if not os.path.exists(directory):
        os.makedirs(directory)

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
    parser = argparse.ArgumentParser()

    parser.add_argument('--new_model_path', type=str,default=None, help='Train and generates output of a new model')

    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)

    if(opt.new_model_path == None):
        print "No model path provided, exiting"
        exit()

    print "Setting up folders"
    setup_folders()

    print "Training new model"
    common_arguments = "--emb_size 300 "
    common_arguments+= "--state_size 300 "
    common_arguments+= "--max_grad_nor 5.0 "
    common_arguments+= "--output_dir " + opt.new_model_path + " "
    common_arguments+= "--gpu "
    log_file_path = "--log_file_path " + opt.new_model_path + "/training.log"
    os.system("python train.py " + common_arguments + log_file_path)

    print "Dumping new model parameters"
    os.system("python get_params.py " + common_arguments)

    num_steps  = "--num_steps 1 "
    batch_size = "--batch_size 1 "
    out_token_loss_file = "--out_token_loss_file model_output.tsv"
    os.system("python test.py " + num_steps + batch_size + common_arguments)

    print "Sending email"
    send_email("ettrai@u.northwestern.edu")

    print
    print 'Total time: {}s'.format(time.time() - global_time)
    print


