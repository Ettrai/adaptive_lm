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

    new_model_path = opt.new_model_path

    if os.path.exists(new_model_path):
        if len(os.listdir(new_model_path)) > 0 and opt.force_path!=True:
            print new_model_path + " is not empty, assuming wrong parameters"
            print "Exiting"
            exit()

    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path)

def send_email(receiver):

    hostname  = socket.gethostname()

    model_name = opt.new_model_path.split("/")
    model_name = model_name[-1]

    text = "Generation of a new model on " + hostname + " completed. \n"
    text+= "The data about model \"" + model_name + "\" are now available. \n"
    text+= "You can find the model in \"" +  opt.new_model_path + "\""

    print text

    msg = MIMEText(text)

    sender = "noreply@" + hostname

    msg['Subject'] = "Model '" + model_name + "' on " + hostname + " has been generated"
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
    parser.add_argument('--force_path', action='store_true', help='Erases content of chosen folder if not empty')

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
    log_file_path = "--log_file_path " + opt.new_model_path + "/training.log "
    special = ""
    if(opt.special_train):
        special = "--special_train "
    if(opt.freeze_model):
        special = "--freeze_model "
    os.system("python train.py " + common_arguments + log_file_path + special)

    print "Dumping new model parameters"
    os.system("python get_params.py " + common_arguments)

    print "Testing generated model"
    num_steps  = "--num_steps 1 "
    batch_size = "--batch_size 1 "
    out_token_loss_file = "--out_token_loss_file model_output.tsv "
    os.system("python test.py " + num_steps + batch_size + out_token_loss_file + common_arguments)

    print "Sending email"
    send_email("ettrai@u.northwestern.edu")

    print
    print 'Total time: {}s'.format(time.time() - global_time)
    print


