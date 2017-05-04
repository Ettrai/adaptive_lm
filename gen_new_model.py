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
    parser.add_argument('--test_on', type=str, default='test', help='Select train, valid, test')

    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)

    if(opt.new_model_path == None):
        print "No model path provided, exiting"
        exit()

    print "Setting up folders"
    # setup_folders()

    print "Training new model"
    common_arguments = "--emb_size " + str(opt.emb_size) + " "
    common_arguments+= "--state_size " + str(opt.state_size) + " "
    common_arguments+= "--max_grad_nor " + str(opt.max_grad_norm) + " "
    common_arguments+= "--output_dir " + opt.new_model_path + " "
    common_arguments+= "--gpu "
    log_file_path = "--log_file_path " + opt.new_model_path + "/training.log "
    special = ""

    if(opt.special_train):
        special = "--special_train "
        special+= "--num_shared_neurons " + str(opt.num_shared_neurons) + " "
        special+= "--gen_from_models " + opt.gen_from_models + " "

    if(opt.freeze_model):
        special = "--freeze_model "
        special +="--threshold " + str(opt.threshold) + " "

    print "train.py command",common_arguments + log_file_path + special
    # os.system("python train.py " + common_arguments + log_file_path + special)

    print "Dumping new model parameters"
    print "get_params.py command",common_arguments
    # os.system("python get_params.py " + common_arguments)

    print "Testing generated model"
    special = "--num_steps 1 "
    special+= "--batch_size 1 "
    special+= "--test_on " + opt.test_on + " "
    special+= "--out_token_loss_file model_output_" + opt.test_on +  ".tsv "
    print"test.py command", common_arguments + special
    os.system("python test.py " + common_arguments + special)

    # print "Sending email"
    # send_email("ettrai@u.northwestern.edu")

    print
    print 'Total time: {}s'.format(time.time() - global_time)
    print


