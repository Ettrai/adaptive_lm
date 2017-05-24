import time
import argparse
import os

import sys

# Email utils
import smtplib
import socket
from email.mime.text import MIMEText

# Bunch class utils
import common_utils

email_receiver = "ettrai@u.northwestern.edu"

def create_model_directory(directory):

    # if os.path.exists(directory):
    #     print "Model output folder already exists"
    #     print "Emailing error message"
    #     print
    #     send_error_mail(email_receiver, "Model output directory already exists!\n")
    #     exit()

    if not os.path.exists(directory):
        print "Creating directory", directory
        os.makedirs(directory)


def send_error_mail(send_to, error):
    hostname  = socket.gethostname()
    text = "Something weird happened on " + hostname + "\n\n"
    # Arguments
    text+= arguments_to_text()

    # Error message
    text+= "This is the error message:\n"
    text+= error
    send_email(send_to, text)

def send_gen_completed_mail(send_to):
    hostname  = socket.gethostname()
    text = "Generation of a new model on " + hostname + " completed. \n"
    text+= "You can find the model in \"" + opt.output_dir + "\" \n\n"
    text+= arguments_to_text()
    send_email(send_to, text)

def send_email(send_to, text):

    hostname  = socket.gethostname()

    msg = MIMEText(text)

    sender = "noreply@" + hostname

    msg['Subject'] = "Updates from " + hostname
    msg['From'] = "noreply@" + hostname
    msg['To'] = send_to

    s = smtplib.SMTP('localhost')
    s.sendmail(sender, [send_to], msg.as_string())
    s.quit()

def arguments_to_text():
    text= "These are the arguments passed to " + sys.argv[0] + "script: \n\n"
    text+= opt.__repr__()
    text+= "\n\n"
    return text

if __name__ == "__main__":
    print

    global_time = time.time()
    parser = common_utils.get_common_argparse()

    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)

    arguments = ""
    for argument in sys.argv[1:]:
        arguments+= argument + " "

    print "Setting up folders"
    create_model_directory(opt.output_dir)

    train_command =  "python train.py " + arguments
    print "Training new model"
    print train_command
    os.system(train_command)
    print

    dump_command = "python get_params.py " + arguments
    dump_command = dump_command.replace("training.log", "parameter_dump.log")
    print "Dumping new model parameters"
    print dump_command
    os.system(dump_command)
    print

    test_command = "python test.py " + arguments
    test_command+= "--num_steps 1 "
    test_command+= "--batch_size 1 "
    test_set = "--test_on " + "test" + " "
    test_set+= "--out_token_loss_file model_output_" + "test" +  ".tsv "
    test_command = test_command.replace("training.log", "testing_validation_set.log")
    print "Testing generated model on testing set"
    print test_command + test_set
    os.system(test_command + test_set)
    print

    test_set = "--test_on " + "valid" + " "
    test_set+= "--out_token_loss_file model_output_" + "valid" +  ".tsv "
    test_command = test_command.replace("testing_validation_set.log","testing_test_set.log")
    print test_command + test_set
    print "Testing generated model on validation set"
    os.system(test_command + test_set)
    print

    print "Adding generated model to list of models to push from Finagle"
    rsync_command = "rsync -av "
    source_dir= opt.output_dir + "/"
    dest_dir= "peroni:/nfs-scratch/emt1627/tf-ensemble/data/r1.0/" + opt.output_dir + "/"
    rsync_command +=source_dir + " " + dest_dir + "\n"
    clean_command = "rm -rf " + opt.output_dir + "\n"
    with open("models/to_push/" + socket.gethostname() + str(global_time), "w") as myfile:
        myfile.write(rsync_command)
        myfile.write(clean_command)

    print "Model generation completed, sending email"
    send_gen_completed_mail(email_receiver)

    print
    print 'Total time: {}s'.format(time.time() - global_time)
    print


