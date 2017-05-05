import os
import time

# Email utils
import smtplib
import socket
from email.mime.text import MIMEText

# Bunch class utils
import common_utils

def send_email(receiver):

    hostname  = socket.gethostname()

    model_name = opt.new_model_path.split("/")
    model_name = model_name[-1]

    text = "Job on " + hostname + " completed. \n"
    text+= "The \"" + opt.test_on + "\" output data about model \"" + model_name + "\" are now available. \n"
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
    parser.add_argument('--test_on', type=str, default='test', help='Select train, valid, test')

    args = parser.parse_args()
    opt = common_utils.Bunch.default_model_options()
    opt.update_from_ns(args)

    if(opt.new_model_path == None):
        print "No model path provided, exiting"
        exit()

    model_path = opt.new_model_path

    common_arguments = "--emb_size " + str(opt.emb_size) + " "
    common_arguments+= "--state_size " + str(opt.state_size) + " "
    common_arguments+= "--max_grad_nor " + str(opt.max_grad_norm) + " "
    common_arguments+= "--output_dir " + model_path + " "
    common_arguments+= "--gpu "
    log_file_path = "--log_file_path " + model_path + "/training.log "


    print "Fetching model"
    rsync_command = "rsync -av "
    source_dir= "peroni:/nfs-scratch/emt1627/tf-ensemble/data/r1.0/" + model_path + "/"
    dest_dir= model_path
    rsync_command +=source_dir + " " + dest_dir
    print rsync_command
    os.system(rsync_command)

    print "Testing generated model"
    special = "--num_steps 1 "
    special+= "--batch_size 1 "
    special+= "--test_on " + opt.test_on + " "
    special+= "--out_token_loss_file model_output_" + opt.test_on +  ".tsv "
    print"test.py command", common_arguments + special
    os.system("python test.py " + common_arguments + special)

    print "Pushing model"
    rsync_command = "rsync -av "
    rsync_command+= dest_dir + " " + source_dir
    print rsync_command
    os.system(rsync_command)

    print "Purging local model"
    command = "rm -rf " + model_path
    print command
    os.system(command)

    print "Sending email"
    send_email("ettrai@u.northwestern.edu")

    print
    print 'Total time: {}s'.format(time.time() - global_time)
    print


# def gen_output(data_set):
#
#     file_list = glob.glob("../data/r1.0/models/*/")
#
#     counter = 1
#
#     for gen_for in file_list:
#
#         to_call = "python gen_new_model.py --emb_size 300 --state_size 300 --max_grad_nor 5.0 "
#         to_call+= "--test_on " + data_set + " "
#         to_call+= "--new_model_path " + gen_for + " "
#         # print to_call
#
#         # tmux = "tmux new -s " + "run_" + str(counter) + " -d "
#         # tmux+= "'"
#         # tmux+="source ~/Env/tensorflow-r1.0;"
#         # tmux+=to_call
#         # tmux+="'"
#         #
#         # counter +=1
#
#         os.system(to_call)
#
#     return
#
# if __name__ == "__main__":
#     print
#     gen_output("train")
#     print
#
