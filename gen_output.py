import glob
import os

def gen_output(data_set):

    file_list = glob.glob("../data/r1.0/models/*/")

    counter = 1

    for gen_for in file_list:

        to_call = "python gen_new_model.py --emb_size 300 --state_size 300 --max_grad_nor 5.0 "
        to_call+= "--test_on " + data_set + " "
        to_call+= "--new_model_path " + gen_for + " "
        # print to_call

        # tmux = "tmux new -s " + "run_" + str(counter) + " -d "
        # tmux+= "'"
        # tmux+="source ~/Env/tensorflow-r1.0;"
        # tmux+=to_call
        # tmux+="'"
        #
        # counter +=1

        os.system(to_call)

    return

if __name__ == "__main__":
    print
    gen_output("train")
    print

