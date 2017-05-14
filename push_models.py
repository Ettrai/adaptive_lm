import glob
import os

def push_models():

    file_list = glob.glob("models/to_push/*")

    for model_to_push in file_list:

        commands = []
        with open(model_to_push) as f:
            commands = f.readlines()

        commands = [x.strip() for x in commands]
        for command in commands:
            print command
            # os.system(command)

        command = "rm -rf " + model_to_push
        print command
        # os.system(command)
        print


if __name__ == "__main__":
    push_models()



