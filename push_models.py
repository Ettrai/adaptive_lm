import glob
import os

# Email utils
import smtplib
import socket
from email.mime.text import MIMEText

def push_models():

    # Uploads all the correctly generated models
    models_directories = glob.glob("models/to_push/*")
    for model_to_push in models_directories:

        commands = []
        with open(model_to_push) as f:
            commands = f.readlines()

        commands = [x.strip() for x in commands]
        for command in commands:
            print command
            os.system(command)

        command = "rm -rf " + model_to_push
        print command
        os.system(command)
        print

    # Check which models probably failed
    to_check = []
    directories_list = glob.glob("models/*/")

    for directory in directories_list:
        models_folders = glob.glob(directory + "*/")

        for model in models_folders:
            temp = model + "\n"
            to_check.append(temp)

    if (len(to_check) != 0):
        text = "Check the following models folders on Finagle:\n\n"
        for model in to_check:
            text+=  " - " + model

        send_to = "ettrai@u.northwestern.edu"
        print "Sending email to " + send_to
        send_email(send_to , text)

    else:
        print "Push models completed with 0 warnings"

def send_email(receiver, text):

    hostname  = socket.gethostname()

    msg = MIMEText(text)

    sender = "noreply@" + hostname

    msg['Subject'] = "[Project X] push_models.py log"
    msg['From'] = "noreply@" + hostname
    msg['To'] = receiver

    s = smtplib.SMTP('localhost')
    s.sendmail(sender, [receiver], msg.as_string())
    s.quit()


if __name__ == "__main__":
    push_models()



