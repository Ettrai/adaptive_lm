import common_utils
import logging
import time
import csv

import numpy as np

import test

def gather_model_outputs(model_name):

    models_path = "models/"
    # models_names = ["m1" , "m2" , "m3", "m4"]

    opt.num_steps = 1
    opt.batch_size = 1
    opt.emb_size = 300
    opt.state_size = 300
    opt.max_grad_nor= 5.0
    opt.vocab_ppl_file = "model_output.tsv"

    opt.output_dir = models_path + model_name
    opt.logger = logger

    #logger.info('Configurations:\n{}'.format(opt.__repr__()))
    # CANNOT RUN SESSIONS LIKE THIS
    # ERRORS STARTING FROM THE SECOND ITERATION!!!
    test.main(opt)


def ensemble_models_outputs(arguments):
    models_path = "models/"
    models_names = arguments.split(',')

    # models_names = ["m1" , "m2" , "m3", "m4"]
    #models_names = ["m1" , "m2" ]

    num_models = len(models_names)

    models_outputs = []
    for index in range(num_models):
        models_outputs.append([])

    for index, model in enumerate(models_names):
        file = models_path + model + "/" + "model_output.tsv"
        with open(file, 'rb') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tsvin:
                models_outputs[index].append(float(row[1]))

    #Check number of tokens
    num_tokens = len(models_outputs[0])

    for index in range(num_models):
        if(len(models_outputs[index]) != num_tokens):
            print "Number of tokens differs between models!"
            exit()

    for index in range(num_models):
        print "Model " + models_names[index] + " average loss: " + str(np.mean(models_outputs[index]))
        print "Model " + models_names[index] + " PPL: " + str(np.exp(np.mean(models_outputs[index])))


    ensemble_output = []

    for token in range(num_tokens):
        accumulate = 0.0
        for model in range(num_models):
            accumulate += models_outputs[model][token]
        average = accumulate / num_models
        ensemble_output.append(average)

    print
    print "Ensemble average loss: " + str(np.mean(ensemble_output))
    print "Ensemble PPL: " + str(np.exp(np.mean(ensemble_output)))

if __name__ == "__main__":
    print

    global_time = time.time()
    parser = common_utils.get_common_argparse()
    parser.add_argument('--compute_output', type=bool,default=False, help='compute output of a model')
    parser.add_argument('--compute_ensemble', type=bool,default=False, help='compute output of a model')
    parser.add_argument('--ensemble_of', type=str,default=None, help='models names')
    parser.add_argument('--model_name', type=str,default=None, help='model to gather outputs from')

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

    if(opt.compute_output):
        gather_model_outputs(opt.model_name)

    if(opt.compute_ensemble):
        ensemble_models_outputs(opt.ensemble_of)

    print
    logger.info('Total time: {}s'.format(time.time() - global_time))
    print
