import common_utils
import logging
import time

import test

def gather_model_outputs(model_name):

    models_path = "models/"
    # models_names = ["m1" , "m2" , "m3", "m4"]

    opt.num_steps = 10
    opt.batch_size = 100
    opt.emb_size = 300
    opt.state_size = 300
    opt.max_grad_nor= 5.0
    opt.vocab_ppl_file = "model_output.tsv"

    opt.output_dir = models_path + model_name
    opt.logger = logger
    # logger.info('Configurations:\n{}'.format(opt.__repr__()))
    # CANNOT RUN SESSIONS LIKE THIS
    # ERRORS STARTING FROM THE SECOND ITERATION!!!
    test.main(opt)


def ensemble_models_outputs():
    print "I will ensemble models one day"

def main():
    # gather_model_outputs("m1")
    gather_model_outputs("m2")
    logger.info('Total time: {}s'.format(time.time() - global_time))

if __name__ == "__main__":
    global_time = time.time()
    parser = common_utils.get_common_argparse()
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

    main()
