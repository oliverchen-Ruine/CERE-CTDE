import os
from datetime import datetime

from run import run
from utils.param_utils import print_parameters
from multiprocessing import Process
from parameters.parameters_preamble import *


# COMPARISON = True
COMPARISON = False


def main():
    from parameters import parameters
    # parameters.Parameters_27Mvs30M()
    # env_name = [parameters.Parameters_somanybaneling(), parameters.Parameters_2s3z(), parameters.Parameters_8m(),
    #             parameters.Parameters_MMM(), parameters.Parameters_10Mvs11M(), parameters.Parameters_3s5z(),
    #             parameters.Parameters_3S5Zvs3S6Z(), parameters.Parameters_MMM2(), parameters.Parameters_Corridor(),
    #             parameters.Parameters_6Hvs8Z()]
    env_name = [parameters.Parameters_2_corridors]

    for env in env_name:
        params = env

        for run_number in range(params.RUNS):
            current_date = datetime.now()
            month = current_date.month
            day = current_date.day
            hour = current_date.hour
            minute = current_date.minute
            seed_value = int(f"{month:02d}{day:02d}{hour:02d}{minute:02d}")
            params.ENV_ARGS['seed'] = seed_value
            params.ENV_ARGS['env_args']['seed'] = seed_value
            print_parameters(params, params.param_overlapped_dict)
            run(params, run_number)
            print("Run:{}/{} is finished".format(run_number, params.RUNS))

    print("Exiting script")

    os._exit(os.EX_OK)


def comparison_main():
    from parameters import multi_parameters
    params_list = multi_parameters.MultiParameters().params_list
    RUNS = params_list[0].RUNS

    for run_number in range(RUNS):
        ps = [Process(target=run, args=(params, run_number)) for params in params_list]
        for p in ps:
            p.daemon = False
            p.start()
        for p in ps:
            p.join()

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


if __name__ == '__main__':
    if COMPARISON:
        comparison_main()
    else:
        main()
