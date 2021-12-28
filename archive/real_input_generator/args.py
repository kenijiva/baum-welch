from configargparse import ArgumentParser
import argparse

def get_parser():
    parser = ArgumentParser(description='Baum-Welch input data generator',
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                          
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add('-N', '--num_states', type=int, default=6, help="number of states in the model")
    parser.add('-M', '--num_observation_symbols', type=int, default=6, help="number of distinct observation symbols per state")
    parser.add('-T', '--num_observed_symbols', type=int, default=6, help="number of observed symbols")

    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()

    return args
