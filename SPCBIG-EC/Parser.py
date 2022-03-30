import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contracts Reentrancy Detection')

    parser.add_argument('-D', '--dataset', type=str, default='data/dataset_reentry.txt',
                        choices=['train_data/infinite_loop_1317.txt',
                                 'train_data/reentrancy_1671.txt',
                                 'train_data/reentrancy_code_snippets_2000.txt',
                                 'data/reentrancy/graph_data_smart_contract_2000.txt',
                                 'data/dataset_integer_big.txt',
                                 'data/dataset_reentry.txt',
                                 'data/dataset_all.txt',
                                 'data/dataset_CDAV.txt',
                                 'data/dataset_integerUnderFlow.txt',
                                 'train_data/timestamp.txt'])
    parser.add_argument('-M', '--model', type=str, default='CNN-BiGRU-ATT',
                        choices=['CNN-BiGRU-ATT','SPCNN-BiGRU-ATT','SCNN-BiGRU-ATT'])
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--vector_dim', type=int, default=300, help='dimensions of vector')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='threshold')



    return parser.parse_args()
