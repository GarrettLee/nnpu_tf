#!/usr/bin/python3
import os
import sys
import copy

import helper
import trainer

__author__ = 'garrett_local'


def exp(dataset_name):
    cfg_path = './cfg/{}'.format(dataset_name)
    pn_cfg = helper.read_cfg_file(os.path.join(cfg_path, 'PN'))
    upu_cfg = helper.read_cfg_file(os.path.join(cfg_path, 'uPU'))
    nnpu_cfg = helper.read_cfg_file(os.path.join(cfg_path, 'nnPU'))
    assert \
        upu_cfg['dataset']['dataset_name'] == \
        nnpu_cfg['dataset']['dataset_name'] and \
        upu_cfg['network']['network_name'] == \
        nnpu_cfg['network']['network_name'] and \
        pn_cfg['dataset']['dataset_name'] == \
        nnpu_cfg['dataset']['dataset_name'] and \
        pn_cfg['network']['network_name'] == \
        nnpu_cfg['network']['network_name']
    exp_name = 'exp_{}_{}_{}'.format(
        nnpu_cfg['dataset']['dataset_name'],
        nnpu_cfg['network']['network_name'],
        helper.get_unique_name()
    )
    log_data = helper.LogData()

    # upu and nnpu.
    PuDataset, PnDataset = helper.load_dataset(upu_cfg)
    pu_dataset = PuDataset(upu_cfg['dataset'])
    training_iterator = pu_dataset.get_training_iterator()

    Network = helper.load_network(upu_cfg)
    upu_trainer = trainer.TrainerBase(upu_cfg['trainer'])
    upu_trainer.setup_network(Network(upu_cfg['network'], pu_dataset.prior))
    nnpu_trainer = trainer.TrainerBase(nnpu_cfg['trainer'])
    nnpu_trainer.setup_network(Network(nnpu_cfg['network'], pu_dataset.prior))

    epoch = 0
    upu_train_accum, nnpu_train_accum = [], []
    for data in training_iterator:
        upu_train_accum.append(upu_trainer.train(data))
        nnpu_train_accum.append(nnpu_trainer.train(data))
        if training_iterator.epoch_finished > epoch:
            epoch = training_iterator.epoch_finished

            # train losses.
            upu_train_loss = sum(upu_train_accum) / float(len(upu_train_accum))
            nnpu_train_loss = sum(nnpu_train_accum) / float(len(nnpu_train_accum))
            upu_train_accum.clear()
            nnpu_train_accum.clear()

            # test 0-1 losses.
            test_iter = pu_dataset.get_testing_iterator()
            upu_test_loss = upu_trainer.evaluate_error(copy.deepcopy(test_iter))
            nnpu_test_loss = nnpu_trainer.evaluate_error(test_iter)

            print(
                'Epoch: {0:>5}, upu train: {1:7.4f}, upu test: {2:7.4f}, '
                'nnpu train: {3:7.4f}, nnpu test: {4:7.4f}'
                .format(epoch, upu_train_loss, upu_test_loss, nnpu_train_loss,
                        nnpu_test_loss))
            log_data.log_loss('upu train', upu_train_loss)
            log_data.log_loss('nnpu train', nnpu_train_loss)
            log_data.log_loss('upu test', upu_test_loss)
            log_data.log_loss('nnpu test', nnpu_test_loss)

    # pn.
    pn_dataset = PnDataset(pn_cfg['dataset'], pu_dataset.prior)
    pn_trainer = trainer.TrainerBase(pn_cfg['trainer'])
    pn_trainer.setup_network(Network(pn_cfg['network'], pu_dataset.prior))
    pn_training_iterator = pn_dataset.get_training_iterator()
    epoch = 0
    pn_accum = []
    for data in pn_training_iterator:
        pn_accum.append(pn_trainer.train(data))
        if pn_training_iterator.epoch_finished > epoch:
            epoch = pn_training_iterator.epoch_finished

            pn_train_loss = sum(pn_accum) / float(len(pn_accum))
            pn_accum.clear()

            test_set = pn_dataset.get_testing_iterator()
            pn_test_loss = pn_trainer.evaluate_error(test_set)
            log_data.log_loss('pn test', pn_test_loss)
            log_data.log_loss('pn train', pn_train_loss)
            print('Epoch: {0:>5}, pn train: {1:7.4f}, pn test: {2:7.4f}'
                  .format(epoch, pn_train_loss, pn_test_loss))
    helper.save_log_data(log_data, exp_name)
    helper.settle_saved_data(exp_name)
    return exp_name

if __name__ == '__main__':
    # MNIST or CIFAR10
    dataset = sys.argv[1]

    num_trial = int(sys.argv[2])
    exp_set_path = './exp_set/{0}/{0}.txt'.format(dataset)
    exps = []
    for i in range(num_trial):
        exps.append(exp(dataset))
    helper.save_basetrings_as_text(
        exps,
        exp_set_path
    )
    print('Result: {}'.format(exps))
    helper.draw_losses('{}.txt'.format(dataset), '{}'.format(dataset))
