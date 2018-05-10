import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import helper.file_helper as file_manager


__author__ = 'garrett_local'


def _not_sequence(var):
    return not isinstance(var, list) and not isinstance(var, tuple)


def _create_colors(var):
    if _not_sequence(var):
        return 'green'

    # Create color dynamically.
    jet = plt.get_cmap('jet')
    colors = []
    assert not _not_sequence(var)
    if len(var) > 0 and _not_sequence(var[0]):
        colors = list(range(len(var)))
        c_norm = matplotlib.colors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        scalar_map = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=jet)
        colors = [scalar_map.to_rgba(colors[i]) for i in range(len(colors))]
    # For histograms.
    else:
        for group in var:
            assert not _not_sequence(group)
            for element in group:
                assert _not_sequence(element)
        group_num = len(var)
        for i, group in enumerate(var):
            c = []
            for j in range(len(group)):
                c.append(float(i) / float(group_num) + j)
            colors.append(c)

        c_norm = matplotlib.colors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        scalar_map = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=jet)
        colors = [[scalar_map.to_rgba(colors[i][j]) for j in range(len(colors[i]))]
                  for i in range(len(colors))]
    return colors


def draw_losses(exp_set_name, exp_set_type='unnpu'):
    exps = file_manager.load_basestrings(
        './exp_set/{}/{}'.format(exp_set_type, exp_set_name)
    )
    upu_train = []
    upu_test = []
    nnpu_train = []
    nnpu_test = []
    pn_train = []
    pn_test = []
    for exp in exps:
        log_data = file_manager.read_exp_log_data(exp)
        upu_train.append(log_data.losses['upu train'])
        upu_test.append(log_data.losses['upu test'])
        nnpu_train.append(log_data.losses['nnpu train'])
        nnpu_test.append(log_data.losses['nnpu test'])
        pn_train.append(log_data.losses['pn train'])
        pn_test.append(log_data.losses['pn test'])
    plots = []
    legend = []
    title = 'loss(epoch)'
    upu_train = np.vstack(upu_train).mean(axis=0)
    upu_test = np.vstack(upu_test).mean(axis=0)
    upu_train_plot, = plt.plot(upu_train, 'y--')
    upu_test_plot, = plt.plot(upu_test, 'y-')
    plots.extend([upu_train_plot, upu_test_plot])
    legend.extend(['uPU train', 'uPU test'])

    nnpu_train = np.vstack(nnpu_train).mean(axis=0)
    nnpu_test = np.vstack(nnpu_test).mean(axis=0)
    nnpn_train_plot, = plt.plot(nnpu_train, 'r--')
    nnpn_test_plot, = plt.plot(nnpu_test, 'r-')
    plots.extend([nnpn_train_plot, nnpn_test_plot])
    legend.extend(['nnPU train', 'nnPU test'])

    pn_train = np.vstack(pn_train).mean(axis=0)
    pn_test = np.vstack(pn_test).mean(axis=0)
    pn_train_plot, = plt.plot(pn_train, 'b--')
    pn_test_plot, = plt.plot(pn_test, 'b-')
    plots.extend([pn_train_plot, pn_test_plot])
    legend.extend(['PN train', 'PN test'])

    plt.legend(
        plots,
        legend,
        loc='upper right'
    )
    plt.title(title)
    plt.grid(True, linestyle="-.")
    plt.show()
