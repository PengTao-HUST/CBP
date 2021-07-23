import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import seaborn as sns


__all__ = [
    'plot_series',
    'plot_compare_mul_loss_acc',
    'plot_mul_loss_acc',
    'plot_out_traj',
    'plot_w_traj',
    'plot_loss',
    'plot_train_time',
    'plot_compare_loss_acc',
    'plot_loss_acc',
    'plot_3d_w',
    'plot_xor_weight',
    'plot_lyapunov_exponent_with_z'
]

def plot_series(series, xlabel='epoch', ylabel=None, title=None,
                save_path=None, **kwargs):
    n_ele = len(series)
    fig, ax = plt.subplots()
    ax.scatter(range(n_ele), series, s=5, **kwargs)
    ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    plt.show()


def plot_w_traj(ws, first=6, save_prefix=None, suffix='w_traj',
                suptitle=None, figsize=None):
    ws = np.asarray(ws)
    max_epoch, max_col = ws.shape
    if figsize is None: figsize = (6, 8)
    iter = min(max_col, first)
    fig, ax = plt.subplots(iter, 1, figsize=figsize, sharex=True)
    if iter == 1: ax = [ax]

    for i in range(iter):
        ax[i].scatter(range(max_epoch), ws[:,i], s=5)
    for i, v in enumerate([f'w{i}' for i in range(1, iter+1)]):
        ax[i].set_ylabel(v)
    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(f'{save_prefix}{suffix}.png')
        plt.close()
    else:
        plt.show()


def plot_xor_weight(ws, save_prefix=None, suffix='xor_weight', suptitle=None):
    ws1 = np.array([w[0].flatten() for w in ws])
    bs1 = np.array([w[1].flatten() for w in ws])
    ws2 = np.array([w[2].flatten() for w in ws])
    bs2 = np.array([w[3].flatten() for w in ws])
    w = np.hstack([ws1, bs1, ws2, bs2])
    plot_w_traj(w, first=9, save_prefix=save_prefix, suffix=suffix, suptitle=suptitle)


def plot_out_traj(outs, first=4, save_prefix=None, suffix='out_traj',
                  suptitle=None, figsize=None):
    outs = np.asarray(outs)
    max_epoch, max_col = outs.shape
    if figsize is None: figsize = (6, 8)
    iter = min(max_col, first)
    fig, ax = plt.subplots(iter, 1, figsize=figsize, sharex=True)
    if iter == 1: ax = [ax]

    for i in range(iter):
        ax[i].scatter(range(max_epoch), outs[:,i], s=5)
    for i, v in enumerate([f'out{i}' for i in range(1, iter+1)]):
        ax[i].set_ylabel(v)
    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(f'{save_prefix}{suffix}.png')
        plt.close()
    else:
        plt.show()


def plot_3d_w(ws, inds=[0, 4, 8], save_prefix=None, suffix='3d_w'):
    ws = np.asarray(ws)
    max_epoch = ws.shape[0]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter3D(ws[:,inds[0]], ws[:,inds[1]], ws[:,inds[2]],
                      c=range(max_epoch), cmap='rainbow')
    ax.set_xlabel(f'w{inds[0]+1}')
    ax.set_ylabel(f'w{inds[1]+1}')
    ax.set_zlabel(f'w{inds[2]+1}')
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    if save_prefix is not None:
        wind = '-'.join([str(i+1) for i in inds])
        fig.savefig(f'{save_prefix}{suffix}_{wind}.png')
        plt.close()
    else:
        plt.show()


def plot_loss(loss, save_prefix=None, suffix='train_loss', log_scale=False):
    n_iter = len(loss)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(n_iter), loss, s=5)
    ax.set_xlabel('iteration')
    ax.set_ylabel('train loss')
    if log_scale:
        ax.set_yscale('log')
    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(f'{save_prefix}{suffix}.png')
        plt.close()
    else:
        plt.show()


def plot_lyapunov_exponent_with_z(nolds_les, z=20, beta=0.99, max_iter=800,
                                  save_prefix=None, suffix='lyapunov_exponent', **kws):
    n_points = len(nolds_les)
    assert n_points == max_iter
    z_list = [z * beta ** i for i in range(max_iter)]
    fig, ax = plt.subplots()
    ax.scatter(z_list, nolds_les, **kws)
    ax.set_xlabel('z (chaotic intensity)')
    ax.set_ylabel('Lyapunov exponent')
    ax.axhline(y=0, c='red', ls='dashed', lw=2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, z)
    ax.grid()
    fig.tight_layout()

    if save_prefix is not None:
        fig.savefig(f'{save_prefix}{suffix}.png')
        plt.close()
    else:
        plt.show()


def plot_loss_acc(loss_list, acc_list, save_prefix=None,
                  log_scale=False, acc_train=False, suptitle=None):
    y_acc = 'train acc' if acc_train else 'test acc'
    max_epoch1 = len(loss_list)
    max_epoch2 = len(acc_list)
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].scatter(range(max_epoch1), loss_list, s=5)
    ax[1].scatter(range(max_epoch2), acc_list, s=5)
    ylabels = ['train loss', y_acc]
    xlabels = ['iteration', 'epoch']
    for i in range(2):
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xlabel(xlabels[i])
    if log_scale:
        ax[0].set_yscale('log')

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(f'{save_prefix}loss_acc.png')
        plt.close()
    else:
        plt.show()


def plot_mul_loss_acc(train_loss, train_acc, save_prefix=None, acc1=None, acc2=None,
                      loss_lim=None, acc_lim=None, alpha=0.5, acc_train=False,
                      cs=None, despine=False, loss_log_scale=True, legend=True,
                      despine_all=False, figsize=None, xlabels=None,
                      ylabels=None, dpi=150):
    train_loss = np.asarray(train_loss)
    train_acc = np.asarray(train_acc)
    y_acc = 'train acc' if acc_train else 'test acc'
    n_line = train_loss.shape[1] // 2
    if cs is None:
        cs = sns.color_palette('pastel') #['b', 'r']
    if figsize is None:
        figsize = (6, 8)

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    ax[0].plot(train_loss[:, :n_line], c=cs[0], ls='--', alpha=alpha)
    ax[0].plot(train_loss[:, n_line:], c=cs[1], ls='-', alpha=alpha)
    ax[1].plot(train_acc[:, :n_line], c=cs[0], ls='--', alpha=alpha)
    ax[1].plot(train_acc[:, n_line:], c=cs[1], ls='-', alpha=alpha)

    if loss_log_scale: ax[0].set_yscale('log')

    if xlabels is None:
        ax[0].set_xlabel('iteration')
        ax[1].set_xlabel('epoch')
    else:
        ax[0].set_xlabel(xlabels[0])
        ax[1].set_xlabel(xlabels[1])

    if ylabels is None:
        ax[0].set_ylabel('train loss')
        ax[1].set_ylabel(y_acc)
    else:
        ax[0].set_ylabel(ylabels[0])
        ax[1].set_ylabel(ylabels[1])

    if legend:
        ax[0].legend(handles=ax[0].lines[::n_line], labels=['BP', 'CBP'], loc='upper right')

    if loss_lim is not None: ax[0].set_ylim(loss_lim)
    if acc_lim is not None: ax[1].set_ylim(acc_lim)

    if acc1 is not None:
        ax[1].axhline(y=acc1/100, ls=':', c='gray')
    if acc2 is not None:
        ax[1].axhline(y=acc2/100, ls=':', c='gray')

    if despine:
        for i in range(2):
            sns.despine(ax=ax[i], left=despine_all, bottom=despine_all)
    fig.tight_layout()

    if save_prefix is not None:
        if loss_lim is not None or acc_lim is not None:
            fig.savefig(f'{save_prefix}mul_loss_acc_zoom.png', dpi=dpi)
        else:
            fig.savefig(f'{save_prefix}mul_loss_acc.png', dpi=dpi)
        plt.close()
    else:
        plt.show()


def plot_compare_loss_acc(df, save_prefix=None, loss_lim=None,
                          acc_lim=None, point=True, acc_train=False, ylabels=None):
    y_acc = 'train acc'if acc_train else 'test acc'
    fig, ax = plt.subplots(1, 2, figsize=(7, 5))
    ax1 = sns.boxplot(x='method', y='train loss', data=df, ax=ax[0], width=.5, palette="pastel")
    ax2 = sns.boxplot(x='method', y=y_acc, data=df, ax=ax[1], width=.5, palette="pastel")
    if point:
        ax1 = sns.stripplot(x='method', y='train loss', data=df, size=4, color='.3', ax=ax1,
                            dodge=True, palette="pastel", linewidth=1.)
        ax2 = sns.stripplot(x='method', y=y_acc, data=df, size=4, color='.3', ax=ax2,
                            dodge=True, palette="pastel", linewidth=1.)

    for a in ax:
        sns.despine(ax=a)
        a.set_xlabel('')

    if ylabels is not None:
        ax1.set_ylabel(ylabels[0])
        ax2.set_ylabel(ylabels[1])

    fig.tight_layout()

    if loss_lim is not None: ax[0].set_ylim(loss_lim)
    if acc_lim is not None: ax[1].set_ylim(acc_lim)

    if save_prefix is not None:
        if loss_lim is not None or acc_lim is not None:
            fig.savefig(f'{save_prefix}comp_loss_acc_zoom.png')
        else:
            fig.savefig(f'{save_prefix}comp_loss_acc.png')
        plt.close()
    else:
        plt.show()


def plot_compare_mul_loss_acc(df, save_prefix=None, loss_lim=None, figsize=None,
                              acc_lim=None, point=True, acc_train=False, legend=True,
                              xticks_rotation=None, despine_all=False, ylabels=None,
                              dpi=150, adjust_params=None, ax1_xticks=False):
    y_acc = 'train acc'if acc_train else 'test acc'
    if figsize is None:
        figsize = (8, 10)

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    ax1 = sns.boxplot(x='net', y='train loss', data=df, ax=ax[0], width=.6, palette="pastel", hue='method')
    ax2 = sns.boxplot(x='net', y=y_acc, data=df, ax=ax[1], width=.6, palette="pastel", hue='method')
    if point:
        ax1 = sns.stripplot(x='net', y='train loss', data=df, size=4, color='.3',
                            ax=ax1, hue='method', dodge=True, palette="pastel", linewidth=1.)
        ax2 = sns.stripplot(x='net', y=y_acc, data=df, size=4, color='.3',
                            ax=ax2, hue='method', dodge=True, palette="pastel", linewidth=1.)

    for a in ax:
        sns.despine(ax=a, left=despine_all, bottom=despine_all)
        if not legend:
            a.get_legend().remove()

    if not ax1_xticks:
        ax1.set_xticks([])
    ax1.set_xlabel('')
    ax2.set_xlabel('model')
    if ylabels is not None:
        ax1.set_ylabel(ylabels[0])
        ax2.set_ylabel(ylabels[1])

    if xticks_rotation is not None:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=xticks_rotation)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=xticks_rotation)

    if loss_lim is not None: ax1.set_ylim(loss_lim)
    if acc_lim is not None: ax2.set_ylim(acc_lim)
    if adjust_params is not None: plt.subplots_adjust(adjust_params)
    fig.tight_layout()

    if save_prefix is not None:
        if loss_lim is not None or acc_lim is not None:
            fig.savefig(f'{save_prefix}comp_mul_loss_acc_zoom.png', dpi=dpi)
        else:
            fig.savefig(f'{save_prefix}comp_mul_loss_acc.png', dpi=dpi)
        plt.close()
    else:
        plt.show()


def plot_train_time(df, acc1, acc2, capsize=.2, errwidth=2, palette=None, xlim=None,
                    max_y=None, save_prefix=None, figsize=None, despine_all=False, dpi=150):
    if max_y is None:
        max_y = int(max(df.loc[:, 'train time (s)'])) + 1
    if figsize is None:
        figsize = (7, 5)
    if palette is None:
        palette = 'pastel' # 'vlag'

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    sns.barplot(x='method', y='train time (s)', data=df, ax=ax[0], capsize=capsize,
                errwidth=errwidth, palette=palette)
    sns.barplot(x='method', y=f'time for {acc1}% (s)', data=df, ax=ax[1], capsize=capsize,
                errwidth=errwidth, palette=palette)
    sns.barplot(x='method', y=f'time for {acc2}% (s)', data=df, ax=ax[2], capsize=capsize,
                errwidth=errwidth, palette=palette)

    ylabels = ['training time (s)',
               f'time required to reach {acc1}% accuracy (s)',
               f'time required to reach {acc2}% accuracy (s)']
    for i in range(3):
        sns.despine(ax=ax[i], left=despine_all, bottom=despine_all)
        ax[i].set_xlabel('')
        ax[i].set_ylim(0, max_y)
        ax[i].set_ylabel(ylabels[i])
        if xlim is not None:
            ax[i].set_xlim(xlim)

    fig.tight_layout()
    if save_prefix is not None:
        fig.savefig(f'{save_prefix}train_time.png', dpi=dpi)
        plt.close()
    else:
        plt.show()
