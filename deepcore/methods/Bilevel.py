from .CoresetMethod import CoresetMethod
from torch.autograd import grad
from scipy.sparse.linalg import cg, LinearOperator
import torch, copy
import numpy as np
from jax.api import jit
from neural_tangents import stax


# Acknowlegement to
# https://github.com/zalanborsos/bilevel_coresets

def cross_entropy(K, alpha, y, weights, lmbda):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
    if lmbda > 0:
        loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss_value


def weighted_mse(K, alpha, y, weights, lmbda):
    loss = torch.mean(torch.sum((torch.matmul(K, alpha) - y) ** 2, dim=1) * weights)
    if lmbda > 0:
        loss += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss


class Bilevel(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, outer_loss_fn="cross_entropy",
                 inner_loss_fn="cross_entropy", max_outer_it=40, max_inner_it=300, outer_lr=0.01,
                 inner_lr=0.25, max_conj_grad_it=50, candidate_batch_size=200, div_tol=10, print_interval=10, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)

        self.outer_loss_fn = cross_entropy if outer_loss_fn == 'cross_entropy' else weighted_mse
        self.inner_loss_fn = cross_entropy if inner_loss_fn == 'cross_entropy' else weighted_mse
        self.out_dim = self.args.num_classes
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.max_conj_grad_it = max_conj_grad_it
        self.candidate_batch_size = candidate_batch_size
        self.div_tol = div_tol
        self.print_interval = print_interval

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

    def hvp(self, loss, params, v):
        dl_p = self.flat_grad(grad(loss, params, create_graph=True, retain_graph=True))
        return self.flat_grad(grad(dl_p, params, grad_outputs=v, retain_graph=True), reshape=True, detach=True)

    def inverse_hvp(self, loss, params, v):
        # TODO: refactor this to perform cg in pytorch
        op = LinearOperator((len(v), len(v)),
                            matvec=lambda x: self.hvp(loss, params,
                                                      torch.from_numpy(x).to(loss.device).float()).cpu().numpy())
        return torch.from_numpy(cg(op, v.cpu().numpy(), maxiter=self.max_conj_grad_it)[0]).float().to(loss.device)

    def implicit_grad_batch(self, inner_loss, outer_loss, weights, params):
        dg_dalpha = self.flat_grad(grad(outer_loss, params), detach=True) * 1e-3
        ivhp = self.inverse_hvp(inner_loss, params, dg_dalpha)
        dg_dtheta = self.flat_grad(grad(inner_loss, params, create_graph=True, retain_graph=True))
        return -self.flat_grad(grad(dg_dtheta, weights, grad_outputs=ivhp), detach=True)

    def solve_bilevel_opt_representer_proxy(self, K_X_S, K_S_S, y_X, y_S, data_weights, inner_reg):
        m = K_S_S.shape[0]

        # create the weight tensor
        weights = torch.ones([m], dtype=torch.float, requires_grad=True)
        outer_optimizer = torch.optim.Adam([weights], lr=self.outer_lr)

        # initialize the representer coefficients
        alpha = torch.randn(size=[m, self.out_dim], requires_grad=True)
        alpha.data *= 0.01
        for outer_it in range(self.max_outer_it):
            # perform inner opt
            outer_optimizer.zero_grad()
            inner_loss = np.inf
            while inner_loss > self.div_tol:

                def closure():
                    inner_optimizer.zero_grad()
                    inner_loss = self.inner_loss_fn(K_S_S, alpha, y_S, weights, inner_reg)
                    inner_loss.backward()
                    return inner_loss

                inner_optimizer = torch.optim.LBFGS([alpha], lr=self.inner_lr, max_iter=self.max_inner_it)

                inner_optimizer.step(closure)
                inner_loss = self.inner_loss_fn(K_S_S, alpha, y_S, weights, inner_reg)
                if inner_loss > self.div_tol:
                    # reinitialize upon divergence
                    print("Warning: inner opt diverged, try setting lower inner learning rate.")
                    alpha = torch.randn(size=[m, self.out_dim], requires_grad=True)
                    alpha.data *= 0.01

            # calculate outer loss
            outer_loss = self.outer_loss_fn(K_X_S, alpha, y_X, data_weights, 0)

            # calculate the implicit gradient
            weights._grad.data = self.implicit_grad_batch(inner_loss, outer_loss, weights, alpha).clamp_(-1, 1)
            outer_optimizer.step()

            # project weights to ensure positivity
            weights.data = torch.max(weights.data, torch.zeros(m).float())

        return weights, alpha, outer_loss, inner_loss

    def build_with_representer_proxy_batch(self, X, y, m, kernel_fn_np, data_weights=None,
                                           cache_kernel=False, start_size=1, inner_reg=1e-4):
        """Build a coreset of size m based on (X, y, weights).
       Args:
           X (np.ndarray or torch.Tensor): array of the data, its type depends on the kernel function you use
           y (np.ndarray or torch.Tensor): labels, np.ndarray or torch.Tensor of type long (for classification)
               or float (for regression)
           m (int): size of the coreset
           kernel_fn_np (function): kernel function of the proxy model
           data_weights (np.ndarray): weights of X
           cache_kernel (bool): if True, the Gram matrix is calculated and saved at start. Use 'True' only on small
                datasets.
           start_size (int): number of coreset points chosen at random at the start
       Returns:
           (coreset_inds, coreset_weights): coreset indices and weights
       """
        n = X.shape[0]
        selected_inds = np.random.choice(n, start_size, replace=None)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(data_weights, np.ndarray):
            data_weights = torch.from_numpy(data_weights).float()
        elif data_weights is None:
            data_weights = torch.ones(n).float()
        if m >= X.shape[0]:
            return np.arange(X.shape[0]), np.ones(X.shape[0])

        kernel_fn = lambda x, y: torch.from_numpy(kernel_fn_np(x, y)).float()

        if cache_kernel:
            K = kernel_fn(X, X)

        def calc_kernel(inds1, inds2):
            if cache_kernel:
                return K[inds1][:, inds2]
            else:
                return kernel_fn(X[inds1], X[inds2])

        for i in range(start_size - 1, m):
            # calculate the kernel between the data and the selected points
            K_X_S = calc_kernel(np.arange(n), selected_inds)

            # calculate the kernel between the selected points
            K_S_S = K_X_S[selected_inds]

            # solve bilevel opt on current set S
            coreset_weights, alpha, outer_loss, inner_loss = self.solve_bilevel_opt_representer_proxy(K_X_S, K_S_S, y,
                                                                                                      y[selected_inds],
                                                                                                      data_weights,
                                                                                                      inner_reg)

            # generate candidate inds
            candidate_inds = np.setdiff1d(np.arange(n), selected_inds)
            candidate_inds = np.random.choice(candidate_inds,
                                              np.minimum(self.candidate_batch_size, len(candidate_inds)),
                                              replace=False)
            all_inds = np.concatenate((selected_inds, candidate_inds))
            new_size = len(all_inds)

            K_X_S = calc_kernel(np.arange(n), all_inds)
            K_S_S = K_X_S[all_inds]

            weights_all = torch.zeros([new_size], requires_grad=True)
            weights_all.data[:i + 1] = copy.deepcopy(coreset_weights.data)
            alpha_all = torch.zeros([new_size, self.out_dim], requires_grad=True)
            alpha_all.data[:i + 1] = copy.deepcopy(alpha.data)
            inner_loss = self.inner_loss_fn(K_S_S, alpha_all, y[all_inds], weights_all, inner_reg)
            outer_loss = self.outer_loss_fn(K_X_S, alpha_all, y, data_weights, 0)

            weights_all_grad = self.implicit_grad_batch(inner_loss, outer_loss, weights_all, alpha_all)

            # choose point with the highest negative gradient
            chosen_ind = weights_all_grad[i + 1:].argsort()[0]
            chosen_ind = candidate_inds[chosen_ind]
            selected_inds = np.append(selected_inds, chosen_ind)
            if (i + 1) % self.print_interval == 0:
                print('Coreset size {}, outer_loss {:.3}, inner loss {:.3}'.format(i + 1, outer_loss, inner_loss))

        return selected_inds[:-1], coreset_weights.detach().cpu().numpy()

    def select(self, **kwargs):
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(100, 1., 0.05),
            stax.Relu(),
            stax.Dense(100, 1., 0.05),
            stax.Relu(),
            stax.Dense(10, 1., 0.05))
        fnn_kernel_fn = jit(kernel_fn, static_argnums=(2,))

        _, _, kernel_fn = stax.serial(
            stax.Conv(32, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
            stax.Relu(),
            stax.Conv(64, (5, 5), (1, 1), padding='SAME', W_std=1., b_std=0.05),
            stax.Relu(),
            stax.Flatten(),
            stax.Dense(128, 1., 0.05),
            stax.Relu(),
            stax.Dense(10, 1., 0.05))
        cnn_kernel_fn = jit(kernel_fn, static_argnums=(2,))

        def generate_fnn_ntk(X, Y):
            return np.array(fnn_kernel_fn(X, Y, 'ntk'))

        def generate_cnn_ntk(X, Y):
            n = X.shape[0]
            m = Y.shape[0]
            K = np.zeros((n, m))
            for i in range(m):
                K[:, i:i + 1] = np.array(cnn_kernel_fn(X, Y[i:i + 1], 'ntk'))
            return K

        def ResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
            Main = stax.serial(
                stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
                stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
            Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
                channels, (3, 3), strides, padding='SAME')
            return stax.serial(stax.FanOut(2),
                               stax.parallel(Main, Shortcut),
                               stax.FanInSum())

        def ResnetGroup(n, channels, strides=(1, 1)):
            blocks = []
            blocks += [ResnetBlock(channels, strides, channel_mismatch=True)]
            for _ in range(n - 1):
                blocks += [ResnetBlock(channels, (1, 1))]
            return stax.serial(*blocks)

        def Resnet(block_size, num_classes):
            return stax.serial(
                stax.Conv(64, (3, 3), padding='SAME'),
                ResnetGroup(block_size, 64),
                ResnetGroup(block_size, 128, (2, 2)),
                ResnetGroup(block_size, 256, (2, 2)),
                ResnetGroup(block_size, 512, (2, 2)),
                stax.Flatten(),
                stax.Dense(num_classes, 1., 0.05))

        _, _, resnet_kernel_fn = Resnet(block_size=2, num_classes=10)
        resnet_kernel_fn = jit(resnet_kernel_fn, static_argnums=(2,))

        def generate_resnet_ntk(X, Y, skip=25):
            n = X.shape[0]
            m = Y.shape[0]
            K = np.zeros((n, m))
            for i in range(0, m, skip):
                K[:, i:i + skip] = np.array(resnet_kernel_fn(X, Y[i:i + skip], 'ntk'))
            return K / 100

        if self.args.model.startswith("ResNet"):
            kernel_fn = lambda x, y: generate_resnet_ntk(x.transpose(0, 2, 3, 1), y.transpose(0, 2, 3, 1), skip=20)
        else:
            kernel_fn = lambda x, y: generate_cnn_ntk(
                x.reshape(-1, self.args.im_size[0], self.args.im_size[0], self.args.channel),
                y.reshape(-1, self.args.im_size[0], self.args.im_size[0], self.args.channel))

        inds, weights = self.build_with_representer_proxy_batch(self.dst_train.train_data.cpu().numpy(),
                                                                self.dst_train.targets.cpu().numpy(), self.coreset_size,
                                                                kernel_fn,
                                                                cache_kernel=True,
                                                                start_size=10, inner_reg=1e-7)

        return inds, weights
