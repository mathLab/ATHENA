"""
Module for Nonlinear Level-set Learning (NLL).

:References:

    - Guannan Zhang, Jiaxin Zhang, Jacob Hinkle.
      Learning nonlinear level sets for dimensionality reduction in function
      approximation. NeurIPS 2019, 13199-13208.
      arxiv: https://arxiv.org/abs/1902.10652

"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the overall data type
torch.set_default_tensor_type(torch.DoubleTensor)


class NonlinearLevelSet():
    """Nonlinear Level Set class. It is implemented as a Reversible neural
    networks (RevNet).

    :param int n_layers: number of layers of the RevNet.
    :param int active_dim: number of active dimensions.
    :param float lr: learning rate.
    :param int epochs: number of ephocs.
    :param float dh: so-called time step of the RevNet. Default is 0.25.
    :param `torch.optim.Optimizer` optimizer: optimizer used in the training of
        the RevNet. Its argument are passed in the dict optim_args when
        :py:meth:`train` is called.
    :param `torch.optim.lr_scheduler._LRScheduler` scheduler: scheduler used in
        the training of the RevNet. Its argument are passed in the dict
        scheduler_args when :py:meth:`train` is called. Default is None.

    :cvar `BackwardNet` backward: backward net of the RevNet. See
        :class:`BackwardNet` class in :py:mod:`nll` module.
    :cvar `ForwardNet` forward: forward net of the RevNet. See
        :class:`ForwardNet` class in :py:mod:`nll` module.
    :cvar list loss_vec: list containg the loss at every epoch.
    """
    def __init__(self,
                 n_layers,
                 active_dim,
                 lr,
                 epochs,
                 dh=0.25,
                 optimizer=optim.Adam,
                 scheduler=None):
        self.n_layers = n_layers
        self.active_dim = active_dim
        self.lr = lr
        self.epochs = epochs
        self.dh = dh
        if issubclass(optimizer, optim.Optimizer):
            self.optimizer = optimizer

        if scheduler and issubclass(scheduler,
                                    torch.optim.lr_scheduler._LRScheduler):
            self.scheduler = scheduler
        else:
            self.scheduler = None

        self.backward = None
        self.forward = None
        self.loss_vec = []

    def train(self,
              inputs,
              gradients,
              outputs=None,
              interactive=False,
              target_loss=0.0001,
              optim_args=None,
              scheduler_args=None):
        """
        Train the whole RevNet.

        :param torch.Tensor inputs: DoubleTensor n_samples-by-n_params
            containing the points in the full input space.
        :param torch.Tensor gradients: DoubleTensor n_samples-by-n_params
            containing the gradient samples wrt the input parameters.
        :param numpy.ndarray outputs: array n_samples-by-1 containing the
            corresponding function evaluations. Needed only for the interactive
            mode. Default is None.
        :param bool interactive: if True a plot with the loss function decay,
            and the sufficient summary plot will be showed and updated every
            10 epochs, and at the last epoch. Default is False.
        :param float target_loss: loss threshold. Default is 0.0001.
        :param dict optim_args: dictionary passed to the optimizer.
        :param dict scheduler_args: dictionary passed to the scheduler.
        :raises: ValueError: in interactive mode outputs must be provided for
            the sufficient summary plot.
        """
        if optim_args is None:
            optim_args = {}
        if scheduler_args is None:
            scheduler_args = {}
        if inputs.shape[1] % 2 != 0:
            raise ValueError('The parameter space\'s dimension must be even.')

        if interactive:
            if outputs is None:
                raise ValueError(
                    'outputs in interactive mode have to be provided!')
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title('Loss function decay')
            ax2.grid(linestyle='dotted')
            plt.ion()
            plt.show()

        # Build the forward network
        self.forward = ForwardNet(n_params=inputs.shape[1],
                                  n_layers=self.n_layers,
                                  dh=self.dh,
                                  active_dim=self.active_dim)
        # Initialize the gradient
        self.forward.zero_grad()
        optimizer = self.optimizer(self.forward.parameters(), self.lr,
                                   **optim_args)

        # Initialize scheduler if present
        if self.scheduler:
            sched = self.scheduler(optimizer, **scheduler_args)

        # Training
        for i in range(self.epochs):
            optimizer.zero_grad()
            mapped_inputs = self.forward(inputs)
            loss = self.forward.customized_loss(inputs, mapped_inputs,
                                                gradients)

            if i % 10 == 0 or i == self.epochs - 1:
                print(f'epoch = {i}, loss = {loss}')
                if interactive:
                    ax1.cla()
                    ax1.set_title('Sufficient summary plot')
                    ax1.plot(mapped_inputs.detach().numpy()[:, 0], outputs,
                             'bo')
                    ax1.grid(linestyle='dotted')
                    ax2.plot(i, loss.detach().numpy(), 'ro')
                    plt.draw()
                    plt.pause(0.00001)

                self.loss_vec.append(loss.detach().numpy())
                # Build the inverse network based on the trained forward network
                self.backward = BackwardNet(n_params=inputs.shape[1],
                                            n_layers=self.n_layers,
                                            dh=self.dh)
                self.backward.zero_grad()

                for j in range(self.backward.n_layers):
                    name_y = f'fc{j + 1}_y'
                    name_z = f'fc{j + 1}_z'
                    getattr(self.backward, name_y).weight = torch.nn.Parameter(
                        getattr(self.forward, name_y).weight)
                    getattr(self.backward, name_z).weight = torch.nn.Parameter(
                        getattr(self.forward, name_z).weight)
                    getattr(self.backward, name_y).bias = torch.nn.Parameter(
                        getattr(self.forward, name_y).bias)
                    getattr(self.backward, name_z).bias = torch.nn.Parameter(
                        getattr(self.forward, name_z).bias)

                if torch.mean(
                        torch.abs(
                            torch.add(-1 * inputs,
                                      self.backward(
                                          self.forward(inputs))))) > 1e-5:
                    print('self.backward is wrong!')
                    print(
                        torch.mean(
                            torch.abs(
                                torch.add(-1 * inputs,
                                          self.backward(
                                              self.forward(inputs))))))

                if loss < target_loss:
                    break

            loss.backward()
            optimizer.step()

            if self.scheduler:
                sched.step()

        if interactive:
            plt.ioff()
            plt.show()

    def plot_sufficient_summary(self,
                                inputs,
                                outputs,
                                filename=None,
                                figsize=(10, 8),
                                title=''):
        """
        Plot the sufficient summary.

        :param torch.Tensor inputs: DoubleTensor n_samples-by-n_params
            containing the points in the full input space.
        :param numpy.ndarray outputs: array n_samples-by-1 containing the
            corresponding function evaluations.
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Defaults to (10, 8).
        :param str title: title of the plot.
        :raises: ValueError

        .. warning::
            Plot only available for active dimensions up to 1.
        """
        plt.figure(figsize=figsize)
        plt.title(title)

        if self.active_dim == 1:
            reduced_inputs = self.forward(inputs)[:, 0]
            plt.plot(reduced_inputs.detach().numpy(), outputs, 'bo')
            plt.xlabel('Reduced input')
            plt.ylabel('Output')
        else:
            raise ValueError(
                'Sufficient summary plots cannot be made in more than 1 ' \
                'dimension.'
            )

        plt.grid(linestyle='dotted')

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_loss(self, filename=None, figsize=(10, 8), title=''):
        """
        Plot the loss function decay.

        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Defaults to (10, 8).
        :param str title: title of the plot.
        """
        plt.figure(figsize=figsize)
        plt.title(title)
        x_range = list(range(1, self.epochs + 1, 10)) + [self.epochs]
        plt.plot(x_range, self.loss_vec, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(linestyle='dotted')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def save_forward(self, outfile):
        """
        Save the forward map for future inference.

        :param str outfile: filename of the net to save.
            Use either .pt or .pth. See notes below.

        .. note::
            A common PyTorch convention is to save models using either a .pt or
            .pth file extension.
        """
        torch.save(self.forward.state_dict(), outfile)

    def load_forward(self, infile, n_params):
        """
        Load the forward map for inference.

        :param str infile: filename of the saved net to load. See notes below.
        :param int n_params: number of input parameters.

        .. note::
            A common PyTorch convention is to save models using either a .pt or
            .pth file extension.
        """
        self.forward = ForwardNet(n_params, self.n_layers, self.dh,
                                  self.active_dim)
        self.forward.load_state_dict(torch.load(infile))
        self.forward.eval()

    def save_backward(self, outfile):
        """
        Save the backward map for future inference.

        :param str outfile: filename of the net to save.
            Use either .pt or .pth. See notes below.

        .. note::
            A common PyTorch convention is to save models using either a .pt or
            .pth file extension.
        """
        torch.save(self.backward.state_dict(), outfile)

    def load_backward(self, infile, n_params):
        """
        Load the backward map for inference.

        :param str infile: filename of the saved net to load. See notes below.
        :param int n_params: number of input parameters.

        .. note::
            A common PyTorch convention is to save models using either a .pt or
            .pth file extension.
        """
        self.backward = BackwardNet(n_params, self.n_layers, self.dh)
        self.backward.load_state_dict(torch.load(infile))
        self.backward.eval()


class ForwardNet(nn.Module):
    """
    Forward net class. It is part of the RevNet.

    :param int n_params: number of input parameters.
    :param int n_layers: number of layers of the RevNet.
    :param float dh: so-called time step of the RevNet.
    :param int active_dim: number of active dimensions.

    :cvar slice omega: a slice object indicating the active dimension to keep.
        For example to keep the first two dimension `omega = slice(2)`. It is
        automatically set with `active_dim`.
    """
    def __init__(self, n_params, n_layers, dh, active_dim):
        super().__init__()
        self.n_params = n_params // 2
        self.n_layers = n_layers
        self.dh = dh
        self.omega = slice(active_dim)

        for i in range(self.n_layers):
            setattr(self, f'fc{i + 1}_y',
                    nn.Linear(self.n_params, 2 * self.n_params))
            setattr(self, f'fc{i + 1}_z',
                    nn.Linear(self.n_params, 2 * self.n_params))

    def forward(self, inputs):
        """
        Maps original inputs to transformed inputs.

        :param torch.Tensor inputs: DoubleTensor n_samples-by-n_params
            containing the points in the original full input space.
        :return mapped_inputs: DoubleTensor n_samples-by-n_params with
            the nonlinear transformed inputs.
        :rtype: torch.Tensor
        """
        bb = torch.split(inputs, self.n_params, dim=1)
        vars()['var0_y'] = torch.clone(bb[0])
        vars()['var0_z'] = torch.clone(bb[1])

        for i in range(self.n_layers):
            name_y = f'fc{i + 1}_y'
            name_z = f'fc{i + 1}_z'
            var_y0 = f'var{i}_y'
            var_z0 = f'var{i}_z'
            var_y1 = f'var{i + 1}_y'
            var_z1 = f'var{i + 1}_z'

            sig_y = torch.unsqueeze(
                torch.tanh(getattr(self, name_y)(vars()[var_z0])), 2)
            K_y = torch.transpose(getattr(self, name_y).weight, 0, 1)
            vars()[var_y1] = torch.add(
                vars()[var_y0],
                1 * self.dh * torch.squeeze(torch.matmul(K_y, sig_y), 2))

            sig_z = torch.unsqueeze(
                torch.tanh(getattr(self, name_z)(vars()[var_y1])), 2)
            K_z = torch.transpose(getattr(self, name_z).weight, 0, 1)
            vars()[var_z1] = torch.add(
                vars()[var_z0],
                -1 * self.dh * torch.squeeze(torch.matmul(K_z, sig_z), 2))

        return torch.cat((vars()[var_y1], vars()[var_z1]), 1)

    def customized_loss(self, inputs, mapped_inputs, gradients):
        """
        Custom loss function.

        :param torch.Tensor inputs: DoubleTensor n_samples-by-n_params
            containing the points in the full input space.
        :param torch.Tensor mapped_inputs: DoubleTensor
            n_samples-by-n_params containing the mapped points in the
            full input space. They are the result of the forward application.
        :param torch.Tensor gradients: DoubleTensor n_samples-by-n_params
            containing the gradient samples wrt the input parameters.
        """
        # Define the weights and bias of the inverse network
        for i in range(self.n_layers):
            name_y = f'fc{i + 1}_y'
            name_z = f'fc{i + 1}_z'

            inv_name_y_weight = f'inv_fc{i + 1}_y_weight'
            inv_name_z_weight = f'inv_fc{i + 1}_z_weight'
            vars()[inv_name_y_weight] = getattr(self, name_y).weight
            vars()[inv_name_z_weight] = getattr(self, name_z).weight

            inv_name_y_bias = f'inv_fc{i + 1}_y_bias'
            inv_name_z_bias = f'inv_fc{i + 1}_z_bias'
            vars()[inv_name_y_bias] = getattr(self, name_y).bias
            vars()[inv_name_z_bias] = getattr(self, name_z).bias

        Jacob = torch.empty(inputs.size()[0], 2 * self.n_params,
                            2 * self.n_params)

        for j in range(2 * self.n_params):
            output_dy = torch.clone(mapped_inputs)
            output_dy[:, j] += 0.001

            bb = torch.split(output_dy, self.n_params, dim=1)
            var_y0 = f'var{self.n_layers - 1}_y'
            var_z0 = f'var{self.n_layers - 1}_z'
            vars()[var_y0] = bb[0]
            vars()[var_z0] = bb[1]

            for i in range(self.n_layers - 1, -1, -1):
                inv_name_y_weight = f'inv_fc{i + 1}_y_weight'
                inv_name_z_weight = f'inv_fc{i + 1}_z_weight'
                inv_name_y_bias = f'inv_fc{i + 1}_y_bias'
                inv_name_z_bias = f'inv_fc{i + 1}_z_bias'
                var_y0 = f'var{i}_y'
                var_z0 = f'var{i}_z'
                var_y1 = f'var{i - 1}_y'
                var_z1 = f'var{i - 1}_z'

                sig_z = torch.tanh(
                    torch.add(
                        torch.matmul(vars()[inv_name_z_weight],
                                     torch.unsqueeze(vars()[var_y0], 2)),
                        torch.unsqueeze(vars()[inv_name_z_bias], 1)))
                K_z = torch.transpose(vars()[inv_name_z_weight], 0, 1)
                vars()[var_z1] = torch.add(
                    vars()[var_z0],
                    1 * self.dh * torch.squeeze(torch.matmul(K_z, sig_z), 2))

                sig_y = torch.tanh(
                    torch.add(
                        torch.matmul(vars()[inv_name_y_weight],
                                     torch.unsqueeze(vars()[var_z1], 2)),
                        torch.unsqueeze(vars()[inv_name_y_bias], 1)))
                K_y = torch.transpose(vars()[inv_name_y_weight], 0, 1)
                vars()[var_y1] = torch.add(
                    vars()[var_y0],
                    -1 * self.dh * torch.squeeze(torch.matmul(K_y, sig_y), 2))

            dx = torch.cat((vars()[var_y1], vars()[var_z1]), 1)

            # # Test the invertibility
            # # Warning: it is computational intense
            # if torch.mean(torch.abs(torch.add(-1 * output_dy,
            #                                   self.forward(dx)))) > 1e-5:
            #     print('Something is wrong in Jacobian computation')
            #     print(torch.mean(
            #         torch.abs(torch.add(-1 * output_dy, self.forward(dx)))))

            for k in range(2 * self.n_params):
                Jacob[:, j, k] = torch.add(dx[:, k], -1 * inputs[:, k])

        JJ2 = torch.unsqueeze(torch.sqrt(torch.sum(torch.mul(Jacob, Jacob), 2)),
                              2)
        JJJ = torch.div(Jacob, JJ2.expand(-1, -1, 2 * self.n_params))
        ex_data = torch.unsqueeze(gradients, 2)
        loss_weights = torch.clone(torch.squeeze(torch.matmul(JJJ, ex_data), 2))
        # anisotropy weigths
        loss_weights[:, self.omega] = 0.0
        loss_anisotropy = torch.sqrt(
            torch.mean(torch.sum(torch.mul(loss_weights, loss_weights), 1)))

        J_det = torch.empty(inputs.shape[0])
        for k in range(inputs.shape[0]):
            eee = torch.svd(JJJ[k, :, :])[1]
            J_det[k] = torch.prod(eee)
        loss_det = torch.abs(torch.prod(J_det - 1.0))
        return loss_anisotropy + loss_det


class BackwardNet(nn.Module):
    """Backward Net class. It is part of the RevNet.

    :param int n_params: number of input parameters.
    :param int n_layers: number of layers of the RevNet.
    :param float dh: so-called time step of the RevNet.
    """
    def __init__(self, n_params, n_layers, dh):
        super().__init__()
        self.n_params = n_params // 2
        self.n_layers = n_layers
        self.dh = dh

        for i in range(self.n_layers):
            setattr(self, f'fc{i + 1}_y',
                    nn.Linear(self.n_params, 2 * self.n_params))
            setattr(self, f'fc{i + 1}_z',
                    nn.Linear(self.n_params, 2 * self.n_params))

    def forward(self, mapped_inputs):
        """
        Maps transformed inputs to original inputs.

        :param torch.Tensor mapped_inputs: DoubleTensor n_samples-by-n_params
            containing the nonlinear transformed inputs.
        :return inputs: DoubleTensor n_samples-by-n_params with
            the points in the original full input space.
        :rtype: torch.Tensor
        """
        y, z = torch.split(mapped_inputs, self.n_params, dim=1)

        for i in range(self.n_layers - 1, -1, -1):
            name_y = f'fc{i + 1}_y'
            name_z = f'fc{i + 1}_z'

            sig_z = torch.unsqueeze(torch.tanh(getattr(self, name_z)(y)), 2)
            K_z = torch.transpose(getattr(self, name_z).weight, 0, 1)
            z = torch.add(
                z, 1 * self.dh * torch.squeeze(torch.matmul(K_z, sig_z), 2))

            sig_y = torch.unsqueeze(torch.tanh(getattr(self, name_y)(z)), 2)
            K_y = torch.transpose(getattr(self, name_y).weight, 0, 1)
            y = torch.add(
                y, -1 * self.dh * torch.squeeze(torch.matmul(K_y, sig_y), 2))

        return torch.cat((y, z), 1)
