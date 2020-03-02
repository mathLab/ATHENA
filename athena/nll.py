"""[summary]

[description]
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the overall data type
torch.set_default_tensor_type(torch.DoubleTensor)

class NonlinearLevelSet(object):
    """Nonlinear Level Set class
    
    [description]
    """
    def __init__(self, n_layers, active_dim, lr, epochs, dh=0.25):
        self.n_layers = n_layers
        self.active_dim = active_dim
        self.lr = lr
        self.epochs = epochs
        self.dh = dh
        self.forward = None
        self.backward = None
        self.loss_vec = []

    def train(self, inputs, gradients, outputs=None, interactive=False):
        """
        all the input paraemters have to be torch
        """
        if interactive:
            if outputs is None:
                raise ValueError('outputs in interactive mode have to be provided!')
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
                                  n_train=inputs.shape[0],
                                  active_dim=self.active_dim)
        # Initialize the gradient
        self.forward.zero_grad()
        optimizer = optim.SGD(self.forward.parameters(), lr=self.lr)
        
        # Training 
        for i in range(self.epochs):
            optimizer.zero_grad()
            mapped_inputs = self.forward(inputs)
            loss = self.forward.customized_loss_diff(inputs, mapped_inputs, gradients)
            
            if i % 10 == 0:
                print('epoch = {}, loss = {}'.format(i, loss))
                if interactive:
                    ax1.cla()
                    ax1.set_title('Sufficient summary plot')
                    ax1.plot(mapped_inputs.detach().numpy()[:, 0], outputs, 'bo')
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
                    name_y = 'fc' + str(j+1) + '_y'
                    name_z = 'fc' + str(j+1) + '_z'
                    getattr(self.backward, name_y).weight = torch.nn.Parameter(getattr(self.forward, name_y).weight)
                    getattr(self.backward, name_z).weight = torch.nn.Parameter(getattr(self.forward, name_z).weight)
                    getattr(self.backward, name_y).bias = torch.nn.Parameter(getattr(self.forward, name_y).bias)
                    getattr(self.backward, name_z).bias = torch.nn.Parameter(getattr(self.forward, name_z).bias)
        
                if torch.mean(torch.abs(torch.add(-1 * inputs, self.backward(self.forward(inputs))))) > 1e-5:
                    print('self.backward is wrong!')
                    print(torch.mean(torch.abs(torch.add(-1 * inputs, self.backward(self.forward(inputs))))))
        
                if loss < 0.0001: 
                    break
        
            loss.backward()
            optimizer.step()
            
        # Build the inverse network based on the trained forward network
        self.backward = BackwardNet(n_params=inputs.shape[1],
                                    n_layers=self.n_layers, 
                                    dh=self.dh)
        self.backward.zero_grad()
        
        for i in range(self.backward.n_layers):
            name_y = 'fc{}_y'.format(i+1)
            name_z = 'fc{}_z'.format(i+1)
            getattr(self.backward, name_y).weight = torch.nn.Parameter(getattr(self.forward, name_y).weight)
            getattr(self.backward, name_z).weight = torch.nn.Parameter(getattr(self.forward, name_z).weight)
            getattr(self.backward, name_y).bias = torch.nn.Parameter(getattr(self.forward, name_y).bias)
            getattr(self.backward, name_z).bias = torch.nn.Parameter(getattr(self.forward, name_z).bias)    
        
        # Test the invertibility of the self.backward
        if torch.mean(torch.abs(torch.add(-1 * inputs, self.backward(self.forward(inputs))))) > 1e-5:
            print('self.backward is wrong!')
            print(torch.mean(torch.abs(torch.add(-1 * inputs, self.backward(self.forward(inputs))))))

        if interactive:
            plt.ioff()
            plt.show()
    
    def plot_sufficient_summary(self, inputs, outputs):
        """
        inputs torch
        outputs numpy
        only for 1 active dim
        """
        reduced_inputs = self.forward(inputs)[:, 0]
        plt.figure()
        plt.title('Sufficient summary plot')
        plt.plot(reduced_inputs.detach().numpy(), outputs, 'bo')
        plt.xlabel('Reduced input')
        plt.xlabel('Output')
        plt.grid(linestyle='dotted')
        plt.show()

    def plot_loss(self):
        """
        """
        plt.figure()
        plt.title('Loss function decay')
        plt.plot(range(1, self.epochs+1, 10), self.loss_vec, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(linestyle='dotted')
        plt.show()

    def save(self, outfile):
        """
        Save the forward map for future inference.

        :param str outfile: filename of the net to save.
            Use either .pt or .pth. See notes below.

        .. note::
            A common PyTorch convention is to save models using either a .pt or
            .pth file extension.
        """
        torch.save(self.forward.state_dict(), outfile)

    def load(self, infile):
        """
        Load the forward map for inference.

        :param str infile: filename of the saved net to load. See notes below.

        .. note::
            A common PyTorch convention is to save models using either a .pt or
            .pth file extension.
        """
        self.forward = ForwardNet(*args, **kwargs)
        self.forward.load_state_dict(torch.load(infile))
        self.forward.eval()


class ForwardNet(nn.Module):
    """
    :cvar slice omega: a slice object indicating the active dimension to keep.
        For example to keep the first two dimension omega=slice(2).
    """
    def __init__(self, n_params, n_layers, dh, n_train, active_dim):
        super().__init__()
        self.n_params = n_params//2
        self.n_layers = n_layers
        self.dh = dh
        self.n_train = n_train
        self.omega = slice(active_dim)

        for i in range(self.n_layers):
            name_y = 'fc{}_y'.format(i+1)
            name_z = 'fc{}_z'.format(i+1)
            setattr(self, name_y, nn.Linear(self.n_params, 2 * self.n_params))
            setattr(self, name_z, nn.Linear(self.n_params, 2 * self.n_params))

    def forward(self, x):
        bb = torch.split(x, self.n_params, dim=1)
        vars()['var0_y'] = torch.clone(bb[0])
        vars()['var0_z'] = torch.clone(bb[1])

        for i in range(self.n_layers):
            name_y = 'fc{}_y'.format(i+1)
            name_z = 'fc{}_z'.format(i+1)
            var_y0 = 'var{}_y'.format(i)
            var_z0 = 'var{}_z'.format(i)
            var_y1 = 'var{}_y'.format(i+1)
            var_z1 = 'var{}_z'.format(i+1)
            
            sig_y = torch.unsqueeze(torch.tanh(getattr(self, name_y)(vars()[var_z0])), 2)
            K_y = torch.transpose(getattr(self, name_y).weight, 0, 1)
            vars()[var_y1] = torch.add(vars()[var_y0],  1 * self.dh * torch.squeeze(torch.matmul(K_y, sig_y), 2))

            sig_z = torch.unsqueeze(torch.tanh(getattr(self, name_z)(vars()[var_y1])), 2)
            K_z = torch.transpose(getattr(self, name_z).weight, 0, 1)
            vars()[var_z1] = torch.add(vars()[var_z0], -1 * self.dh * torch.squeeze(torch.matmul(K_z, sig_z), 2))

        return torch.cat((vars()[var_y1], vars()[var_z1]), 1)


    def customized_loss_diff(self, x, output, grad_data):
        # Define the weights and bias of the inverse network
        for i in range(self.n_layers):
            name_y = 'fc{}_y'.format(i+1)
            name_z = 'fc{}_z'.format(i+1)

            inv_name_y_weight = 'inv_fc{}_y_weight'.format(i+1)
            inv_name_z_weight = 'inv_fc{}_z_weight'.format(i+1)
            vars()[inv_name_y_weight] = getattr(self, name_y).weight
            vars()[inv_name_z_weight] = getattr(self, name_z).weight

            inv_name_y_bias = 'inv_fc{}_y_bias'.format(i+1)
            inv_name_z_bias = 'inv_fc{}_z_bias'.format(i+1)
            vars()[inv_name_y_bias] = getattr(self, name_y).bias
            vars()[inv_name_z_bias] = getattr(self, name_z).bias

        Jacob = torch.empty(x.size()[0], 2 * self.n_params, 2 * self.n_params)

        for j in range(2 * self.n_params):
            output_dy = torch.clone(output)
            output_dy[:, j] += 0.001

            bb = torch.split(output_dy, self.n_params, dim=1)
            var_y0 = 'var{}_y'.format(self.n_layers-1)
            var_z0 = 'var{}_z'.format(self.n_layers-1)
            vars()[var_y0] = bb[0]
            vars()[var_z0] = bb[1]

            for i in range(self.n_layers-1, -1, -1):
                inv_name_y_weight = 'inv_fc{}_y_weight'.format(i+1)
                inv_name_z_weight = 'inv_fc{}_z_weight'.format(i+1)
                inv_name_y_bias = 'inv_fc{}_y_bias'.format(i+1)
                inv_name_z_bias = 'inv_fc{}_z_bias'.format(i+1)
                var_y0 = 'var{}_y'.format(i)
                var_z0 = 'var{}_z'.format(i)
                var_y1 = 'var{}_y'.format(i-1)
                var_z1 = 'var{}_z'.format(i-1)

                sig_z = torch.tanh(torch.add(torch.matmul(vars()[inv_name_z_weight], torch.unsqueeze(vars()[var_y0], 2)), torch.unsqueeze(vars()[inv_name_z_bias], 1)))
                K_z = torch.transpose(vars()[inv_name_z_weight], 0, 1)
                vars()[var_z1] = torch.add(vars()[var_z0], 1 * self.dh * torch.squeeze(torch.matmul(K_z, sig_z), 2))

                sig_y = torch.tanh(torch.add(torch.matmul(vars()[inv_name_y_weight], torch.unsqueeze(vars()[var_z1], 2)), torch.unsqueeze(vars()[inv_name_y_bias], 1)))
                K_y = torch.transpose(vars()[inv_name_y_weight], 0, 1)
                vars()[var_y1] = torch.add(vars()[var_y0], -1 * self.dh * torch.squeeze(torch.matmul(K_y, sig_y), 2))

            dx = torch.cat((vars()[var_y1], vars()[var_z1]), 1)

            # Test the invertibility 
            if torch.mean(torch.abs(torch.add(-1 * output_dy, self.forward(dx)))) > 1e-5:
                print('Something is wrong in Jacobian computation')
                print(torch.mean(torch.abs(torch.add(-1 * output_dy, self.forward(dx)))))

            for k in range(2 * self.n_params):
                Jacob[:, j, k] = torch.add(dx[:, k], -1 * x[:, k])

        ex_data = torch.unsqueeze(grad_data, 2)
        norm_data = torch.sqrt(torch.sum(torch.mul(ex_data, ex_data), 1))

        JJ2 = torch.unsqueeze(torch.sqrt(torch.sum(torch.mul(Jacob, Jacob), 2)), 2)
        JJJ = torch.div(Jacob, JJ2.expand(-1, -1, 2 * self.n_params))
        loss1 = torch.clone(torch.squeeze(torch.matmul(JJJ, ex_data), 2))
        # anisotropy weigths
        loss1[:, self.omega] = 0.0
        loss2 = torch.sqrt(torch.mean(torch.sum(torch.mul(loss1, loss1), 1)))

        J_det = torch.empty(self.n_train)
        for k in range(self.n_train):
            eee = torch.svd(JJJ[k, :, :])[1]
            J_det[k] = torch.prod(eee)
        loss6 = torch.prod(J_det - 1.0)
        return loss2 + loss6


class BackwardNet(nn.Module):
    def __init__(self, n_params, n_layers, dh):
        super().__init__()
        self.n_params = n_params//2
        self.n_layers = n_layers
        self.dh = dh

        for i in range(self.n_layers):
            name_y = 'fc{}_y'.format(i+1)
            name_z = 'fc{}_z'.format(i+1)
            setattr(self, name_y, nn.Linear(self.n_params, 2 * self.n_params))
            setattr(self, name_z, nn.Linear(self.n_params, 2 * self.n_params))

    def forward(self, x):
        y, z = torch.split(x, self.n_params, dim=1)
        
        for i in range(self.n_layers-1, -1, -1):
            name_y = 'fc{}_y'.format(i+1)
            name_z = 'fc{}_z'.format(i+1)

            sig_z = torch.unsqueeze(torch.tanh(getattr(self, name_z)(y)), 2)
            K_z = torch.transpose(getattr(self, name_z).weight, 0, 1) 
            z = torch.add(z,  1 * self.dh * torch.squeeze(torch.matmul(K_z, sig_z), 2))

            sig_y = torch.unsqueeze(torch.tanh(getattr(self, name_y)(z)), 2)
            K_y = torch.transpose(getattr(self, name_y).weight, 0, 1)
            y = torch.add(y, -1 * self.dh * torch.squeeze(torch.matmul(K_y, sig_y), 2))

        return torch.cat((y, z), 1)
