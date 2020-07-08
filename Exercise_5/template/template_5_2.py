from __future__ import print_function
import numpy as np
import tqdm
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F


NEPOCHS = 10
BATCHSIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """The main function of this script."""
    # Test KL implementation
    test_kl(0., 1.)
    test_kl(1., 3.)
    test_kl(-1., 2.)

    # Create model
    vae = VAE()
    vae = vae.to(device=device)
    print(vae)

    # Download Data
    mnist_train, _ = utils.load_fashion_mnist('train')
    mnist_test, _  = utils.load_fashion_mnist('test')

    # Set Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=mnist_train.to(device=device),batch_size=BATCHSIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=mnist_test.to(device=device),batch_size=BATCHSIZE,shuffle=False)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    # Run training
    run_training(vae, train_loader, optimizer, NEPOCHS)
    print('Done training!')

    # Test the model
    run_test(vae, test_loader)

class VAE(nn.Module):
    """VAE class containing the encoder and decoder"""
    def __init__(self):
        super(VAE, self).__init__()
        self.z_dim = 5

        # -------------------------------------
        # TODO: ENTER CODE HERE (Exercise 5.2b)
        # -------------------------------------
        # You can define the network parts using 
        # nn.Sequential(nn.Linear(28*28, 512),...)

    
    def encoder(self, x):
        """Function defining the fully connected encoder.
        Args:
            x (torch.Tensor): batch of data points
        Return:
            mu (torch.Tensor): mean of q(z | x)
            logvar (torch.Tensor): log of variance of q(z | x)
        """
        # -------------------------------------
        # TODO: ENTER CODE HERE (Exercise 5.2b)
        # -------------------------------------
        return mu, logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decoder(self, z, sigmoid=False):
        """Function defining the fully connected decoder.
        Args:
            z (torch.Tensor): batch of latent codes
            sigmoid (bool): wether to use sigmoid or not
        Return:
            x_pred (torch.Tensor): batch of logits for p(x | z)
        """
        # -------------------------------------
        # TODO: ENTER CODE HERE (Exercise 5.2b)
        # -------------------------------------

        if sigmoid:
            x_pred = torch.sigmoid(x_pred)
        return x_pred
    
    def forward(self, x, sigmoid=False):
        mu_z, logvar_z = self.encoder(x)
        z = self.sample(mu_z, logvar_z)
        x_pred = self.decoder(z, sigmoid=sigmoid)
        return x_pred, mu_z, logvar_z

def loss_function(pred, real, mu_z, logvar_z):
    kl = compute_kl(mu_z, logvar_z)
    kl = kl.mean()
    reconstr_err = F.binary_cross_entropy_with_logits(pred, real, reduction='none')
    reconstr_err = reconstr_err.sum(dim=1).mean()
    return reconstr_err + kl, reconstr_err, kl

def run_training(vae, train_loader, optimizer, n_epochs):
    """Run training.

    Args:
        vae (nn.Module): tensorflow Session object
        train_loader (dataloader): iterator over training data
        optimizer: optimizer
        nepochs (int): number of epochs to run training
    """

    for i_epoch in range(n_epochs):
        print('Start epoch %d' % i_epoch)

        for idx, x in enumerate(train_loader):
            vae.train()
            optimizer.zero_grad()

            # Reshape to feed it into fully connected network
            x = x.reshape(-1, 28*28)

            # Execute model
            x_pred, mu_z, logvar_z = vae(x)

            # Derive Losses
            loss_all, reconstr_err_out, kl_out = loss_function(x_pred, x, mu_z, logvar_z) 
            
            # Derive gradients
            loss_all.backward()

            # Apply optimizer
            optimizer.step()

            if (idx % 100) == 0:
                print('[epoch=%d, it=%d] loss = %.4f, kl = %.4f, reconstr_err = %.4f'
                      % (i_epoch, idx, loss_all, kl_out, reconstr_err_out))


def run_test(vae, test_loader):
    """Run simple evaluation for trained model.

    Args:
        vae (nn.Module): tensorflow Session object
        test_loader (dataloader): iterator over training data
    """
    x_in = []
    x_pred_out = []
    losses = []
    vae.eval()

    for idx, x in enumerate(test_loader):
        # Reshape to feed into fully connected network
        x = x.reshape(-1, 28*28)
        # Evaluate loss and reconstruction
        # -------------------------------------------
        # TODO: ENTER CODE HERE (Exercise 5.2c)
        # -------------------------------------
        # compute x_pred, x and the loss and append to lists
        # x_in, x_pred_out, losses
        # -------------------------------------------


    x_in = np.concatenate(x_in, axis=0)
    x_pred_out = np.concatenate(x_pred_out, axis=0)
    x_pred_out = utils.sigmoid(x_pred_out)

    # Get some samples
    # -------------------------------------
    # TODO: Implement code for sampling (Exercise 5.2c)
    # -------------------------------------
    # 1) sample a random vector z (size(20,5)) from a normal distribution
    #    with mean 0 and variance 1
    # 2) use the decoder function of the vae to predict x_rnd_out from z_rand
    # Output: x_rnd_out
    # HINT: Apply the sigmoid function to the output of your network



    # Compute and print Loss and MSE
    print('Loss (test): %.4f +- %.4f' % (np.mean(losses), np.std(losses)))
    print('MSE: %.4f' % np.mean((x_in - x_pred_out)**2))

    # Plot 5 reconstructions and 5 samples
    print('Reconstructions:')
    utils.plot_reconstructions(x_in[:5], x_pred_out[:5], out_path='out/vae')
    print('Random Samples:')
    utils.plot_samples(x_rnd_out[:5].cpu().detach().numpy(), out_path='out/vae')


def compute_kl(mu_z, logvar_z):
    """Compute KL-divergence KL(q | p) of Gaussian q to unit Gaussian p.

    Args:
        mu_z (torch.Tensor): batch of means of q
        logvar_z (torch.Tensor): batch of log-variance of q

    Returns:
        kl (torch.Tensor): KL-divergence for each example in the mini-batch.
    """
    # -------------------------------------
    # TODO: ENTER CODE HERE (Exercise 5.2a)
    # -------------------------------------
    
    return kl

def test_kl(mu, var):
    """Small unit test for the `compute_kl` function.
    
    Args:
        mu (float): mean of 1-dimensional Gaussian
        sigma (float): standard deviation of 1-dimensional Gaussian
    """
    mu = torch.tensor(mu)
    var = torch.tensor(var)
    # Create input tensors
    logvar = torch.log(var)

    # Create graph
    kl = compute_kl(mu, logvar)

    # Ground truth
    kl_theory = 0.5 * (-2 * np.log(var) - 1 + var*var + mu*mu)

    # Show output
    print('##########################')
    print('# Test KL-Implementation #')
    print('##########################')
    print('mu = %.4f' % mu)
    print('sigma = %.4f' % var)
    print('True KL: %.4f' % kl_theory)
    print('Predicted KL %.4f' % kl)
    print()


if __name__ == '__main__':
    main()
