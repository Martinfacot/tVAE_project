"""TVAE module with integrated loss visualization."""

import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax': # continuous, MSE
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else: # discrete, CrossEntropy
                ed = st + span_info.dim
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                    )
                )
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
        self,
        embedding_dim=128, #latent space dimension
        compress_dims=(128, 128), # encoder dimension
        decompress_dims=(128, 128), # decoder dimension
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        verbose=False,
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss', 'Reconstruction Loss', 'KLD Loss'])
        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._fitted = False

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer() 
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data) # preprocess data
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss', 'Reconstruction Loss', 'KLD Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Total Loss: {loss:.3f} | Recon Loss: {loss_1:.3f} | KLD Loss: {loss_2:.3f}'
            iterator.set_description(iterator_description.format(loss=0, loss_1=0, loss_2=0))

        for i in iterator:
            epoch_loss_1 = 0.0
            epoch_loss_2 = 0.0
            epoch_total_loss = 0.0
            num_batches = 0

            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu # reparameterization trick
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                # Accumulate losses
                epoch_loss_1 += loss_1.item()
                epoch_loss_2 += loss_2.item()
                epoch_total_loss += loss.item()
                num_batches += 1

                # Record batch losses
                batch_loss_df = pd.DataFrame({
                    'Epoch': [i],
                    'Batch': [id_],
                    'Loss': [loss.item()],
                    'Reconstruction Loss': [loss_1.item()],
                    'KLD Loss': [loss_2.item()],
                })
                self.loss_values = pd.concat([self.loss_values, batch_loss_df], ignore_index=True)

            # Calculate average losses for the epoch
            if num_batches > 0:
                epoch_loss_1 /= num_batches
                epoch_loss_2 /= num_batches
                epoch_total_loss /= num_batches
                
            if self.verbose:
                iterator.set_description(
                    iterator_description.format(
                        loss=epoch_total_loss, loss_1=epoch_loss_1, loss_2=epoch_loss_2
                    )
                )
        
        self._fitted = True

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)

    def get_loss_values(self):
        """Get the loss values from the model.

        Returns:
            pd.DataFrame:
                Dataframe containing the loss values per epoch.
        """
        if not self._fitted:
            raise RuntimeError('Loss values are not available yet. Please fit the model first.')

        return self.loss_values.copy()

    def plot_loss_over_epochs(self, loss_values=None):
        """Plot the loss components across epochs.

        Args:
            loss_values (pd.DataFrame, optional):
                DataFrame containing loss values. If None, uses the model's loss values.
        """
        if loss_values is None:
            loss_values = self.get_loss_values()

        # Group by epoch and calculate mean loss per epoch
        epoch_loss = loss_values.groupby('Epoch')['Loss'].mean().reset_index()

        plt.figure(figsize=(12, 6))

        # Plot mean loss per epoch
        plt.subplot(1, 2, 1)
        plt.plot(epoch_loss['Epoch'], epoch_loss['Loss'], 'g-', label='Total Loss')
        plt.title('Mean Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot batch losses across epochs
        plt.subplot(1, 2, 2)
        for epoch in sorted(loss_values['Epoch'].unique()):
            epoch_data = loss_values[loss_values['Epoch'] == epoch]
            plt.scatter([epoch] * len(epoch_data), epoch_data['Loss'], 
                        alpha=0.3, s=10, color='blue')
        plt.title('Loss per Batch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_detailed_loss(self, loss_values=None):
        """Plot the detailed loss components across epochs.
        
        This function visualizes the reconstruction loss (loss_1), KLD loss (loss_2),
        and the total loss across training epochs.
    
        Args:
            loss_values (pd.DataFrame, optional):
                DataFrame containing loss values. If None, uses the model's loss values.
        """
        if loss_values is None:
            loss_values = self.get_loss_values()
    
        # Group by epoch and calculate mean losses per epoch
        epoch_losses = loss_values.groupby('Epoch').agg({
            'Loss': 'mean',
            'Reconstruction Loss': 'mean',
            'KLD Loss': 'mean'
        }).reset_index()
    
        # Create figure with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot total loss
        axes[0, 0].plot(epoch_losses['Epoch'], epoch_losses['Loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss per Epoch')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot reconstruction loss
        axes[0, 1].plot(epoch_losses['Epoch'], epoch_losses['Reconstruction Loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss per Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot KLD loss
        axes[1, 0].plot(epoch_losses['Epoch'], epoch_losses['KLD Loss'], 'g-', linewidth=2)
        axes[1, 0].set_title('KLD Loss per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot all losses together for comparison
        axes[1, 1].plot(epoch_losses['Epoch'], epoch_losses['Loss'], 'b-', label='Total Loss', linewidth=2)
        axes[1, 1].plot(epoch_losses['Epoch'], epoch_losses['Reconstruction Loss'], 'r-', 
                       label='Reconstruction Loss', linewidth=2)
        axes[1, 1].plot(epoch_losses['Epoch'], epoch_losses['KLD Loss'], 'g-', label='KLD Loss', linewidth=2)
        axes[1, 1].set_title('All Losses Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Return epoch losses DataFrame for further analysis if needed
        return epoch_losses