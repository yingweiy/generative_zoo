from Modules.MLPGAN import MLPGAN
from Utilities import data_utilities as d_u

# mnist
dataset = d_u.MNIST_loader(root='dataset/mnist', image_size=32)
n_chan = 1  #number of channels

gan = MLPGAN(image_size=32, n_z=128, n_chan=n_chan,
             hiddens={'gen':  64, 'dis': 64},
             depths={'gen': 5, 'dis': 5},
             ngpu=1)

# Optional arguments
batch_size = 100
n_iters = 1e05
opt_dets = {'gen': {'name': 'adam',
                    'learn_rate': 1e-04,
                    'betas': (0.5, 0.99)},
            'dis': {'name': 'sgd',
                    'learn_rate': 1e-04,
                    'momentum': 0.9,
                    'nesterov': True}
            }

# Call training
gan.train(dataset=dataset, batch_size=batch_size,
          n_iters=n_iters, optimizer_details=opt_dets)
