from Modules.MLPGAN import MLPGAN
from Utilities import data_utilities as d_u

# mnist
dataset = d_u.MNIST_loader(root='dataset/mnist', image_size=32)
n_chan = 1  #number of channels

gan = MLPGAN(image_size=32, n_z=64, n_chan=n_chan,
             hiddens={'gen':  [256, 512, 1024], 'dis': [1024, 512, 256]},
             depths={'gen': 4, 'dis': 4},
             ngpu=1)

# Optional arguments
batch_size = 100
n_iters = 1e05
opt_dets = {'gen': {'name': 'adam',
                    'learn_rate': 1e-04},
            'dis': {'name': 'adam',
                    'learn_rate': 1e-04}
            }

# Call training
gan.train(dataset=dataset, batch_size=batch_size,
          n_iters=n_iters, optimizer_details=opt_dets)
