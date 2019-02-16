# This is a sample main file highlighting the usage of DCGAN module in DCGAN.py
# Please edit this file based on your requirements
import sys
from Modules.DCGAN import DCGAN
from Utilities import data_utilities as d_u

# Dataset
dset = 'mnist' #sys.argv[1]
root = 'dataset/{}'.format(dset) #sys.argv[2]

# Arguments passed to dataset loaders can be modified based on required usage.
# Please look at the documentation of the loader functions using the help(d_u) command.

if dset == 'mnist':
    dataset = d_u.MNIST_loader(root=root, image_size=32)
    n_chan = 1
elif dset == 'cifar10':
    dataset = d_u.CIFAR10_loader(root=root, image_size=32, normalize=True)
    n_chan = 3
elif dset == 'lsun':
    dataset = d_u.LSUN_loader(root=root, image_size=32, classes=['bedroom'], normalize=True)
    n_chan = 3

# DCGAN object initialization
# Parameters below can be modified
# Please check the documentation using help(dc.DCGAN.__init__)
params = {'image_size': 32, 'n_z': 128, 'n_chan': n_chan, 'hiddens': {'gen':  64, 'dis': 64}}
arch = {'arch_type': 'Generic', 'params': params}
ngpu = 1
loss = 'BCE'

Gen_model = DCGAN(arch, ngpu=ngpu, loss=loss)

# DCGAN training scheme
# Parameters below can be modified based on required usage.
# Please check the documentation using help(dc.DCGAN.train) for more details
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

# Optional arguments
show_period = 50
display_images = True
misc_options = ['init_scheme', 'save_model']

# Call training
Gen_model.train(dataset=dataset, batch_size=batch_size, n_iters=n_iters,pic_folder='pics/{}_dcgan'.format(dset),
                optimizer_details=opt_dets, show_period=show_period,
                display_images=display_images, misc_options=misc_options)

# Voila, your work is done
