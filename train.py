from utils import parse_args
from wgan.wgan import WGAN
from wgangp.wgangp import WGANGP


args = parse_args()

if args.method == 'wgan':
    WGAN(load_saved=args.load_saved).train(epochs=args.epochs, batch_size=args.batch_size,
                                           sample_interval=args.sample_interval)
elif args.method == 'wgangp':
    WGANGP(load_saved=args.load_saved).train(epochs=args.epochs, batch_size=args.batch_size,
                                             sample_interval=args.sample_interval)
else:
    print('\nmethod not supported, try:\n\twgan\n\twgangp')
