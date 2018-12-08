from utils import parse_args
from wgan.wgan import WGAN


args = parse_args()

WGAN(load_saved=args.load_saved).train(epochs=args.epochs, batch_size=args.batch_size,
                                       sample_interval=args.sample_interval)
