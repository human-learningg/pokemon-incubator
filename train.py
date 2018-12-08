from wgan.wgan import WGAN

WGAN().train(epochs=100, batch_size=32, sample_interval=50)