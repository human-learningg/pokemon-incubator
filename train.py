from wgan.wgan import WGAN


WGAN().train(epochs=4000, batch_size=32, sample_interval=50)
