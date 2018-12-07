import dcgan

EPOCHS = 50
BATCH_SIZE = 16

dcgan.train(EPOCHS, BATCH_SIZE, weights=False)
dcgan.generate(BATCH_SIZE)
