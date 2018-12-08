# Pokemon Incubator

## Train

1. Save training images to folder `training-data`

2. Run training script

```bash
python train.py --epochs=100 --batch-size=32 \
    --sample-interval=50 --load-saved=True --method=wgangp
```
