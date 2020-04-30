from trainer import Trainer

def train():
    trainer = Trainer(
        content_dir="../StyleTransfer/datasets/train2017",
        style_dir="./style_datasets",
        log_dir='./log',
        weight_dir='./weight',
        learn_rate=1e-3,
        batch_size=20,
        target_size=256,
        num_workers=0,
        style_weight=1e5,
        content_weight=1,
        tv_weight=1e-7,
        vision_steps=1000,
        save_steps=1,
    )
    trainer.train(20)
train()