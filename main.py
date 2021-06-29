import os
import time
from models import DID
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import opt


print(opt.model_name)

if opt.only_test == False:
    if not os.path.exists(opt.checkpoint):
        os.makedirs(opt.checkpoint)
    if opt.save_temp and not os.path.exists(opt.tempdir):
        os.makedirs(opt.tempdir)
    if opt.save_val_result and not os.path.exists(opt.val_resultdir):
        os.makedirs(opt.val_resultdir)

    logger = TensorBoardLogger('./Experiments/tb_logs', name=opt.train_dataset+'_' + opt.model_name + '_'+str(opt.scale)+'x')
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_psnr',
        dirpath=opt.checkpoint,
        filename=time.strftime(opt.model_name + '_'+str(opt.scale)+'x-{epoch:04d}-' + '-%m%d_%H:%M:%S'),
        save_top_k=-1,
        mode='max'
        )
    if opt.resume_train and len(os.listdir(opt.checkpoint)) > 0:
        model_for_resuming = opt.checkpoint + sorted(os.listdir(opt.checkpoint))[-1]
    else:
        model_for_resuming = None
    model = DID()
    trainer = pl.Trainer(
        gpus=opt.gpus, 
        # precision=16,
        accumulate_grad_batches=1,  # This value can be considered as the actual size of a batch, since batch_size is set to 1
        max_epochs=opt.max_epochs, 
        logger=logger,
        # automatic_optimization=True,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=model_for_resuming  # if you want to restore the full training
        )
    trainer.fit(model)
    checkpoint_callback.best_model_path

else:
    if not os.path.exists(opt.result_of_test):
        os.makedirs(opt.result_of_test)
    model = DID.load_from_checkpoint(opt.model_for_testing)
    print('\n===> path of trained model for testing: ', opt.checkpoint+opt.model_for_testing)
    trainer = pl.Trainer(gpus=opt.gpus)
    trainer.test(model=model)

