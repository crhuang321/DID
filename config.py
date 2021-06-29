class DefaultConfig(object):
    model_name = 'DID'  # DID, DID_no_attention, DID_no_decompose

    train_dataset = 'COCO'
    root = './Datasets/COCO/'
    val_dataset_dir = './Datasets/New_DIV2K/'

    tempdir = './Experiments/temp/'
    save_temp = False  # if you want to save temp results of "cA, (cH, cV, cD)"
    val_resultdir = './Experiments/val_result/'
    save_val_result = False  # if you want to save validation redults of model
    
    max_epochs = 20
    gpus = None #[0]  # None if no gpu available
    scale = 4
    patch_size = 50
    batch_size = 16
    learning_rate = 0.0002
    halve_lr = 2  # After halve_lr epochs, the learning rate becomes half of the original, and then it doesn't change

    checkpoint = './Experiments/checkpoints/' + train_dataset + '_' + model_name + '_checkpoints_' + str(scale) + 'x/'
    resume_train = True  # if you want to restore the full training

    only_test = False #True  # just test the trained model using test dataset
    model_for_testing = './Experiments/checkpoints/......'  # finish after training

    test_path = 'Image25/'
    dataset_for_tesing = './Datasets/' + test_path
    result_of_test = './Experiments/test_result/' + model_name + '/' + test_path


opt = DefaultConfig()

