CONFIG = {
    # Kornia augmentation parameters
    'scale_range' : (0.7, 1.1),
    'rotation' : 20,
    'translation' : (0.1,0.1),
    'brightness' : 0.5,
    'contrast' : 0.5,
    'saturation' : 0.5,
    'hue' : 0.1,
    'probability' : 0.8,

    # Training parameters
    'img_size' : (252,252),
    'batch_size' : 128,
    'num_epochs': 100,
    'lr':0.0001,
    'weight_decay': 0.0001,

    # Hyperparameters for the Depth map model
    'hidden_dim' : 256,
    'lambda_depth' : 20.0,
    'lambda_blank' : 0.5,

    # Hyperparameters for the Classifier model
    'lr_classifier' : 0.001,
    'weight_decay_classifier' : 0.001,
    
}