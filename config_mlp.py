general = {
        'train_table_path': '/home/suhas/research/minBERT/train.csv',
        'test_table_path': '/home/suhas/research/minBERT/test.csv',
        'feature_cols': "CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT".split(","),
        'target_col': "HousValue"
}


trainer = {
        'device': 'auto',
        'num_workers': 4,
        'max_iters': 10000,
        'eval_iters': 100,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'betas': (0.9, 0.95),
        'weight_decay': 0.1,
        'grad_norm_clip': 1.0
        }

model = {
        'model_name': 'mlp',
        'input_size': 13 
        }
