import os
import subprocess
import argparse
from time import gmtime, strftime

def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1)
    parser.add_argument('--lr',
                        type=float,
                        default=None)
    parser.add_argument('--arch',
                        type=str,
                        default='roberta_base',
                        choices=[
			    'roberta_base',
                            'roberta_large',
                        ],
                        help='model architecture')
    parser.add_argument('--task',
                        type=str,
                        default='rte',
                        choices=[
			    'rte', 'sst', 'mnli', 'qnli', 'cola',
                            'qqp', 'mrpc', 'sts',
                        ],
                        help='finetuning task')
    parser.add_argument('--quant-mode',
                        type=str,
                        default='symmetric',
                        choices=[
			    'none', 'symmetric',
                        ],
                        help='quantization mode')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='folder name to store checkpoints')
    parser.add_argument('--checkpoint-suffix', type=str, default=None,
                        help='suffix for checkpoints')
    parser.add_argument('--restore-file', type=str, default=None,
                        help='finetuning from the given checkpoint')
    parser.add_argument('--reset-lr-scheduler', action='store_true')
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()
    return args

args = arg_parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
max_epochs = '12'

task_specs = {
    'rte' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'RTE-bin',
        'num_classes': '2',
        'lr': '2e-5',
        'max_sentences': '16',
        'total_num_updates': '2036',
        'warm_updates': '122'
    },
    'sst' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'SST-2-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '20935',
        #'total_num_updates': '35935',
        'warm_updates': '1256'
    },
    'mnli' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'MNLI-bin',
        'num_classes': '3',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '123873',
        'warm_updates': '7432',
        'max_epochs': '7'
    },
    'qnli' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'QNLI-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '33112',
        'warm_updates': '1986'
    },
    'cola' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'CoLA-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '16',
        'total_num_updates': '5336',
        'warm_updates': '320'
    },
    'qqp' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'QQP-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '113272',
        'warm_updates': '28318'
    },
    'mrpc' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'MRPC-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '16',
        'total_num_updates': '2296',
        'warm_updates': '137'
    },
    'sts' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'STS-B-bin',
        'num_classes': '1',
        'lr': '2e-5',
        'max_sentences': '16',
        'total_num_updates': '3598',
        'warm_updates': '214'
    },
}

spec = task_specs[args.task]
task = spec['task']
criterion = spec['criterion']
dataset = spec['dataset']
num_classes = spec['num_classes']
lr = spec['lr']
max_sentences = spec['max_sentences']
total_num_updates = spec['total_num_updates']
warm_updates = spec['warm_updates']

# no warm update for Q.A.finetuing
if args.quant_mode == 'symmetric':
    warm_updates = '0'

ROBERTA_PATH = '/rscratch/sehoonkim/models/roberta.large/model.pt' if 'large' in args.arch else \
               '/rscratch/sehoonkim/models/roberta.base/model.pt'

if 'max_epochs' in spec:
    max_epochs = spec['max_epochs']

finetuning_args = []
tuning = 'base'

if args.reset_lr_scheduler:
    if args.lr is None:
        raise Exception('please indicate the learning late with --lr')
    else:
        lr = str(args.lr)
    print("LR scheduler reset, lr = %f" % args.lr)
    finetuning_args.append('--reset-lr-scheduler')

if args.restore_file is not None:
    tuning = 'finetuning'
    if args.restore_file == 'default':
        args.restore_file = 'checkpoints_best_acc/%s-best.pt' % args.task

    print("Finetuning from the checkpoint: %s" % args.restore_file)
    finetuning_args.append('--restore-file')
    finetuning_args.append(args.restore_file)
    #if args.reset_lr_scheduler:
    #    if args.lr is None:
    #        raise Exception('please indicate the learning late with --lr')
    #    else:
    #        lr = str(args.lr)
    #    print("LR scheduler reset, lr = %f" % args.lr)

save_dir = 'checkpoints_%s_%s_%s' % (args.task, tuning, args.quant_mode) if args.save_dir is None \
           else args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

time = strftime("%m%d-%H%M%S", gmtime())
checkpoint_suffix = '_large' if 'large' in args.arch  else ''
checkpoint_suffix += '_lr%s_%s' % (lr, time) if args.checkpoint_suffix is None \
                    else args.checkpoint_suffix

print('Hyperparam: lr = %s, dropout = %s, max_epochs = %s' % (str(lr), str(args.dropout), str(max_epochs)),
        flush=True)

subprocess_args = [
    'fairseq-train', dataset,
    '--restore-file', ROBERTA_PATH,
    '--max-positions', '512',
    '--max-sentences', max_sentences,
    '--max-tokens', '4400',
    '--task', task,
    '--reset-optimizer',  '--reset-dataloader', '--reset-meters',
    '--required-batch-size-multiple',  '1',
    '--init-token', '0', '--separator-token', '2',
    '--arch',  args.arch,
    '--criterion', criterion,
    '--num-classes', num_classes,
    '--weight-decay', '0.1', 
    '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--adam-eps', '1e-06',
    '--clip-norm',  '0.0',
    '--lr-scheduler',  'polynomial_decay', '--lr', lr,
    '--total-num-update', total_num_updates, '--warmup-updates', warm_updates,
    '--max-epoch',  max_epochs,
    '--find-unused-parameters',  
    '--best-checkpoint-metric', 'accuracy', 
    '--quant-mode', args.quant_mode,
    '--save-dir', save_dir, '--checkpoint-suffix', checkpoint_suffix,
    '--dropout', str(args.dropout), '--attention-dropout', str(args.dropout),
    ]

if args.task == 'sts':
    subprocess_args += ['--regression-target', '--best-checkpoint-metric', 'loss']
    #subprocess_args += ['--regression-target']
    #subprocess_args.append('--maximize-best-checkpoint-metric')
else:
    subprocess_args.append('--maximize-best-checkpoint-metric')

subprocess_args = subprocess_args + finetuning_args

subprocess.call(subprocess_args)
