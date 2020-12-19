import os
import subprocess
import argparse
from time import gmtime, strftime

def make_dir(args, is_large, lr):
    root = args.output_dir
    no_save = args.no_save
    task_dir = args.task + '-' + ('large' if is_large else 'base')
    hyperparam_dir = 'wd%s_ad%s_d%s_lr%s' % (str(args.weight_decay), 
        str(args.attn_dropout), str(args.dropout),  str(lr))

    time = strftime("%m%d-%H%M%S", gmtime())
    log_name = '%s.log' % time
    ckpt_name = '%s_ckpt' % time

    log_dir = os.path.join(root, args.quant_mode)
    log_dir = os.path.join(log_dir, task_dir)
    log_dir = os.path.join(log_dir, hyperparam_dir)

    log_file = os.path.join(log_dir, log_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not no_save:
        ckpt_dir = os.path.join(log_dir, ckpt_name)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    else:
        ckpt_dir = log_dir # dummy directory

    return log_file, ckpt_dir


def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')

    # hyperparameters
    parser.add_argument('--attn-dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--bs', type=float, default=None, help='batch size')

    parser.add_argument('--arch', type=str, default='roberta_base',
                        choices=['roberta_base', 'roberta_large', ],
                        help='model architecture')
    parser.add_argument('--task', type=str,
                        choices=['RTE', 'SST-2', 'MNLI', 'QNLI',
                                 'CoLA', 'QQP', 'MRPC', 'STS-B',],
                        help='finetuning task')
    parser.add_argument('--quant-mode', type=str,
                        choices=['none', 'symmetric',],
                        help='quantization mode')
    parser.add_argument('--force-dequant', type=str, default='none', 
                        choices=['none', 'gelu', 'layernorm', 'softmax', 'nonlinear'],
                        help='force dequantize the specific layers')

    parser.add_argument('--model-dir', type=str, default='models',
                        help='model directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='folder name to store logs and checkpoints')
    parser.add_argument('--restore-file', type=str, default=None,
                        help='finetuning from the given checkpoint')
    parser.add_argument('--no-save', action='store_true')

    args = parser.parse_args()
    return args

args = arg_parse()
task = args.task


######################## Task specs ##########################

task_specs = {
    'RTE' : {
        'dataset': 'RTE-bin',
        'num_classes': '2',
        'lr': '2e-5',
        'max_sentences': '16',
        'total_num_updates': '2036',
        'warm_updates': '122',
    },
    'SST' : {
        'dataset': 'SST-2-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '20935',
        'warm_updates': '1256'
    },
    'MNLI' : {
        'dataset': 'MNLI-bin',
        'num_classes': '3',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '123873',
        'warm_updates': '7432',
        'valid_interval_sentences': '100000',
    },
    'QNLI' : {
        'dataset': 'QNLI-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '33112',
        'warm_updates': '1986',
        'valid_interval_sentences': '55000',
    },
    'CoLA' : {
        'dataset': 'CoLA-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '16',
        'total_num_updates': '5336',
        'warm_updates': '320'
    },
    'QQP' : {
        'dataset': 'QQP-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '113272',
        'warm_updates': '28318',
        'valid_interval_sentences': '950000',
    },
    'MRPC' : {
        'dataset': 'MRPC-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '16',
        'total_num_updates': '2296',
        'warm_updates': '137'
    },
    'STS-B' : {
        'dataset': 'STS-B-bin',
        'num_classes': '1',
        'lr': '2e-5',
        'max_sentences': '16',
        'total_num_updates': '3598',
        'warm_updates': '214'
    },
}


is_large = 'large' in args.arch
spec = task_specs[task]
dataset = '%s-bin' % task
num_classes = spec['num_classes']
total_num_updates = spec['total_num_updates']
warm_updates = spec['warm_updates']
max_epochs = '6' if task in ['MNLI', 'QQP'] else '12'

lr = str(args.lr) if args.lr else spec['lr'] 
bs = str(args.bs) if args.bs else spec['max_sentences']

log_file, ckpt_dir = make_dir(args, is_large, lr)
model_path = args.model_dir  + '/roberta.large/model.pt' if is_large \
        else args.model_dir + '/roberta.base/model.pt'

valid_subset = 'valid' if task != 'MNLI' else 'valid,valid1'
valid_interval_updates = None
if 'valid_interval_sentences' in spec:
    valid_interval_updates = \
            str(int(int(spec['valid_interval_sentences']) / int(bs)))

print('valid_subset:',valid_subset)
print('valid_interval_updates:', valid_interval_updates)

###############################################################

finetuning_args = []
if args.quant_mode == 'symmetric':
    warm_updates = '0' # no warm update for Q.A.finetuing
    if args.restore_file is None:
        raise Exception('please specify --restore-file for symmetric mode')
    print("Finetuning from the checkpoint: %s" % args.restore_file)
    finetuning_args.append('--restore-file')
    finetuning_args.append(args.restore_file)
    finetuning_args.append('--reset-lr-scheduler')


subprocess_args = [
    'fairseq-train', dataset,
    '--restore-file', model_path,
    '--valid-subset', valid_subset,
    '--max-positions', '512',
    '--max-sentences', bs,
    '--max-tokens', '4400',
    '--task', 'sentence_prediction',
    '--criterion', 'sentence_prediction',
    '--reset-optimizer',  '--reset-dataloader', '--reset-meters',
    '--required-batch-size-multiple',  '1',
    '--init-token', '0', '--separator-token', '2',
    '--arch',  args.arch,
    '--num-classes', num_classes,
    '--weight-decay', str(args.weight_decay), 
    '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--adam-eps', '1e-06',
    '--clip-norm',  '0.0',
    '--lr-scheduler',  'polynomial_decay', '--lr', lr,
    '--total-num-update', total_num_updates, '--warmup-updates', warm_updates,
    '--max-epoch',  max_epochs,
    '--find-unused-parameters',  
    '--best-checkpoint-metric', 'accuracy', 
    '--save-dir', ckpt_dir, 
    '--log-file', log_file,
    '--dropout', str(args.dropout), '--attention-dropout', str(args.attn_dropout),
    '--quant-mode', args.quant_mode,
    '--force-dequant', args.force_dequant,
]

if valid_interval_updates is not None:
    subprocess_args += \
    ['--validate-interval-updates', valid_interval_updates]

if args.no_save:
    subprocess_args += ['--no-save']

if args.task == 'sts':
    subprocess_args += ['--regression-target', '--best-checkpoint-metric', 'loss']
else:
    subprocess_args.append('--maximize-best-checkpoint-metric')

subprocess_args = subprocess_args + finetuning_args

subprocess.call(subprocess_args)
