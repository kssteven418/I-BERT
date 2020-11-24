import os
import subprocess
import argparse
from time import gmtime, strftime

ROBERTA_PATH = 'models'

def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--attn-dropout',
                        type=float,
                        default=0.1)
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1)
    parser.add_argument('--weight-decay',
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
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='folder name to store logs')
    parser.add_argument('--checkpoint-suffix', type=str, default=None,
                        help='suffix for checkpoints')
    parser.add_argument('--restore-file', type=str, default=None,
                        help='finetuning from the given checkpoint')
    parser.add_argument('--reset-lr-scheduler', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    #parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()
    return args

args = arg_parse()
#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
max_epochs = '12'

######################## Task specs ##########################

task_specs = {
    'rte' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'RTE-bin',
        'num_classes': '2',
        'lr': '2e-5',
        'max_sentences': '16',
        'max_sentences_large': '8',
        'total_num_updates': '2036',
        'warm_updates': '122',
    },
    'sst' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'SST-2-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'max_sentences_large': '16',
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
        'max_epochs': '6',
        'valid_interval_updates': '3200',
    },
    'qnli' : {
        'task': 'sentence_prediction',
        'criterion': 'sentence_prediction',
        'dataset': 'QNLI-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'max_sentences_large': '12',
        'total_num_updates': '33112',
        'warm_updates': '1986',
        'max_epochs': '10',
        'valid_interval_updates': '1700',
        'valid_interval_updates_large': '4500',
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
        'warm_updates': '28318',
        'max_epochs': '6',
        'valid_interval_updates': '3200',
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

is_large = 'large' in args.arch
spec = task_specs[args.task]
task = spec['task']
criterion = spec['criterion']
dataset = spec['dataset']
num_classes = spec['num_classes']
lr = spec['lr']
max_sentences = spec['max_sentences']
# bsz adjustment for large models
if is_large and 'max_sentences_large' in spec:
    max_sentences = spec['max_sentences_large']
total_num_updates = spec['total_num_updates']
warm_updates = spec['warm_updates']
valid_subset = 'valid' if args.task != 'mnli' else 'valid,valid1'
if 'max_epochs' in spec:
    max_epochs = spec['max_epochs']
valid_interval_updates = None
if 'valid_interval_updates' in spec:
    valid_interval_updates = spec['valid_interval_updates']
    if is_large and 'valid_interval_updates_large' in spec:
        valid_interval_updates = spec['valid_interval_updates_large']


print('valid_subset:',valid_subset)
print('valid_interval_updates:', valid_interval_updates)

###############################################################

# no warm update for Q.A.finetuing
if args.quant_mode == 'symmetric':
    warm_updates = '0'

ROBERTA_PATH = ROBERTA_PATH + '/roberta.large/model.pt' if is_large \
        else ROBERTA_PATH + '/roberta.base/model.pt'

finetuning_args = []
tuning = 'base'

# set learning rate if specified
if args.lr:
    lr = str(args.lr)

# reset lr scheduler if reset_lr_scheduler
if args.reset_lr_scheduler:
    if args.lr is None:
        raise Exception('please indicate the learning late with --lr')
    finetuning_args.append('--reset-lr-scheduler')

# finetuning if resotre_file is specified
if args.restore_file is not None:
    tuning = 'finetuning'
    print("Finetuning from the checkpoint: %s" % args.restore_file)
    finetuning_args.append('--restore-file')
    finetuning_args.append(args.restore_file)

# checkpoint directory
save_dir = 'checkpoints_%s_%s_%s' % (args.task, tuning, args.quant_mode) if args.save_dir is None \
           else args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

date = strftime("%m%d", gmtime())
time = strftime("%m%d-%H%M%S", gmtime())
checkpoint_suffix = '_large' if is_large else ''
checkpoint_suffix += '_lr%s_%s' % (lr, time) if args.checkpoint_suffix is None \
                    else args.checkpoint_suffix

log_dir = args.log_dir + '-' + tuning
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
task_log_dir = os.path.join(log_dir, args.task + '-' + ('large' if is_large else 'base'))
if not os.path.exists(task_log_dir):
    os.makedirs(task_log_dir)

hyperparam_dir = 'lr%s_d%s_ad%s_wd%s' % (str(lr), str(args.dropout), 
    str(args.attn_dropout), str(args.weight_decay))
hyperparam_dir = os.path.join(task_log_dir, hyperparam_dir)
if not os.path.exists(hyperparam_dir):
    os.makedirs(hyperparam_dir)

log_name = time
log_file = os.path.join(hyperparam_dir, log_name)

print('Hyperparam: lr = %s, dropout = %s, attn_dropout = %s, weight_decay = %s, max_epochs = %s' % \
        (str(lr), str(args.dropout), str(args.attn_dropout), str(args.weight_decay), str(max_epochs)),
        flush=True)

subprocess_args = [
    'fairseq-train', dataset,
    '--restore-file', ROBERTA_PATH,
    '--valid-subset', valid_subset,
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
    '--weight-decay', str(args.weight_decay), 
    '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--adam-eps', '1e-06',
    '--clip-norm',  '0.0',
    '--lr-scheduler',  'polynomial_decay', '--lr', lr,
    '--total-num-update', total_num_updates, '--warmup-updates', warm_updates,
    '--max-epoch',  max_epochs,
    '--find-unused-parameters',  
    '--best-checkpoint-metric', 'accuracy', 
    '--quant-mode', args.quant_mode,
    '--save-dir', save_dir, '--checkpoint-suffix', checkpoint_suffix,
    '--log-file', log_file,
    '--dropout', str(args.dropout), '--attention-dropout', str(args.attn_dropout),
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
