
# 本节用于定义超参，以及解析命令行参数
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--repo_name', type=str, default="ansible", help='name of repo specified for RUN')
parser.add_argument('--head_define', type=str, default="transHead", help='the head chose for EarlyPRClassifier.')
parser.add_argument('--codeBertS_name', type=str, default="codeBert-base", help='ways for Msg And CodeDiff analyzed.')
parser.add_argument('--device', type=int, default=0, help='device id set to use, default is -1, cpu.')
parser.add_argument(
        "--output_dir", type=str, default='output',
        help="The output directory where the model checkpoints and test results will be saved."
    )

parser.add_argument(
    "--learning_rate", default=1e-05, type=float,
    help="The initial learning rate."
)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float,
    help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--adam_weight_decay", default=0.01, type=float,
    help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_beta1", default=0.9, type=float,
    help="Beta1 for Adam optimizer."
)
parser.add_argument(
    "--adam_beta2", default=0.999, type=float,
    help="Beta2 for Adam optimizer."
)
parser.add_argument(
    "--warmup_steps", default=4, type=int,
    help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--num_train_epochs", default=100, type=int,
    help="Total number of training epochs."
)
parser.add_argument(
    '--save_steps', type=int, default=20,
    help='steps (epochs) to save model'
)
parser.add_argument(
    '--seed', type=int, default=3407,  # arXiv:2109.08203
    help='random seed'
)

parser.add_argument('--test_device', type=int, default=-1, required=False,
                    help='Device defined only for testing.')
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_test", action='store_true')
parser.add_argument('--modelPath', type=str, default=None, required=False,
                    help='model trained file path saved for test. Required if --do_test is specified.')

parser.add_argument('--MPmodelPath', type=str, default=None, required=False,
                    help='mergedPR model trained file path saved for test. Required if --do_test is specified.')

parser.add_argument('--mergedPRTest', type=bool, default=False, required=False,
                    help='whether to conduct merged PR Test.')

