import argparse
import os
import random
import logging
import sys
import numpy as np
from PIL import Image


import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from minigpt_utils import visual_attacker, prompt_wrapper
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# Logger类实现，将控制台输出写入日志文件
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag

        # 确保日志文件的目录存在
        log_dir = os.path.dirname(self.filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=2000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--save_dir", type=str, default='output', help="save directory")
    parser.add_argument("--log_file", type=str, default=None, help="log file to save the output")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    # 如果用户没有指定日志文件，使用默认文件名并保存在 save_dir 目录中
    if args.log_file is None:
        args.log_file = os.path.join(args.save_dir, 'output_log.txt')

    return args


def setup_logging(log_file):
    """配置日志输出到控制台和文件"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 设置日志输出到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 设置日志输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

# 使用 argparse 来解析命令行参数
args = parse_args()

# 设置控制台和日志文件输出
sys.stdout = Logger(filename=args.log_file, add_flag=True)
sys.stderr = Logger(filename=args.log_file, add_flag=True)  # 如果需要重定向错误输出

print('>>> Initializing Models')

# 初始化日志输出
setup_logging(args.log_file)
logging.info("Starting the attack process...")

# 加载配置文件
cfg = Config(args)

# 初始化模型
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

# 初始化视觉处理器
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

logging.info('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


import csv

file = open("harmful_corpus/derogatory_corpus.csv", "r")
# file = open("harmful_corpus/derogatory_jambench_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
for i in range(len(data)):
    targets.append(data[i][0])

logging.info(f"Targets: {targets}")

template_img = 'adversarial_images/clean.jpeg'
img = Image.open(template_img).convert('RGB')
img = vis_processor(img).unsqueeze(0).to(model.device)

# 加载文本提示模板
text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input

# 加载对抗攻击器
from minigpt_utils import visual_attacker_spsa_e
my_attacker = visual_attacker_spsa_e.Attacker(args, model, targets, device=model.device, is_rtp=False)

if not args.constrained:
    adv_img_prompt = my_attacker.attack_unconstrained_spsa_SignSGD(text_prompt_template,
                                                  img=img,
                                                  batch_size=8,
                                                  num_iter=5000,
                                                  alpha=args.alpha / 255,
                                                  )


    # adv_img_prompt = my_attacker.attack_unconstrained_spsa_Adam(text_prompt_template,
    #                                                   img=img,
    #                                                   batch_size=8,
    #                                                   num_iter=10000,
    #                                                   alpha=args.alpha / 255,
    #                                                   )

# else:
#     adv_img_prompt = my_attacker.attack_constrained_spsa_SignSGD(text_prompt_template,
#                                                     img=img,
#                                                     batch_size= 8,
#                                                     num_iter=10000,
#                                                     alpha=args.alpha / 255,
#                                                     epsilon=args.eps / 255)


# 保存图片时通过字典键访问 save_dir
save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
logging.info('[Done]')
