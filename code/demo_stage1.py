import argparse
import os
import sys
import random
import pdb
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from time import time

from seechat.common.config import Config
from seechat.common.dist_utils import get_rank
from seechat.common.registry import registry

from transformers import StoppingCriteria, StoppingCriteriaList


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/seechat_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=2, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    # ========================================
    #             Model Initialization
    # ========================================
    model_config = cfg.model_cfg
    model_type = model_config.model_type
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print('Initialization Finished')

    device = 'cuda:{}'.format(args.gpu_id)

    txt_root = "./fast-data/"
    f = open(os.path.join(txt_root, "caption_res.txt"), 'w')

    img_root = "./fast-data/caption_img/vqa"
    image_names = os.listdir(img_root)
    image_names.sort()

    for image_name in tqdm(image_names):
        path = os.path.join(img_root, image_name)
        image = Image.open(path)
        image = vis_processor(image).unsqueeze(0).to(device)
        image_embed, _ = model.encode_img(image)

        history = ""
        caption = ""

        flag = True

        if flag:
            loop = 0
            while True:
                prompt = "详细描述这张图片"

                t1 = time()
                image_ids = torch.zeros([1, 32])
                text_ids = model.llama_tokenizer([prompt], return_tensors="pt")["input_ids"]
                input_ids = torch.cat([image_ids, text_ids], dim=1).int().to(device)
                text_embed = model.llama_model.transformer.word_embeddings(text_ids.to(device))
                inputs_embeds = torch.cat([image_embed, text_embed], dim=1)
                prefix_len = inputs_embeds.shape[1]
                
                answer = ""
                for out in model.llama_model.stream_generate(input_ids=input_ids, inputs_embeds=inputs_embeds, max_new_tokens=512, do_sample=True, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1.0, temperature=1.0,):
                    cur_token = out.tolist()[0][-1]
                    cur_answer = model.llama_tokenizer.decode([cur_token])
                    print(cur_answer, end='')
                    sys.stdout.flush()
#                     answer += cur_answer
#                     print(answer, end='\r')
                t2 = time()

                print()
                print("%.1fs" % (t2 - t1))
                print()

                loop += 1
        else:
            while True:
                try:
                    prompt = input("prompt: ")
                except:
                    continue
                if prompt == "":
                    history = ""
                    prompt = "详细描述这张图片"
                history += prompt

                t1 = time()
                image_ids = torch.zeros([1, 32])
                text_ids = model.llama_tokenizer([history], return_tensors="pt")["input_ids"]
                input_ids = torch.cat([image_ids, text_ids], dim=1).int().to(device)
                text_embed = model.llama_model.transformer.word_embeddings(text_ids.to(device))
                inputs_embeds = torch.cat([image_embed, text_embed], dim=1)
                prefix_len = inputs_embeds.shape[1]

                out_token = model.llama_model.generate(
                    input_ids=input_ids, inputs_embeds=inputs_embeds, max_new_tokens=512,
                    num_beams=1, do_sample=False, min_length=1, top_p=0.9,
                    repetition_penalty=1.0, length_penalty=1.0, temperature=1.0,
                ).tolist()[0]
                out_sentence = model.llama_tokenizer.decode(out_token[prefix_len:])
                history += out_sentence
                t2 = time()

                print("%.1fs" % (t2 - t1))
                print("answer: ", out_sentence)
                f.write(image_name + '\n')
                f.write(out_sentence + '\n\n')
                f.flush()

    f.close()

    pdb.set_trace()
