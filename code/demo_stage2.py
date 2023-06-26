import argparse
import os
import sys
import pdb
from tqdm import tqdm
import torch
from PIL import Image
from time import time

from seechat.common.config import Config
from seechat.common.registry import registry

import random
import numpy as np
import torch.backends.cudnn as cudnn

seed = 42
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
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print('Initialization Finished')

    device = 'cuda:{}'.format(args.gpu_id)

    img_root = "../images/"
    image_names = os.listdir(img_root)
    image_names.sort()

    # prompt_template = "###Human: <Img><ImageHere></Img> 请详细描述这张图片。 ###Assistant: "
    before = "###Human: <Img>"
    before_ids = model.llama_tokenizer([before], return_tensors="pt", add_special_tokens=False)["input_ids"]
    before_embed = model.llama_model.transformer.word_embeddings(before_ids.to(device))

    first_prompt = "详细描述这张图片"
    first_after = "</Img> %s ###Assistant: " % first_prompt

    gen_kwargs = {
        "max_new_tokens": 512,
        "num_beams": 1,
        "do_sample": False,
        "min_length": 1,
        "top_p": 0.9,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0
    }

    use_stream = False

    for image_name in tqdm(image_names):
        path = os.path.join(img_root, image_name)
        print(path)
        image = Image.open(path).convert("RGB")
        image = vis_processor(image).unsqueeze(0).to(device)
        image_embed, _ = model.encode_img(image)

        caption = ""

        loop = 0
        # while True:
        if loop == 0:
            after = first_after
        else:
            try:
#                     prompt = input("question: ")
                prompt = "详细描述这张图片"
                if prompt == "break":
                    break
            except:
                continue
            after = first_after + caption + "###Human: " + prompt + " ###Assistant: "

        t1 = time()

        image_ids = torch.zeros([1, 32])
        after_ids = model.llama_tokenizer([after], return_tensors="pt")["input_ids"]
        after_embed = model.llama_model.transformer.word_embeddings(after_ids.to(device))
        input_ids = torch.cat([before_ids, image_ids, after_ids], dim=1).long().to(device)
        inputs_embeds = torch.cat([before_embed, image_embed, after_embed], dim=1)

        gen_kwargs.update({"input_ids": input_ids, "inputs_embeds": inputs_embeds})

        if use_stream:
            answer = ""
            print("answer: ", end='')
            for outs in model.llama_model.stream_generate(**gen_kwargs):
                cur_token = outs.tolist()[0][-1]
                cur_answer = model.llama_tokenizer.decode([cur_token])
                answer += cur_answer
                print(cur_answer, end='')
                sys.stdout.flush()
            print()

            if loop == 0:
                caption = answer

        else:
            prefix_len = inputs_embeds.shape[1]
            out_token = model.llama_model.generate(**gen_kwargs).tolist()[0]
            answer1 = model.llama_tokenizer.decode(out_token[prefix_len:])
            print("answer1: ", answer1)
            answer1 = answer1.replace(",", "，")
            answer1 = answer1.replace("Human:", "")
            answer1 = answer1.replace("<Img>", "")
            answer1 = answer1.replace("</Img>", "")
            answer1 = answer1.replace(" ", "")
            answer2 = answer1.split('###')[0]
            answer2 = answer2.split('Assistant:')[-1].strip()
            if answer2 == '':
                answer1 = answer1.replace("###", "")
                answer1 = answer1.replace("Assistant:", "")
                answer1 = answer1.strip()
                answer = answer1
            else:
                answer = answer2

            print("answer2: ", answer)
            print("*"*50)

            if loop == 0:
                caption = answer

        t2 = time()
        print("%.1fs" % (t2 - t1))

        loop += 1
#             break
# pdb.set_trace()
