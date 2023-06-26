import argparse
import requests
import torch
from PIL import Image
from pprint import pprint

from seechat.common.config import Config
from seechat.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="./eval_configs/seechat_eval_stage2.yaml",
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


class CALL():
    def __init__(self):
        print('Initializing Chat')
        args = parse_args()
        cfg = Config(args)

        # ========================================
        #             Model Initialization
        # ========================================
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        print('Initialization Finished')

        self.device = 'cuda:{}'.format(args.gpu_id)

        self.before = "###Human: <Img>"
        self.before_ids = self.model.llama_tokenizer([self.before], return_tensors="pt",
                                                     add_special_tokens=False)["input_ids"]
        self.before_embed = self.model.llama_model.transformer.word_embeddings(self.before_ids.to(self.device))
        self.first_after = "</Img> 详细描述这张图片 ###Assistant: "

        self.gen_kwargs = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": True,
            "min_length": 1,
            "top_p": 0.9,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0
        }

        self.user_history = {}
        '''
        {
            "user_id": {
                "session_id": {
                    "img_embed": "",
                    "history": ""
                }
            }
        }
        '''

    def __get_img_embed(self, image_url):
        image = Image.open(
            requests.get(
                image_url,
                stream=True
            ).raw
        ).convert('RGB')
        image = self.vis_processor(image).unsqueeze(0).to(self.device)
        image_embed, _ = self.model.encode_img(image)
        return image_embed

    def response(self, user_id, session_id, image_url, ask):
        try:
            if user_id in self.user_history.keys() and session_id in self.user_history[user_id].keys():  # > 1st
                after = self.first_after + self.user_history[user_id][session_id]["history"] + \
                        "###Human: " + ask + " ###Assistant: "
            else:  # 1st
                try:
                    img_embed = self.__get_img_embed(image_url)
                except:
                    ret = {
                        "user_id": user_id,
                        "session_id": session_id,
                        "image_url": image_url,
                        "ask": ask,
                        "answer": None,
                        "error_code": "1",
                        "error_msg": "image read error"
                    }
                    pprint(ret)
                    return ret

                self.user_history[user_id] = {
                    session_id: {
                        "img_embed": img_embed,
                        "history": None
                    }
                }
                after = self.first_after

            try:
                image_ids = torch.zeros([1, 32])
                after_ids = self.model.llama_tokenizer([after], return_tensors="pt")["input_ids"]
                after_embed = self.model.llama_model.transformer.word_embeddings(after_ids.to(self.device))
                input_ids = torch.cat([self.before_ids, image_ids, after_ids], dim=1).long().to(self.device)
                inputs_embeds = torch.cat([self.before_embed, self.user_history[user_id][session_id]["img_embed"],
                                           after_embed], dim=1)
                prefix_len = inputs_embeds.shape[1]

                self.gen_kwargs.update({"input_ids": input_ids, "inputs_embeds": inputs_embeds})

                out_token = self.model.llama_model.generate(**self.gen_kwargs).tolist()[0]
                answer1 = self.model.llama_tokenizer.decode(out_token[prefix_len:])
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

            except:
                ret = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "image_url": image_url,
                    "ask": ask,
                    "answer": None,
                    "error_code": "2",
                    "error_msg": "model generate error"
                }
                pprint(ret)
                return ret

            if self.user_history[user_id][session_id]["history"] is None:
                self.user_history[user_id][session_id]["history"] = answer

            ret = {
                "user_id": user_id,
                "session_id": session_id,
                "image_url": image_url,
                "ask": ask,
                "answer": answer,
                "error_code": "0",
                "error_msg": None
            }
            pprint(ret)

            try:
                f = open("./llm/log.txt", 'a')
                for k, v in ret.items():
                    f.write(f"{k}:{v}\n")
                f.write("\n")
                f.flush()
                f.close()
            except:
                pass

            return ret

        except:
            ret = {
                "user_id": user_id,
                "session_id": session_id,
                "image_url": image_url,
                "ask": ask,
                "answer": None,
                "error_code": "3",
                "error_msg": "unknown error"
            }
            pprint(ret)
            return ret


call = CALL()

