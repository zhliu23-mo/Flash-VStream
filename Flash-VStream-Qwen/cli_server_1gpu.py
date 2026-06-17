"""
    File: cli_2process.py
    Description: This file demonstrates an implementation of a multiprocess Real-time Long Video System. With a multiprocess logging module.
        main process: CLI server I/O, LLM inference
        process-1: logger listener
        process-2: frame generator, 
        process-3: frame memory manager
    Author: Haoji Zhang, Haotian Liu (This code is based on https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py)
    Date: 2025-02
"""
import argparse
import requests
import logging
from logging.handlers import QueueHandler

import torch
import numpy as np
import time
import os

from torch.multiprocessing import Process, Queue, Manager
from transformers import TextStreamer
from decord import VideoReader
from datetime import datetime
from PIL import Image
from io import BytesIO

from models import (
    FlashVStreamQwen2VLConfig,
    FlashVStreamQwen2VLProcessor,
    get_real_grid_thw,
    get_spatial_real_grid_thw,
    DEFAULT_FLASH_MEMORY_CONFIG
)
from models.vstream_qwen2vl_realtime import FlashVStreamQwen2VLModel

from qwen_vl_utils import process_vision_info

class _Metric:
    def __init__(self):
        self._latest_value = None
        self._sum = 0.0
        self._max = 0.0
        self._count = 0

    @property
    def val(self):
        return self._latest_value

    @property
    def max(self):
        return self._max

    @property
    def avg(self):
        if self._count == 0:
            return float('nan')
        return self._sum / self._count

    def add(self, value):
        self._latest_value = value
        self._sum += value
        self._count += 1
        if value > self._max:
            self._max = value

    def __str__(self):
        latest_formatted = f"{self.val:.6f}" if self.val is not None else "None"
        average_formatted = f"{self.avg:.6f}"
        max_formatted = f"{self.max:.6f}"
        return f"{latest_formatted} ({average_formatted}, {max_formatted})"
        

class MetricMeter:
    def __init__(self):
        self._metrics = {}

    def add(self, key, value):
        if key not in self._metrics:
            self._metrics[key] = _Metric()
        self._metrics[key].add(value)

    def val(self, key):
        metric = self._metrics.get(key)
        if metric is None or metric.val is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.val

    def avg(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.avg

    def max(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.max

    def __getitem__(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise KeyError(f"The key '{key}' does not exist.")
        return str(metric)


def listener(queue, filename):
    ############## Start sub process-1: Listener #############
    import sys, traceback
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    h = logging.FileHandler(filename)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    while True:
        try:
            record = queue.get()
            if record is None:  # None is a signal to finish
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('[listener] Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def worker_configurer(queue):
    h = QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)

def video_stream_similator(video_file, frame_queue, log_queue, video_fps=1.0, play_speed=1.0):
    ############## Start sub process-2: Simulator #############
    worker_configurer(log_queue)
    logger = logging.getLogger('video_stream_similator')
    logger.setLevel(logging.DEBUG)

    # vr = VideoReader(video_file)
    # sample_fps = round(vr.get_avg_fps() / video_fps)
    # frame_idx = [i for i in range(0, len(vr), sample_fps)]
    # video = vr.get_batch(frame_idx).asnumpy()
    # video = np.repeat(video, 6, axis=0)

    frame_paths = os.listdir(video_file)
    frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    frame_paths = [os.path.join(video_file, frame_path) for frame_path in frame_paths]

    frame_paths = frame_paths * 3

    sleep_time = 1 / video_fps / play_speed
    time_meter = MetricMeter()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_paths,
                    "max_pixels": 4*224*224,
                    "max_frames": 3000,
                },
                {"type": "text", "text": "question"},
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    logger.info(f'[Simulator] video_inputs={len(video_inputs)}, video_len={len(video_inputs[0])}, size={np.array(video_inputs[0][0]).shape}')

    video = video_inputs[0]
    length = len(video)
    logger.info(f'[Simulator] start, length = {length}')
    step = 120  # must have this line, or modify main()
    try:
        start = 0
        while start < length:
            start_time = time.perf_counter()
            end = min(start + step, length)  # first 120 for initialization
            step = 1
            video_clip = video[start: end]
            frame_queue.put(video_clip)
            if start > 0:
                time_meter.add('real_sleep', start_time - last_start)
                # logger.info(f'[Simulator] write {end - start} frames,\t{start} to {end} (total {length}),\treal_sleep={time_meter["real_sleep"]}')
            if end < length:
                time.sleep(sleep_time)
            last_start = start_time
            start = end
        frame_queue.put(None)
    except Exception as e:
        logger.info(f'[Simulator] Exception: {e}')
        time.sleep(0.1)
    logger.info(f'[Simulator] Process: end')

def frame_memory_manager(model, processor, flash_memory_config, frame_queue, log_queue):
    torch.cuda.set_device(0) ##### 1->0 gpu.device
    model = model.cuda()
    ############## Start sub process-3: Memory Manager #############
    worker_configurer(log_queue)
    logger = logging.getLogger('frame_memory_manager')
    logger.setLevel(logging.DEBUG)

    time_meter = MetricMeter()
    logger.info(f'[MemManager] start')
    frame_cnt = 0
    while True:
        video_clip = frame_queue.get()
        if video_clip is None:
            logger.info(f'[MemManager] Ooops, get None')
            break

        video_inputs = processor.image_processor(
            images=None, 
            videos=video_clip, 
            return_tensors='pt', 
            additional_pool_size=flash_memory_config['flash_memory_temporal_poolsize']
        )

        start_time = time.perf_counter()
        with torch.inference_mode():
            time_list = model.embed_new_video_clip(**video_inputs, start_idx=frame_cnt)
        end_time = time.perf_counter()
        video_clip_shape = np.array(video_clip).shape
        if frame_cnt > 0:
            time_meter.add('memory_latency', end_time - start_time)
            time_meter.add('memory_latency_encoder', time_list[2] - time_list[1] + time_list[6] - time_list[5])
            time_meter.add('memory_latency_readwrite', time_list[3] - time_list[2] + time_list[7] - time_list[6])
            time_meter.add('memory_latency_cluster', time_list[4] - time_list[3])
            time_meter.add('memory_latency_retrieve', time_list[5] - time_list[4])
            logger.info(f'[MemManager] End embedding, embedded frames {video_clip_shape},\tidx={frame_cnt},\tmemory_latency={time_meter["memory_latency"]}')
            logger.info(f'[MemManager] times={[time_list[i + 1] - time_list[i] for i in range(7)]}')
            for name in time_meter._metrics:
                logger.info(f'[MemManager] Metrics: {name}={time_meter[name]}')
        else:
            logger.info(f'[MemManager] End embedding, embedded frames {video_clip_shape},\tidx={frame_cnt},\tmemory_latency={end_time - start_time:.6f}, not logged')
        frame_cnt += video_clip_shape[0]
    logger.info(f'[MemManager] end')

def main(args):
    torch.multiprocessing.set_start_method('spawn', force=True)

    log_queue = Queue()
    frame_queue = Queue(maxsize=10)
    processes = []

    ############## Start listener process #############
    p1 = Process(target=listener, args=(log_queue, args.log_file))
    processes.append(p1)
    p1.start()

    ############## Start main process #############
    worker_configurer(log_queue)
    logger = logging.getLogger(__name__)

    model_path = args.model_path
    model_config = FlashVStreamQwen2VLConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if args.flash_memory_dict is not None:
        model_config.vision_config.flash_memory_config = args.flash_memory_dict
        
    if getattr(model_config.vision_config, 'flash_memory_config', None) is None:
        logger.warn(f'[main] Qwen2VLVisionConfig.flash_memory_config is not set. Set it to default, sample 10000')
        model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG

    model = FlashVStreamQwen2VLModel.from_pretrained(
        model_path, 
        config=model_config,
        device_map="cuda", 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).eval()
    processor = FlashVStreamQwen2VLProcessor.from_pretrained(model_path)
    if args.flash_memory_dict is not None:
        flash_memory_config = args.flash_memory_dict
        logger.info(f"[main] Load processor success!, using new processor={flash_memory_config}, instead of training time={model.config.vision_config.flash_memory_config}")
    else:
        flash_memory_config = model.config.vision_config.flash_memory_config
        logger.info(f"[main] Load processor success!, processor with flash_memory_config={flash_memory_config}")

    with Manager() as manager:
        image_tensor = None
        model.use_video_streaming_mode = True
        model.video_embedding_memory = manager.list()

        # video_fps = 0.1, 视频帧率，每10秒抽1帧
        # play_speed = 1.0, 倍速播放，1倍速即每10秒保存1帧的feature到上述地址model.video_embedding_address
        logger.info(f'[main] Important: set video_fps = {args.video_fps}')
        logger.info(f'[main] Important: set play_speed = {args.play_speed}')

        ############## Start simulator process #############
        p2 = Process(target=video_stream_similator, 
                     args=(args.video_file, frame_queue, log_queue, args.video_fps, args.play_speed))
        processes.append(p2)
        p2.start()

        ############## Start memory manager process #############
        p3 = Process(target=frame_memory_manager, 
                     args=(model, processor, flash_memory_config, frame_queue, log_queue))
        processes.append(p3)
        p3.start()


        # 启动server
        start_time = datetime.now()
        time_meter = MetricMeter()
        conv_cnt = 0
        time.sleep(10)
        while True:
            time.sleep(10)

            cuda_list = model.get_video_embedding_memory_cuda_list()
            if cuda_list is None or len(cuda_list) == 0:
                logger.info(f'[main] cuda_list is empty, skip')
                continue
           # else:
                # video_embed_shape = cuda_list[-1]
                # 计算实际的 video token 数量 ####### 修改部分
                #total_tokens = 0
                #for embed_tensor in cuda_list:
                 # if hasattr(embed_tensor, 'shape'):
                 #   total_tokens += embed_tensor.shape[0]  # 累加所有视频 token
    
                #video_embed_size = total_tokens // 4  # 除以4是因为每个 token 有4个sub token
                # logger.info(f'[main] cuda_list len={len(cuda_list)}, calculated video_embed_size={video_embed_size}')
                # video_embed_size = 10800 #### 修改方案1
                # logger.info(f'[main] cuda_list is not empty, len={len(cuda_list)}, set video_embed_size={video_embed_size}')
                
                # 从 cuda_list 中获取实际的 video_embeds 形状
                # cuda_list 结构：[tem_x, tem_thw, tem_weights, tem_timestamp, 
                #                  spa_x, spa_thw, spa_positions, x, thw, 
                #                  small_x, small_thw, video_embeds, video_embeds_shape]
                # video_embeds_shape = cuda_list[-1]  # 获取最后一个元素，即 video_embeds.shape
                
                # video_embeds_shape 是 torch.Size([1, actual_tokens, hidden_dim])
                # 我们需要的是第一维，而不是第二维（actual_tokens）
                #if isinstance(video_embeds_shape, (list, tuple, torch.Size)):
                #    actual_video_tokens = video_embeds_shape[0]
               # else:
                #    actual_video_tokens = int(video_embeds_shape)
                
                #logger.info(f'[main] cuda_list is not empty, len={len(cuda_list)}, video_embeds_shape={video_embeds_shape}, using actual_video_tokens={actual_video_tokens}')
            # 从内存中读取 tem_thw 和 spa_thw
            tem_thw = cuda_list[1]   # 索引1 = tem_thw
            spa_thw = cuda_list[5]   # 索引5 = spa_thw

            # 计算实际需要的 video pad token 数量（与 calc_am_rope 一致）
            tem_size = tem_thw.prod().item() // 4
            spa_size = spa_thw.prod().item() // 4
            actual_video_tokens = spa_size + tem_size  # = 8592
            
            # 关键：processor 内部会 // 4，所以这里要 * 4
            dummy_video_tokens_value = actual_video_tokens * 4  # = 34368

            logger.info(
                f"[main] tem_thw={tem_thw.tolist()}, spa_thw={spa_thw.tolist()}, "
                f"tem_size={tem_size}, spa_size={spa_size}, actual_video_tokens={actual_video_tokens}, "
                f"dummy_video_tokens_value={dummy_video_tokens_value}"
            )
            
            # 获取当前时间
            now = datetime.now()
            conv_start_time = time.perf_counter()
            # 将当前时间格式化为字符串
            current_time = now.strftime("%H:%M:%S")
            duration = now.timestamp() - start_time.timestamp()

            # 打印当前时间
            #inp = 'what is in the video?'
            inp = \
"""Please choose the correct answer from the options below, output the option letter (A, B, C, or D):
A. A person running a marathon and sharing their experience
B. A cooking tutorial showing how to make a special dish
C. A man is playing with a little girl
D. A dog training session in a park"""
            print("\nCurrent Time:", current_time, "Run for:", duration)
            
            inp = input("user: ")  # <-- 等待用户输入
            if inp.lower() in ['exit', 'quit', 'q']: #
              break #
            
            print(f"user: {inp}", end="\n")
            print(f"assistant: ", end="")
            # every conversation is a new conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<|vision_start|><|video_pad|><|vision_end|>" + inp},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            #text += "Best Option: (" # 不加上best option
            inputs = processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
                flash_memory_config=flash_memory_config,
                # dummy_video_tokens=video_embed_size * 4, #### 被修改
                dummy_video_tokens=dummy_video_tokens_value,  # processor 会 // 4 得到 8592
            )
            try:
                input_len = int(inputs.input_ids.shape[1])
                attn_len = int(inputs.attention_mask.sum().item()) if hasattr(inputs, "attention_mask") else -1
            except Exception:
                input_len, attn_len = -1, -1
            logger.info(
                f"[main] prompt_len={input_len}, attn_len={attn_len}, "
                f"max_pos={getattr(model.config, 'max_position_embeddings', 'n/a')}, "
                f"eos_id={getattr(model.config, 'eos_token_id', 'n/a')}"
            )
            inputs = inputs.to("cuda")

            llm_start_time = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
              generated_ids = model.generate(
                  **inputs,
                  max_new_tokens=256,
                  min_new_tokens=8,
                  use_cache=True,
                  temperature=0.7,
                  top_p=0.8,
                  repetition_penalty=1.1,
                  no_repeat_ngram_size=3,
              )
              llm_times = model.user_log_times if hasattr(model, 'user_log_times') else [llm_start_time, time.perf_counter()]
            llm_end_time = time.perf_counter()
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            try:
                gen_len = int(generated_ids_trimmed[0].shape[0])
                eos_id = getattr(model.config, 'eos_token_id', None)
                eos_hit = bool((generated_ids_trimmed[0] == eos_id).any().item()) if eos_id is not None else False
                logger.info(f"[main] gen_len={gen_len}, eos_hit={eos_hit}")
            except Exception as e:
                logger.info(f"[main] gen_len/eos check failed: {e}")
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs = outputs[0].strip()
            # 简单重复检测（用于定位输出卡住）
            try:
                toks = outputs.split()
                rep_ratio = 1.0 - (len(set(toks)) / max(1, len(toks)))
                logger.info(f"[main] output_len={len(outputs)}, token_count={len(toks)}, rep_ratio={rep_ratio:.3f}")
            except Exception:
                pass
            print(f"{outputs}", end="\n")
            conv_end_time = time.perf_counter()
            if conv_cnt > 0:
                time_meter.add('conv_latency', conv_end_time - conv_start_time)
                time_meter.add('llm_latency', llm_end_time - llm_start_time)
                time_meter.add('real_sleep', conv_start_time - last_conv_start_time)
                time_meter.add('llm_latency_memoryio', llm_times[1] - llm_times[0])
                logger.info(f'CliServer: idx={conv_cnt},\treal_sleep={time_meter["real_sleep"]},\tconv_latency={time_meter["conv_latency"]}')
                logger.info(f'CliServer: llm_latency={time_meter["llm_latency"]}')
                logger.info(f'CliServer: llm_latency_memoryio={time_meter["llm_latency_memoryio"]}')
            else:
                logger.info(f'CliServer: idx={conv_cnt},\tconv_latency={conv_end_time - conv_start_time},\tllm_latency={llm_end_time - llm_start_time}')
            conv_cnt += 1
            last_conv_start_time = conv_start_time

    # 强制所有子进程完成
    for p in processes:
        p.terminate()
    print("所有进程已完成.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="output/best_ckpt")
    parser.add_argument("--video-file", type=str, default="data/eval_video/videomme/frames/goyWFUzCqF4")

    parser.add_argument("--log-file", type=str, default="server_cli.log")
    parser.add_argument("--use_1process", action="store_true")
    parser.add_argument("--video_fps", type=float, default=0.5)
    parser.add_argument("--play_speed", type=float, default=1.0)

    parser.add_argument("--flash_memory_dict", type=str, default=None)
    args = parser.parse_args()
    args.flash_memory_dict = dict(
        flash_memory_temporal_length=40,  # 120 -》 40
        flash_memory_temporal_method='kmeans_ordered',
        flash_memory_temporal_poolsize=2,
        flash_memory_temporal_pca_dim=32,
        flash_memory_spatial_length=20, # 原来60，减少到20
        flash_memory_spatial_method='klarge_retrieve',
    )
    # args.model_path = 'ckpt/Qwen2-VL-2B-Instruct'
    main(args)
