import os
import sys
import json
import time
import numpy as np
sys.path.append('../..')
import torch.utils.data.dataloader as dataloader
import framework.logbase
import framework.run_utils

import caption.models.attention

import controlimcap.readers.imgsgreader as imgsgreader
import controlimcap.models.graphattn
import controlimcap.models.graphflow
import controlimcap.models.graphmemory
import controlimcap.models.flatattn

from controlimcap.models.graphattn import ATTNENCODER

from controlimcap.driver.common import build_parser, evaluate_caption

def main():
  parser = build_parser()
  parser.add_argument('--max_attn_len', type=int, default=10)
  parser.add_argument('--num_workers', type=int, default=0)
  opts = parser.parse_args()

  if opts.mtype == 'node':
    model_cfg = caption.models.attention.AttnModelConfig()
  elif opts.mtype == 'node.role':
    model_cfg = controlimcap.models.flatattn.NodeRoleBUTDAttnModelConfig()
  elif opts.mtype in ['rgcn', 'rgcn.flow', 'rgcn.memory', 'rgcn.flow.memory']:
      model_cfg = controlimcap.models.graphattn.GraphModelConfig()
  model_cfg.load(opts.model_cfg_file)
  max_words_in_sent = model_cfg.subcfgs['decoder'].max_words_in_sent

  path_cfg = framework.run_utils.gen_common_pathcfg(opts.path_cfg_file, is_train=opts.is_train)

  if path_cfg.log_file is not None:
    _logger = framework.logbase.set_logger(path_cfg.log_file, 'trn_%f'%time.time())
  else:
    _logger = None

  if opts.mtype == 'node':
    model_fn = controlimcap.models.flatattn.NodeBUTDAttnModel
  elif opts.mtype == 'node.role':
    model_fn = controlimcap.models.flatattn.NodeRoleBUTDAttnModel
  elif opts.mtype == 'rgcn':
    model_fn = controlimcap.models.graphattn.RoleGraphBUTDAttnModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len
  elif opts.mtype == 'rgcn.flow':
    model_fn = controlimcap.models.graphflow.RoleGraphBUTDCFlowAttnModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len
  elif opts.mtype == 'rgcn.memory':
    model_fn = controlimcap.models.graphmemory.RoleGraphBUTDMemoryModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len
  elif opts.mtype == 'rgcn.flow.memory':
    model_fn = controlimcap.models.graphmemory.RoleGraphBUTDMemoryFlowModel
    model_cfg.subcfgs[ATTNENCODER].max_attn_len = opts.max_attn_len

  _model = model_fn(model_cfg, _logger=_logger, 
    int2word_file=path_cfg.int2word_file, eval_loss=opts.eval_loss)

  if opts.mtype in ['node', 'node.role']:
    reader_fn = imgsgreader.ImageSceneGraphFlatReader
    collate_fn = imgsgreader.flat_collate_fn
  elif opts.mtype in ['rgcn', 'rgcn.memory']:
    reader_fn = imgsgreader.ImageSceneGraphReader
    collate_fn = imgsgreader.sg_sparse_collate_fn
  elif opts.mtype in ['rgcn.flow', 'rgcn.flow.memory']:
    reader_fn = imgsgreader.ImageSceneGraphFlowReader
    collate_fn = imgsgreader.sg_sparse_flow_collate_fn

  if opts.is_train:
    model_cfg.save(os.path.join(path_cfg.log_dir, 'model.cfg'))
    path_cfg.save(os.path.join(path_cfg.log_dir, 'path.cfg'))
    json.dump(vars(opts), open(os.path.join(path_cfg.log_dir, 'opts.cfg'), 'w'), indent=2)

    trn_dataset = reader_fn(path_cfg.name_file['trn'], path_cfg.mp_ft_file['trn'],
      path_cfg.obj_ft_dir['trn'], path_cfg.region_anno_dir['trn'], path_cfg.word2int_file, 
      max_attn_len=opts.max_attn_len, max_words_in_sent=max_words_in_sent, 
      is_train=True, return_label=True, _logger=_logger)
    trn_reader = dataloader.DataLoader(trn_dataset, batch_size=model_cfg.trn_batch_size, 
      shuffle=True, collate_fn=collate_fn, num_workers=opts.num_workers)
    val_dataset = reader_fn(path_cfg.name_file['val'], path_cfg.mp_ft_file['val'],
      path_cfg.obj_ft_dir['val'], path_cfg.region_anno_dir['trn'], path_cfg.word2int_file, 
      max_attn_len=opts.max_attn_len, max_words_in_sent=max_words_in_sent, 
      is_train=False, return_label=True, _logger=_logger)
    val_reader = dataloader.DataLoader(val_dataset, batch_size=model_cfg.tst_batch_size, 
      shuffle=True, collate_fn=collate_fn, num_workers=opts.num_workers)
    for data in val_reader:
        print(data)
        break
 
if __name__ == '__main__':
  main()
#python imgread.py E:\asg2cap-master\MSCOCO\results\ControlCAP\rgcn.flow.memory\mp.resnet101.ctrl.attn.X_101_32x8d.rgcn.2.lstm.layer.1.hidden.512.tie_embed.embed_first/model.json  E:\asg2cap-master\MSCOCO\results\ControlCAP\rgcn.flow.memory\mp.resnet101.ctrl.attn.X_101_32x8d.rgcn.2.lstm.layer.1.hidden.512.tie_embed.embed_first/path.json rgcn.flow.memory   --eval_loss --is_train --num_workers 0