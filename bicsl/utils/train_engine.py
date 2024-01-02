# --------------------------------------------------------
# BiCSL
# Written by Yuwei Sun https://github.com/yuweisunn
# Adapted from https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os, torch, datetime, shutil, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from openvqa.models.model_loader import ModelLoader
from openvqa.utils.optim import get_optim, adjust_lr
from utils.test_engine import test_engine, ckpt_proc
import info_nce
import re, json
import torch


def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
           return loss
        elif reduction == "mean":
           return loss.mean()


class ansNet(nn.Module):
    def __init__(self,__C,token_size,answer_size):
        super(ansNet, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        #if __C.USE_GLOVE:
        #    self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(__C.WORD_EMBED_SIZE, answer_size)
        self.pool = nn.MaxPool1d(8, stride=1)
        self.fc2 = nn.Linear(answer_size, 256)
        
        #self.proj_norm = nn.LayerNorm(answer_size)

    def forward(self, ans):
        #ans_feat = self.embedding(ans)
        #ans_feat, _ = self.lstm(ans_feat)
        #ans_feat = torch.flatten(ans, start_dim=1)
        proj_ans_feat = self.fc(ans)
        proj_ans_feat = F.relu(proj_ans_feat)
        proj_ans_feat = self.pool(proj_ans_feat.transpose(1, 2))
        proj_ans_feat = self.fc2(torch.squeeze(proj_ans_feat,2))
        #proj_ans_feat = torch.squeeze(proj_ans_feat,2) 
        
        return proj_ans_feat


def train_engine(__C, dataset, dataset_eval=None):
    ans_embedds = torch.load('ans.pt')
    criterion = info_nce.InfoNCE(temperature = 0.07).cuda() #NT_Xent(temperature = 0.07).cuda()
    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb
    

    client = 2
    net_list = []
    ansnet_list = []
    optim_list =[]
    optim_ans_list = []
    for i in range(client):
        #print(__C.WORD_EMBED_SIZE)
        net = ModelLoader(__C).Net(
            __C,
            pretrained_emb,
            token_size,
            512
            #ans_size
        )
    
        net.cuda()
        net.train()

        ansnet = ansNet(__C,
            token_size,
            512
            #ans_size
            )
        ansnet.cuda()
        ansnet.train()


        if __C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=__C.DEVICES)
            ansnet = nn.DataParallel(ansnet, device_ids=__C.DEVICES)

        optim_list.append(get_optim(__C, net, data_size))
        optim_ans_list.append(get_optim(__C, ansnet, data_size)) #define the adapter net

        net_list.append(net)
        ansnet_list.append(ansnet)




    # Define Loss Function
    loss_fn = eval('torch.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] + "(reduction='" + __C.LOSS_REDUCTION + "').cuda()")

    # Load checkpoint if resume training
    if __C.RESUME:
        print(' ========== Resume training')

        if __C.CKPT_PATH is not None:
            print('Warning: Now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = __C.CKPT_PATH
        else:
            path = __C.CKPTS_PATH + \
                   '/ckpt_' + __C.CKPT_VERSION + \
                   '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

        # Load the network parameters
        print('Loading ckpt from {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')

        if __C.N_GPU > 1:
            net.load_state_dict(ckpt_proc(ckpt['state_dict']))                        
            ansnet.load_state_dict(ckpt['state_dict_ans'])

        else:
            net.load_state_dict(ckpt['state_dict'])
            ansnet.load_state_dict(ckpt['state_dict_ans'])

        start_epoch = ckpt['epoch']

        # Load the optimizer paramters
        optim = get_optim(__C, net, data_size, ckpt['lr_base'])
        optim._step = int(data_size / __C.BATCH_SIZE * start_epoch)
        optim_ans = get_optim(__C, ansnet, data_size, ckpt['lr_base'])
        optim_ans._step = int(data_size / __C.BATCH_SIZE * start_epoch)
        optim.optimizer.load_state_dict(ckpt['optimizer'])
        optim_ans.optimizer.load_state_dict(ckpt['optimizer_ans'])    

        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

    else:
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            #shutil.rmtree(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
        start_epoch = 0

    loss_sum = 0
    named_params = list(net.named_parameters())
    grad_norm = np.zeros(len(named_params))

    # Define multi-thread dataloader
    # if __C.SHUFFLE_MODE in ['external']:
    #     dataloader = Data.DataLoader(
    #         dataset,
    #         batch_size=__C.BATCH_SIZE,
    #         shuffle=False,
    #         num_workers=__C.NUM_WORKERS,
    #         pin_memory=__C.PIN_MEM,
    #         drop_last=True
    #     )
    # else:
    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.BATCH_SIZE,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM,
        drop_last=True
    )

    dataloader_eval = Data.DataLoader(
        dataset_eval,
        batch_size=__C.EVAL_BATCH_SIZE,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM
    )

    logfile = open(
        __C.LOG_PATH +
        '/log_run_' + __C.VERSION + '.txt',
        'a+'
    )
    logfile.write(str(__C))
    logfile.close()

    # Training script
    for epoch in range(start_epoch, __C.MAX_EPOCH):

        # Save log to file
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            '=====================================\nnowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()

        # Learning Rate Decay
        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)

        # Externally shuffle data list
        # if __C.SHUFFLE_MODE == 'external':
        #     dataset.shuffle_list(dataset.ans_list)

        time_start = time.time()

        for c in range(client):
            net = net_list[c]
            ansnet = ansnet_list[c]
            optim = optim_list[c]
            optim_ans = optim_ans_list[c]

            # Iteration
            for step, (
                    frcn_feat_iter,
                    grid_feat_iter,
                    bbox_feat_iter,
                    ques_ix_iter,
                    ans_iter
            ) in enumerate(dataloader):
                if step == len(dataloader)//2:
                    break
                
                #print(ansnet.state_dict())
                correct = 0 
                optim.zero_grad()
                optim_ans.zero_grad()

                frcn_feat_iter = frcn_feat_iter.cuda()
                grid_feat_iter = grid_feat_iter.cuda()
                bbox_feat_iter = bbox_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_iter = ans_iter.cuda()

                loss_tmp = 0
                for accu_step in range(__C.GRAD_ACCU_STEPS):
                    loss_tmp = 0

                    sub_frcn_feat_iter = \
                        frcn_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * __C.SUB_BATCH_SIZE]
                    sub_grid_feat_iter = \
                        grid_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * __C.SUB_BATCH_SIZE]
                    sub_bbox_feat_iter = \
                        bbox_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * __C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * __C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * __C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * __C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * __C.SUB_BATCH_SIZE]
                    sub_ans_iter = torch.argmax(sub_ans_iter, dim =1)
                    pred_iq  = net(
                        sub_frcn_feat_iter,
                        sub_grid_feat_iter,
                        sub_bbox_feat_iter,
                        sub_ques_ix_iter                 
                        )
                    #pred_iq = F.normalize(pred_iq)
                    #print(sub_ques_ix_iter)
                    #print(sub_ans_iter.shape)
                    #print(pred_ans)
                    
                    #sub_ans_iter = torch.argmax(sub_ans_iter, dim =1)

                    sub_ans_words =[]
                    for a in sub_ans_iter:
                        sub_ans_words.append(ans_embedds[a])

                    pred_ans = ansnet(torch.stack(sub_ans_words))
                    loss = criterion(pred_iq, pred_ans)

                    loss.backward()
                    
                    optim.step()
                    optim_ans.step()
                    
                    # Training acc 
                    pred_iq  = net(
                        sub_frcn_feat_iter,
                        sub_grid_feat_iter,
                        sub_bbox_feat_iter,
                        sub_ques_ix_iter
                        )
                    
                    # Evaluate over all answers
                    ans_embedds = ans_embedds.cuda()
                    pred_ans_all = ansnet(ans_embedds)
                    
                    # Evaluate within the batch
                    #sub_ans_iter = torch.squeeze(sub_ans_iter,0)
                    
                    #embedds_batch = []
                    #for a in sub_ans_iter:
                    #    embedds_batch.append(ans_embedds[a])
                    #pred_ans_all = ansnet(torch.stack(embedds_batch))

                    pred_np = pred_iq.cpu().data
                    pred_ans_all = pred_ans_all.cpu().data
                   
                    _, indices = (100.0 * pred_np @ pred_ans_all.T).softmax(dim=-1).topk(1)
                    
                    preds = []
                    for j in indices:
                        # modified for batch-wise eval
                        preds.append(j[0])
                    pred_argmax = np.array((preds))
                    
                    for i in range(pred_argmax.shape[0]):
                        if pred_argmax[i] == sub_ans_iter[i]:
                            correct = correct+1 
                    
                    loss_tmp += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                    loss_sum += loss.cpu().data.numpy() * __C.GRAD_ACCU_STEPS
                
                
                if __C.VERBOSE:
                    #print("print something")
                    if dataset_eval is not None:
                        mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
                    else:
                        mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']

                    print("\r[Version %s][Model %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e, Acc: %.3f, correct sum: %s" % (
                        __C.VERSION,
                        __C.MODEL_USE,
                        __C.DATASET,
                        epoch + 1,
                        step,
                        int(data_size / __C.BATCH_SIZE),
                        mode_str,
                        loss_tmp / __C.SUB_BATCH_SIZE,
                        optim._rate,
                        correct/__C.SUB_BATCH_SIZE,
                        correct
                    ), end='          ')

                # Gradient norm clipping
                if __C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        __C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * __C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                
            time_end = time.time()
            elapse_time = time_end-time_start
            print('Finished in {}s'.format(int(elapse_time)))
            epoch_finish = epoch + 1

            # Save checkpoint
            if __C.N_GPU > 1:
                state = {
                    'state_dict': net.module.state_dict(),
                    'state_dict_ans': ansnet.state_dict(),
                    'optimizer': optim.optimizer.state_dict(),
                    'optimizer_ans': optim_ans.optimizer.state_dict(),
                    'lr_base': optim.lr_base,
                    'epoch': epoch_finish
                }
            else:
                state = {
                    'state_dict': net.state_dict(),
                    'state_dict_ans': ansnet.state_dict(),
                    'optimizer': optim.optimizer.state_dict(),
                    'optimizer_ans': optim_ans.optimizer.state_dict(),
                    'lr_base': optim.lr_base,
                    'epoch': epoch_finish
                }
            torch.save(
                state,
                __C.CKPTS_PATH +
                '/ckpt_' + __C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            # Logging
            logfile = open(
                __C.LOG_PATH +
                '/log_run_' + __C.VERSION + '.txt',
                'a+'
            )
            logfile.write(
                'Epoch: ' + str(epoch_finish) +
                ', Loss: ' + str(loss_sum / data_size) +
                ', Lr: ' + str(optim._rate) + '\n' +
                'Elapsed time: ' + str(int(elapse_time)) + 
                ', Speed(s/batch): ' + str(elapse_time / step) +
                '\n\n'
            )
            logfile.close()
            """
            data_size = dataset_eval.data_size
            correct = 0
            for step, (
                frcn_feat_iter,
                grid_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,
                ans_iter
            ) in enumerate(dataloader_eval):
              print("\rEvaluation: [step %4d/%4d]" % (
                step,
                int(data_size / __C.EVAL_BATCH_SIZE),
              ), end='          ')

              frcn_feat_iter = frcn_feat_iter.cuda()
              grid_feat_iter = grid_feat_iter.cuda()
              bbox_feat_iter = bbox_feat_iter.cuda()
              ques_ix_iter = ques_ix_iter.cuda()
              ans_iter = ans_iter.cuda()

              pred = net(
                frcn_feat_iter,
                grid_feat_iter,
                bbox_feat_iter,
                ques_ix_iter
              )

              #ans_embedds = vector_ans(sub_ans_words)
              ans_embedds = ans_embedds.cuda()
              #sub_ans_words = torch.stack(sub_ans_words).cuda()
              pred_ans = ansnet(ans_embedds)

              pred_np = pred.cpu().data
              pred_ans = pred_ans.cpu().data

              _, indices = (100.0 * pred_np @ pred_ans.T).softmax(dim=-1).topk(1)
              preds = []
              for j in indices:
                preds.append(j[0])
              pred_argmax = np.array((preds))
              for i in range(pred_argmax.shape[0]):
                if pred_argmax[i] == ans_iter[i]:
                    correct = correct+1
              #print(f'correct: {correct}')
            print(f'val acc: {correct/data_size}')  
            """


        global_sd = net_list[0].state_dict()

        for key in global_sd:
              global_sd[key] = torch.sum(torch.stack([model.state_dict()[key] for m, model in enumerate(net_list)]), axis = 0)/client
        # update the global model
        net_list[0].load_state_dict(global_sd) 
        net_list[1].load_state_dict(global_sd) 


        # Eval after every epoch
        if dataset_eval is not None:
            test_engine(
                __C,
                dataset_eval,
                state_dict=net_list[0].state_dict(),
                ans_state_dict=ansnet_list[0].state_dict(),
                validation=True
            )
        # if self.__C.VERBOSE:
        #     logfile = open(
        #         self.__C.LOG_PATH +
        #         '/log_run_' + self.__C.VERSION + '.txt',
        #         'a+'
        #     )
        #     for name in range(len(named_params)):
        #         logfile.write(
        #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
        #                 str(name),
        #                 named_params[name][0],
        #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
        #             )
        #         )
        #     logfile.write('\n')
        #     logfile.close()

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))


           
