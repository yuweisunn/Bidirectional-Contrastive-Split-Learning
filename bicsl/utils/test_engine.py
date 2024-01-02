# --------------------------------------------------------
# BiCSL
# Written by Yuwei Sun https://github.com/yuweisunn
# Adapted from https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os, json, torch, pickle
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from openvqa.models.model_loader import ModelLoader
from openvqa.datasets.dataset_loader import EvalLoader
import torch.nn.functional as F
import info_nce



with open('openvqa/datasets/vqa/answer_dict.json') as json_file:
    ans_dict = json.load(json_file)
    words = list(ans_dict[0].keys())

def vector_ans(stat_ans_list):
    spacy_tool = None
    spacy_tool = en_vectors_web_lg.load()
    vectors = []
    vectors_list = []
    for ans in stat_ans_list:
        vectors = []
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ans.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            vectors.append(spacy_tool(word).vector)

        for i in range(8-len(words)):
            vectors.append([0] * 300)

        vectors_list.append(vectors)

    return torch.FloatTensor(vectors_list)


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


# Evaluation
@torch.no_grad()
def test_engine(__C, dataset, state_dict=None, ans_state_dict=None, validation=False):
    ans_embedds = torch.load('ans.pt')
    # Load parameters
    if __C.CKPT_PATH is not None:
        print('Warning: you are now using CKPT_PATH args, '
              'CKPT_VERSION and CKPT_EPOCH will not work')

        path = __C.CKPT_PATH
    else:
        path = __C.CKPTS_PATH + \
               '/ckpt_' + __C.CKPT_VERSION + \
               '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

    # val_ckpt_flag = False
    if state_dict is None:
        # val_ckpt_flag = True
        print('Loading ckpt from: {}'.format(path))
        state_dict = torch.load(path)['state_dict']
        print('Finish!')

        if __C.N_GPU > 1:
            state_dict = ckpt_proc(state_dict)

    # Store the prediction list
    # qid_list = [ques['question_id'] for ques in dataset.ques_list]
    ans_ix_list = []
    pred_list = []

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        512
        #ans_size
    )
    net.cuda()
    net.eval()

    ansnet = ansNet(__C, token_size, 512)#ans_size)
    ansnet.cuda()
    ansnet.eval()

    if __C.N_GPU >1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)
        ansnet = nn.DataParallel(ansnet, device_ids=__C.DEVICES)
        
    net.load_state_dict(state_dict)
    ansnet.load_state_dict(ans_state_dict)

    dataloader = Data.DataLoader(
        dataset,
        batch_size=__C.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM
        )
    correct = 0
    for step, (
            frcn_feat_iter,
            grid_feat_iter,
            bbox_feat_iter,
            ques_ix_iter,
            ans_iter
    ) in enumerate(dataloader):
        print("\rEvaluation: [step %4d/%4d]" % (
            step,
            int(data_size / __C.EVAL_BATCH_SIZE),
        ), end='          ')

        frcn_feat_iter = frcn_feat_iter.cuda()
        grid_feat_iter = grid_feat_iter.cuda()
        bbox_feat_iter = bbox_feat_iter.cuda()
        ques_ix_iter = ques_ix_iter.cuda()
        ans_iter = ans_iter.cuda()
        sub_ans_iter = torch.argmax(ans_iter, dim = 0)
            
        #if step == 0:
        #    print(ans_iter)
        #    print(sub_ans_iter)
        
        pred_iq = net(
            frcn_feat_iter,
            grid_feat_iter,
            bbox_feat_iter,
            ques_ix_iter
        )


        #break
        
        # Evaluate over all answers
        ans_embedds = ans_embedds.cuda()
        pred_ans = ansnet(ans_embedds)
        
        # Evaluate over answers within a batch 
        #sub_ans_iter = torch.squeeze(sub_ans_iter, 0)
        #embedd_batch = []
        #for a in sub_ans_iter:
        #    embedd_batch.append(ans_embedds[a])
        #pred_ans = ansnet(torch.stack(embedd_batch))
  
        pred_iq = pred_iq.cpu().data
        pred_ans = pred_ans.cpu().data
        _, indices = (100.0 * pred_iq @ pred_ans.T).softmax(dim=-1).topk(1)
        preds = []
        for j in indices:
            #preds.append(sub_ans_iter[j[0]].cpu().data)
            preds.append(j[0])
        pred_argmax = np.array((preds))
        
         
        if step == 0:
            print("check data realiablity")
            #print(sub_ans_iter)
            print(pred_argmax)
        
        #for i in range(pred_argmax.shape[0]):
        #    if pred_argmax[i] == sub_ans_iter[i]:
        #        correct = correct+1
        
        #print(f'correct: {correct}') 
    

        # Save the answer index
        if pred_argmax.shape[0] != __C.EVAL_BATCH_SIZE:
            pred_argmax = np.pad(
                pred_argmax,
                (0, __C.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                mode='constant',
                constant_values=-1
            )

        ans_ix_list.append(pred_argmax)
    
        # Save the whole prediction vector
        if __C.TEST_SAVE_PRED:
            if pred_np.shape[0] != __C.EVAL_BATCH_SIZE:
               pred_np = np.pad(
                    pred_np,
                    ((0, __C.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                    mode='constant',
                    constant_values=-1
                )

            pred_list.append(pred_np)

    print('')
    ans_ix_list = np.array(ans_ix_list).reshape(-1)
    #print(f'val acc: {correct/data_size}')            

    if validation:
        if __C.RUN_MODE not in ['train']:
            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.CKPT_VERSION
        else:
            result_eval_file = __C.CACHE_PATH + '/result_run_' + __C.VERSION
    else:
        if __C.CKPT_PATH is not None:
            result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION
        else:
            result_eval_file = __C.RESULT_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH)


    if __C.CKPT_PATH is not None:
        ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '.pkl'
    else:
        ensemble_file = __C.PRED_PATH + '/result_run_' + __C.CKPT_VERSION + '_epoch' + str(__C.CKPT_EPOCH) + '.pkl'


    if __C.RUN_MODE not in ['train']:
        log_file = __C.LOG_PATH + '/log_run_' + __C.CKPT_VERSION + '.txt'
    else:
        log_file = __C.LOG_PATH + '/log_run_' + __C.VERSION + '.txt'
    EvalLoader(__C).eval(dataset, ans_ix_list, pred_list, result_eval_file, ensemble_file, log_file, validation)


def ckpt_proc(state_dict):
    state_dict_new = {}
    for key in state_dict:
        state_dict_new['module.' + key] = state_dict[key]
        # state_dict.pop(key)

    return state_dict_new
