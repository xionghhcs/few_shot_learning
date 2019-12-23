from torch import nn, optim, autograd
from torch.autograd import Variable
import torch
import os
from sklearn.metrics import accuracy_score
import numpy as np


class SimNet(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, inputs):
        in1 = inputs[0]
        in2 = inputs[1]

        in1_rep = self.encoder(in1)
        in2_rep = self.encoder(in2)

        # 曼哈顿距离
        man_dist = torch.sum(torch.abs(in1_rep - in2_rep), -1)

        sim_score = torch.exp(-man_dist)

        return man_dist, sim_score


class SimNetFewShotFramework:
    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
    
    def __load_model__(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print('Successfully loaded checkpoint {}'.format(ckpt))
            return checkpoint
        else:
            raise Exception("No checkpoint found at {}".format(ckpt))

    def train(self, model, 
              model_name,
              N,K,Q,
              ckpt_dir='./checkpoint',
              batch_size=8,
              learning_rate=0.1,
              lr_step_size=20000,
              weight_decay=1e-5, 
              val_step=1000,
              train_iter=30000,
              pretrain_model=None,
              optimizer=optim.SGD,
              cuda=True):
        # Init
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        bceloss = nn.BCELoss()

        if cuda:
            model = model.cuda()
        
        model.train()

        best_acc = 0

        epochs = 20
        best_acc = 0

        total_loss = 0.0
        total_acc = 0.0
        cnt = 0
        for it in range(train_iter):

            s1_batch, s2_batch, label_batch = self.train_data_loader.next_batch_for_one_shot(20)

            s1_batch = Variable(torch.from_numpy(s1_batch).long())
            s2_batch = Variable(torch.from_numpy(s2_batch).long())
            label_batch = Variable(torch.from_numpy(label_batch).float())

            if cuda:
                s1_batch = s1_batch.cuda()
                s2_batch = s2_batch.cuda()
                label_batch = label_batch.cuda()

            logit, prob = model([s1_batch, s2_batch])

            prob = prob.squeeze(-1)
            logit = logit.squeeze(-1)

            batch_loss = bceloss(prob, label_batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # compute metrics on training data
            acc = accuracy_score(label_batch.cpu().numpy().astype(np.int32), np.where(prob.detach().cpu().numpy() > 0.5, 1, 0))

            total_loss += batch_loss.item()
            total_acc += acc
            cnt += 1

            print('Iter:{} | loss:{:.4f} | acc:{:.4f}\r'.format(it, total_loss / cnt, total_acc / cnt), end='')

            if it % 2000 == 0:
                val_acc = self.eval(model, self.val_data_loader, N ,K, Q)
                model.train()

                if val_acc > best_acc:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = val_acc

        print()

        print('#' * 20)
        print('Finish Training!')
        # 加载最优模型
        checkpoint = self.__load_model__(ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])

        val_acc = self.eval(model, self.val_data_loader, N, K, Q)
        print('Dev accuracy: {:.4f}'.format(val_acc))

        test_acc = self.eval(model, self.test_data_loader, N, K, Q)
        print('Test accuracy: {:.4f}'.format(test_acc))

    def eval(self, model, data_loader, C=5, K=1, Q=10, eval_iter=20):
        """
        support set 中有N个类，每个类有K个样本, 每个类设置10个
        """
        print()
        model.eval()
        total_acc = 0.0
        cnt = 0

        for it in range(eval_iter):

            s, q, label = data_loader.next_batch_for_one_shot_val(C, K, Q)

            q = q[:, np.newaxis, :]
            q = np.tile(q, [1, C*K, 1])
            q = np.reshape(q, [-1, data_loader.max_length])

            s = np.reshape(s, [1, C*K, data_loader.max_length])
            s = np.tile(s, [C*Q, 1, 1])
            s = np.reshape(s, [-1, data_loader.max_length])
            
            q = Variable(torch.from_numpy(q).long()).cuda()
            s = Variable(torch.from_numpy(s).long()).cuda()

            _, score = model([q, s])

            score = score.detach().cpu().numpy()

            score = np.reshape(score, [C*Q, C, K])

            pred = np.argmax(np.mean(score, -1), 1)

            # pred = []

            # for i in range(q.shape[0]):
            #     tmp_s = np.reshape(s, (-1, data_loader.max_length))
            #     tmp_s = Variable(torch.from_numpy(tmp_s).long())
            #     tmp_s = tmp_s.cuda()

            #     tmp_q = q[i]
            #     tmp_q = np.tile(tmp_q, (tmp_s.size()[0], 1))
            #     tmp_q = Variable(torch.from_numpy(tmp_q).long())
            #     tmp_q = tmp_q.cuda()

            #     _, score = model([tmp_s, tmp_q])

            #     score = score.detach().cpu().numpy()
            #     score = np.reshape(score, (C, K))
            #     score = np.mean(score, axis=1)

            #     res = np.argmax(score, axis=0)
            #     pred.append(res)
            
            # pred = np.array(pred)

            acc = accuracy_score(label, pred)
            total_acc += acc
            cnt += 1
            print('[EVAL {}-Way {}-Shot] step:{} | acc: {:.4f}\r'.format(C, K, it, total_acc / cnt), end='')

        print()
        return total_acc / cnt
