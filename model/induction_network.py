import torch

from torch import nn, optim
from torch.autograd import Variable

import sys
import os

class DynamicRouting(nn.Module):
    def __init__(self, iterations, feature_dim, C, K):
        super().__init__()
        self.epsilon = 1e-9
        self.iterations = iterations
        self.feature_dim = feature_dim
        self.W = nn.Linear(feature_dim, feature_dim, bias=True)
    
    def __squash(self, x, dim):
        x_squared_norm = torch.sum(torch.pow(x, 2), dim, keepdim=True)
        scalar_factor = x_squared_norm/(1 + x_squared_norm) / torch.sqrt(self.epsilon + x_squared_norm)
        x_ret = scalar_factor * x
        return x_ret
    
    def forward(self, support_feature, C, K):
        '''
        dynamic routing only accept one batch support set
        support_feature: (C, K, dim)        
        '''
        b = Variable(torch.zeros(C, K), requires_grad=False).cuda()
        support_feature = support_feature.view(-1, self.feature_dim)
        e_ij = self.W(support_feature)
        e_ij = e_ij.view(C, K, self.feature_dim)    #（C, K, dim)
        e_hat_ij = self.__squash(e_ij, dim=2)   #(C, K, dim)

        for i in range(self.iterations):
            di = torch.softmax(b, 1)   #(C, K)
            di = di.unsqueeze(-1)   #(C, K, 1)
            c_hat_i = e_hat_ij * di #（C, K, dim)
            c_hat_i = torch.sum(c_hat_i, 1) #(C, dim)
            c_i = self.__squash(c_hat_i, 1) #(C, dim)
            b = b + e_hat_ij.bmm(c_i.unsqueeze(2)).squeeze()
        return c_i


class RelationModule(nn.Module):
    def __init__(self, h, feature_dim):
        super().__init__()
        self.h = h
        self.feature_dim =  feature_dim
        self.M = Variable(torch.FloatTensor(feature_dim, feature_dim, h), requires_grad=True).cuda()
    
    def forward(self, q, s):
        '''
        q: (CQ, dim)
        s: (C, dim)
        '''
        result = []
        for i in range(self.h):
            m = self.M[:,:,i]
            qs = torch.matmul(q, m) #(CQ, dim)
            qs = torch.matmul(q, s.transpose(1, 0)) #(CQ, C)
            result.append(qs)
        ret = torch.stack(result, 2)
        return ret


class InductionNetwork(nn.Module):

    def __init__(self, encoder, C, K, h=100):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.hidden_size
        self.h = h
        self.dr = DynamicRouting(3, encoder.hidden_size, C, K)
        self.relation_module = RelationModule(h, encoder.hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(h, 1),
            nn.Sigmoid()
        )

    def forward(self, support, query, C, K, Q):
        support_feature = self.encoder(support)
        query_feature = self.encoder(query)

        query_feature = query_feature.view(-1, C*Q, self.hidden_size)

        B = query_feature.size()[0]

        support_feature = support_feature.view(-1, C, K, self.hidden_size)

        dr_result = []
        
        for sub in support_feature:
            res = self.dr(sub, C, K)
            dr_result.append(res)
        
        support_rep = torch.stack(dr_result, 0)

        support_rep = support_rep.view(-1, C, self.hidden_size)

        score_result = []
        for i in range(B):
            s = support_rep[i, :,:]
            q = query_feature[i,:,:]
            score_result.append(self.relation_module(q, s))

        relation_score = torch.stack(score_result).view(-1, C, self.h)

        relation_score = self.fc(relation_score).squeeze(-1)

        pred = torch.argmax(relation_score, 1)
        return relation_score, pred
        
class InductionNetworkFewShotFramework:

    def __init__(self, train_data_loader, dev_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
    
    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    
    def __load_model__(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print('Successfully loaded checkpoint {}'.format(ckpt))
            return checkpoint
        else:
            raise Exception("No checkpoint found at {}".format(ckpt))
    
    def train(self, model,
              model_name,
              B, C, K, Q,
              ckpt_dir='./checkpoint',
              test_result_dir='./test_result',
              learning_rate=1e-3,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              cuda=True,
              pretrain_model=None,
              optimizer=optim.Adam):

        """
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        """
        # Init
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()
        model.train()

        # Training
        best_acc = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0

        mse_loss = nn.MSELoss()

        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            support, _, query, label = self.train_data_loader.next_batch(B, C, K, Q)

            support = Variable(torch.from_numpy(support).long().view(-1, self.train_data_loader.max_length))
            query = Variable(torch.from_numpy(query).long().view(-1, self.train_data_loader.max_length))
            label = Variable(torch.from_numpy(label).long())
            one_hot_label = torch.zeros(B*C*Q, C).scatter_(1, label.view(-1, 1), 1)

            if cuda:
                support = support.cuda()
                query = query.cuda()
                label = label.cuda()
                one_hot_label = one_hot_label.cuda()

            logits, pred = model(support, query, C, K, Q)

            loss = mse_loss(logits, one_hot_label)

            # right = accuracy_score(label.reshape(-1), pred)

            right = self.accuracy(pred, label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            torch.cuda.empty_cache()

            iter_loss += loss.data.item()
            iter_right += right.data.item()
            iter_sample += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, C, K, Q, val_iter, cuda=cuda)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = acc

        print("\n####################\n")
        print("Finish training " + model_name)
        test_acc = self.eval(model, B, C, K, Q, test_iter, ckpt=os.path.join(ckpt_dir, model_name + '.pth.tar'), cuda=cuda)
        print("Test accuracy: {}".format(test_acc))


    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            cuda,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        model.eval()
        if ckpt is None:
            eval_dataset = self.dev_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            support, _, query, label = eval_dataset.next_batch(B, N, K, Q)

            support = Variable(torch.from_numpy(support).long().view(-1, self.train_data_loader.max_length))
            query = Variable(torch.from_numpy(query).long().view(-1, self.train_data_loader.max_length))
            label = Variable(torch.from_numpy(label).long())

            if cuda:
                support = support.cuda()
                query = query.cuda()
                label = label.cuda()

            logits, pred = model(support, query, N, K, Q)
            right = self.accuracy(pred, label)
            iter_right += right.data.item()
            iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()
        print("")
        return iter_right / iter_sample
    