import torch
from torch import nn, optim
from torch.autograd import Variable
import os
import sys


class RelationNetwork(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.hidden_size

        self.g = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, support, query, C, K, Q):
        support_feature = self.encoder(support)
        query_feature = self.encoder(query)

        support_feature = support_feature.view(-1, C, K, self.hidden_size)

        support_feature = torch.sum(support_feature, 2)

        support_feature_ext = support_feature.repeat(1, C * Q, 1)

        support_feature_ext = support_feature_ext.view(-1, self.hidden_size)

        query_feature_ext = query_feature.view(-1, 1, self.hidden_size)

        query_feature_ext = query_feature_ext.repeat(1, C, 1)

        query_feature_ext = query_feature_ext.view(-1, self.hidden_size)

        relation_pair = torch.cat([query_feature_ext, support_feature_ext], -1)

        relations = self.g(relation_pair).view(-1, C)
        pred = torch.argmax(relations, 1)
        return relations, pred


class RelationNetworkFewShotFramework:

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

        cel_loss = nn.CrossEntropyLoss()
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

            loss = mse_loss(logits.view(-1, C), one_hot_label)

            # right = accuracy_score(label.reshape(-1), pred)

            right = self.accuracy(pred, label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

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