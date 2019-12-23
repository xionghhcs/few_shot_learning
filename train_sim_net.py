from model.sim_net import SimNet, SimNetFewShotFramework
from model.proto_net import ProtoNetwork, ProtoNetworkFewShotFramework
from model.proto_hatt import ProtoHATT, ProtoHATTFewShotFramework
from model.matching_network import MatchingNetwork, MatchingNetworkFewShotFramework
from model.relation_network import RelationNetwork, RelationNetworkFewShotFramework
from model.induction_network import InductionNetwork, InductionNetworkFewShotFramework

from model.encoder import Encoder

import utils
from data_loader import DataLoader


def main():

    model_name = 'induction_net'
    C = 5
    K = 5
    Q = 3

    vocab, embedding = utils.load_word2vec('data/tencent_embedding.txt')

    encoder = Encoder(embedding)

    train_data_loader = DataLoader('data/sample_data.json', vocab)
    dev_data_loader = DataLoader('data/sample_data.json', vocab)
    test_data_loader = DataLoader('data/sample_data.json', vocab)
    if model_name == 'sim_net':
        framework = SimNetFewShotFramework(train_data_loader, dev_data_loader, test_data_loader)
        model = SimNet(encoder)
        framework.train(model, 'SimNet', 5, 1, 5)
    elif model_name == 'matching_net':
        framework = MatchingNetworkFewShotFramework(train_data_loader, dev_data_loader, test_data_loader)
        model = MatchingNetwork(encoder)
        framework.train(model, 'matching_net', B=4, C=C, K=K, Q=Q)
    elif model_name == 'proto':
        framework = ProtoNetworkFewShotFramework(train_data_loader, dev_data_loader, test_data_loader)
        model = ProtoNetwork(encoder)
        framework.train(model, 'proto', B=4, C=C, K=K, Q=Q)
    elif model_name == 'proto_hatt':
        framework = ProtoHATTFewShotFramework(train_data_loader, dev_data_loader, test_data_loader)
        model = ProtoHATT(encoder, K)
        framework.train(model, 'proto_hatt', B=4, C=C, K=K, Q=Q)
    elif model_name == 'relation_net':
        framework = RelationNetworkFewShotFramework(train_data_loader, dev_data_loader, test_data_loader)
        model = RelationNetwork(encoder)
        framework.train(model, 'relation_net', B=4, C=C, K=K, Q=Q)
    elif model_name == 'induction_net':
        framework = InductionNetworkFewShotFramework(train_data_loader, dev_data_loader, test_data_loader)
        model = InductionNetwork(encoder, C=C, K=K)
        framework.train(model, 'induction_net', B=4, C=C, K=K, Q=Q)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
