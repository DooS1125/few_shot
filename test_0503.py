import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Data import *
from Sampler import *
from models.Proto import *
from utils.utils import *


def inference(model, test_loader, args):
    model.eval()
    test_loss = []
    test_acc =[]
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            test_loss.append(loss.item())
            test_acc.append(acc)

        test_loss = np.mean(test_loss)
        test_acc = np.mean(test_acc)

        finish_time = time.time()
        epoch_time = finish_time - start_time
        print(f'Test Loss : [{test_loss:.4f}] Test ACC : [{test_acc:.4f}] Epoch_time : [{epoch_time:.0f}s]')

    return test_loss, test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--load', default='./weights/best.pt')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    testset = ESC_data('test')
    test_sampler = CategoriesSampler(testset.label, 30,
                                args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                        num_workers=0, pin_memory=True)

    model = Convnet().to(device)
    model.load_state_dict(torch.load(args.load))

    test_loss, test_acc = inference(model, test_loader, args)

