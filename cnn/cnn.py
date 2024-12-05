import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive
from avalanche.training.plugins import EWCPlugin, ReplayPlugin
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)
    benchmark = SplitMNIST(n_experiences=5)
    # MODEL CREATION
    model = Net()
    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    tb_logger = TensorboardLogger()
    text_logger = TextLogger(open('log.txt', 'a'))
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        # accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        accuracy_metrics(minibatch=True, epoch=True, experience=True),
        loss_metrics(minibatch=True, epoch=True, experience=True),
        # timing_metrics(epoch=True),
        # cpu_usage_metrics(experience=True),
        forgetting_metrics(experience=True, stream=True),
        # StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )
    cl_strategy = Naive(
        model, SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=100,
        evaluator=eval_plugin, device=device
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        # train returns a dictionary which contains all the metric values
        print("len",len(experience.dataset[0]))
        print(f"\tSample: {experience.dataset[0]}")
        for data in experience.dataset:
            inputs, labels,other = data
            print("Labels size:", labels)
            print("Inputs size:", inputs.size())
            break
        res = cl_strategy.train(experience, num_workers=4)
        print("valid",next(model.parameters()).device)
        for metric_name, metric_value in res.items():
            print(f"{metric_name}: {metric_value}")
        print('Training completed')
        print('Computing accuracy on the whole test set')        # eval also returns a dictionary which contains all the metric values
        res = cl_strategy.eval(benchmark.test_stream, num_workers=4)
        for metric_name, metric_value in res.items():
            print(f"{metric_name}: {metric_value}")
        results.append(res)
    with open('ewc_replay_results.txt', 'w') as f:
        for res in results:
            f.write(str(res) + '\n')