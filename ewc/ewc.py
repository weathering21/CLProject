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

if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device",device)
    benchmark = SplitMNIST(n_experiences=5,return_task_id=True)
    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes).to(device)

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
        # forgetting_metrics(experience=True, stream=True),
        # StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    ewc = EWCPlugin(ewc_lambda=0.001)
    replay = ReplayPlugin(mem_size=100)
    cl_strategy = Naive(
        model, SGD(model.parameters(), lr=0.001, momentum=0.9),
        CrossEntropyLoss(), train_mb_size=100, train_epochs=1, eval_mb_size=100,
        evaluator=eval_plugin, device=device,plugins=[ewc,replay]
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("current task id:", experience.task_label)
        print("Current Classes: ", experience.classes_in_this_experience)
        # train returns a dictionary which contains all the metric values
        for data in experience.dataset:
            inputs, labels,other = data
            inputs = inputs.to(device)
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