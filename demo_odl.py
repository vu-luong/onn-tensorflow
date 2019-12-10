import os

from skmultiflow.data import FileStream
from evaluate_prequential import EvaluatePrequential

from data_helper import data_folder, file_list
from odl import OnlineDeepLearner

for file_name in file_list:
    print('Run', file_name)

    # 1. Create a stream
    stream = FileStream(data_folder + '/' + file_name + '.csv')
    stream.prepare_for_use()

    learner = OnlineDeepLearner(seed=1)

    output_path = "result/{}/".format(type(learner).__name__)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 3. Setup the evaluator
    evaluator = EvaluatePrequential(pretrain_size=0,
                                    n_wait=1000,
                                    max_samples=5000000,
                                    output_file='{}/{}.csv'.format(output_path, file_name),
                                    metrics=['accuracy', 'kappa', 'kappa_t', 'kappa_m', 'running_time', 'model_size'])

    # 4. Run evaluation
    evaluator.evaluate(stream=stream, model=learner)
