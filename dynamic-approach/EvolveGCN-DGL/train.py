import argparse
import time
import dgl
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from dataset import EllipticDataset
from model import EvolveGCNO, EvolveGCNH
from utils import Measure
import numpy as np
import matplotlib.pyplot as plt

def train(args, device):
    elliptic_dataset = EllipticDataset(raw_dir=args.raw_dir,
                                       processed_dir=args.processed_dir,
                                       self_loop=True,
                                       reverse_edge=True)

    g, node_mask_by_time = elliptic_dataset.process()
    num_classes = elliptic_dataset.num_classes

    cached_subgraph = []
    cached_labeled_node_mask = []
    for i in range(len(node_mask_by_time)):
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        cached_subgraph.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata['label'] >= 0
        cached_labeled_node_mask.append(valid_node_mask)

    if args.model == 'EvolveGCN-O':
        model = EvolveGCNO(in_feats=int(g.ndata['feat'].shape[1]),
                           n_hidden=args.n_hidden,
                           num_layers=args.n_layers)
    elif args.model == 'EvolveGCN-H':
        model = EvolveGCNH(in_feats=int(g.ndata['feat'].shape[1]),
                           num_layers=args.n_layers)
    else:
        raise NotImplementedError('Unsupported model {}'.format(args.model))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100)
    feature_accumulator = []
    label_accumulator = []

    train_max_index = 30
    valid_max_index = 35
    test_max_index = 48
    time_window_size = args.n_hist_steps
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(',')]
    loss_class_weight = torch.Tensor(loss_class_weight).to(device)

    train_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    valid_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)
    test_measure = Measure(num_classes=num_classes, target_class=args.eval_class_id)

    test_res_f1 = 0
    test_precisions = []
    test_recalls = []
    test_f1s = []

    for epoch in range(args.num_epochs):
        model.train()
        for i in range(time_window_size, train_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            features = model(g_list)
            features = features[cached_labeled_node_mask[i]].detach().cpu().numpy()
            labels = cached_subgraph[i].ndata['label'][cached_labeled_node_mask[i]].detach().cpu().numpy()

            # Accumulate features and labels
            feature_accumulator.extend(features)
            label_accumulator.extend(labels)

            # Fit the RandomForestClassifier incrementally
            rf_classifier.fit(feature_accumulator, label_accumulator)

        # Train measure
        model.eval()
        for i in range(time_window_size, train_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            features = model(g_list)
            features = features[cached_labeled_node_mask[i]].detach().cpu().numpy()
            labels = cached_subgraph[i].ndata['label'][cached_labeled_node_mask[i]].detach().cpu().numpy()

            predictions = rf_classifier.predict_proba(features)  # Use predict_proba for probabilities

            predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            train_measure.append_measures(predictions_tensor, labels_tensor)

        # Get each epoch measures during training.
        cl_precision, cl_recall, cl_f1 = train_measure.get_total_measure()
        train_measure.update_best_f1(cl_f1, epoch)
        train_measure.reset_info()

        print("Train Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

        # Validation measure
        model.eval()
        for i in range(train_max_index + 1, valid_max_index + 1):
            g_list = cached_subgraph[i - time_window_size:i + 1]
            features = model(g_list)
            features = features[cached_labeled_node_mask[i]].detach().cpu().numpy()
            labels = cached_subgraph[i].ndata['label'][cached_labeled_node_mask[i]].detach().cpu().numpy()

            predictions = rf_classifier.predict_proba(features)

            predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            valid_measure.append_measures(predictions_tensor, labels_tensor)

        cl_precision, cl_recall, cl_f1 = valid_measure.get_total_measure()
        valid_measure.update_best_f1(cl_f1, epoch)
        valid_measure.reset_info()

        print("Eval Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
              .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

        if epoch - valid_measure.target_best_f1_epoch >= args.patience:
            print("Best eval Epoch {}, Cur Epoch {}".format(valid_measure.target_best_f1_epoch, epoch))
            break

        if epoch == valid_measure.target_best_f1_epoch:
            print("###################Epoch {} Test###################".format(epoch))
            timestep_precisions = []
            timestep_recalls = []
            timestep_f1s = []
            for i in range(valid_max_index + 1, test_max_index + 1):
                g_list = cached_subgraph[i - time_window_size:i + 1]
                features = model(g_list)
                features = features[cached_labeled_node_mask[i]].detach().cpu().numpy()
                labels = cached_subgraph[i].ndata['label'][cached_labeled_node_mask[i]].detach().cpu().numpy()

                predictions = rf_classifier.predict_proba(features)

                predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.long)

                test_measure.append_measures(predictions_tensor, labels_tensor)

                cl_precision, cl_recall, cl_f1 = test_measure.get_total_measure()
                timestep_precisions.append(cl_precision)
                timestep_recalls.append(cl_recall)
                timestep_f1s.append(cl_f1)

                print("  Test | Time {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                      .format(valid_max_index + i + 1, cl_precision, cl_recall, cl_f1))

            cl_precision, cl_recall, cl_f1 = test_measure.get_total_measure()
            test_measure.update_best_f1(cl_f1, epoch)
            test_measure.reset_info()

            test_res_f1 = cl_f1

            # Save the test precision, recall, and f1 for the current epoch
            test_precisions.append(cl_precision)
            test_recalls.append(cl_recall)
            test_f1s.append(cl_f1)

            print("  Test | Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}"
                  .format(epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1))

            # Plot for each test timestep
            timesteps = range(valid_max_index + 1, test_max_index + 1)
            plt.figure()
            plt.plot(timesteps, timestep_precisions, label='Precision')
            plt.plot(timesteps, timestep_recalls, label='Recall')
            plt.plot(timesteps, timestep_f1s, label='F1 Score')
            plt.xlabel('Timestep')
            plt.ylabel('Score')
            plt.title('Test Precision, Recall, and F1 Score at Each Timestep')
            plt.legend()
            plt.savefig(f'test_metrics_timestep_epoch_{epoch}.png')
            plt.show()

    print("Best test f1 is {}, in Epoch {}"
          .format(test_measure.target_best_f1, test_measure.target_best_f1_epoch))
    if test_measure.target_best_f1_epoch != valid_measure.target_best_f1_epoch:
        print("The Epoch get best Valid measure not get the best Test measure, "
              "please checkout the test result in Epoch {}, which f1 is {}"
              .format(valid_measure.target_best_f1_epoch, test_res_f1))

    # Plot test precision, recall, and f1 over epochs
    epochs = range(len(test_precisions))
    plt.figure()
    plt.plot(epochs, test_precisions, label='Test Precision')
    plt.plot(epochs, test_recalls, label='Test Recall')
    plt.plot(epochs, test_f1s, label='Test F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Test Precision, Recall, and F1 over Epochs')
    plt.legend()
    plt.savefig('test_metrics_over_epochs.png')
    plt.show()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("EvolveGCN")
    argparser.add_argument('--model', type=str, default='EvolveGCN-O',
                           help='We can choose EvolveGCN-O or EvolveGCN-H,'
                                'but the EvolveGCN-H performance on Elliptic dataset is not good.')
    argparser.add_argument('--raw-dir', type=str,
                           default='/home/Elliptic/elliptic_bitcoin_dataset/',
                           help="Dir after unzip downloaded dataset, which contains 3 csv files.")
    argparser.add_argument('--processed-dir', type=str,
                           default='/home/Elliptic/processed/',
                           help="Dir to store processed raw data.")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training.")
    argparser.add_argument('--num-epochs', type=int, default=12)
    argparser.add_argument('--n-hidden', type=int, default=256)
    argparser.add_argument('--n-layers', type=int, default=2)
    argparser.add_argument('--n-hist-steps', type=int, default=5,
                           help="If it is set to 5, it means in the first batch,"
                                "we use historical data of 0-4 to predict the data of time 5.")
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--loss-class-weight', type=str, default='0.35,0.65',
                           help='Weight for loss function. Follow the official code,'
                                'we need to change it to 0.25, 0.75 when use EvolveGCN-H')
    argparser.add_argument('--eval-class-id', type=int, default=1,
                           help="Class type to eval. On Elliptic, type 1(illicit) is the main interest.")
    argparser.add_argument('--patience', type=int, default=100,
                           help="Patience for early stopping.")

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))
