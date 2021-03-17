import  matplotlib.pyplot as plt
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def plot_roc_curve(fpr, tpr, label=None):
	plt.plot(fpr, tpr, linewidth=2, label=label)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
	plt.ylabel('True Positive Rate (Recall)', fontsize=16)
	plt.grid(True)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
	plt.legend(loc="center right", fontsize=16)
	plt.xlabel("Threshold", fontsize=16)
	plt.grid(True)
	plt.axis([-50000, 50000, 0, 1])


def plot_precision_vs_recall(precisions, recalls):
	plt.plot(recalls, precisions, "b-", linewidth=2)
	plt.xlabel("Recall", fontsize=16)
	plt.ylabel("Precision", fontsize=16)
	plt.axis([0, 1, 0, 1])
	plt.grid(True)
