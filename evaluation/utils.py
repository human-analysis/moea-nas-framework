from ofa.utils import AverageMeter, accuracy


def get_metric_dict():
    return {
        'top1': AverageMeter(),
        'top5': AverageMeter(),
    }


def update_metric(metric_dict, output, labels):
    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
    metric_dict['top1'].update(acc1[0].item(), output.size(0))
    metric_dict['top5'].update(acc5[0].item(), output.size(0))


def get_metric_vals(metric_dict, return_dict=False):
    if return_dict:
        return {
            key: metric_dict[key].avg for key in metric_dict
        }
    else:
        return [metric_dict[key].avg for key in metric_dict]