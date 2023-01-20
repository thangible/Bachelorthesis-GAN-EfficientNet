import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms 
from ParserConfig import config_parser
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

# def valid_classifier(classifier, cat_count, loader_test, tb_writer, device, current_epoch):
#     accuracy = MulticlassAccuracy(
#         num_classes=cat_count, top_k=1).to(device)
#     f1score = MulticlassF1Score(
#         num_classes=cat_count).to(device)
#     precision = MulticlassPrecision(
#         num_classes=cat_count).to(device)
#     recall = MulticlassRecall(num_classes=cat_count).to(device)

#     accuracy_top3 = MulticlassAccuracy(
#         num_classes=cat_count, top_k=3).to(device)
#     f1score_top3 = MulticlassF1Score(
#         num_classes=cat_count, top_k=3).to(device)
#     precision_top3 = MulticlassPrecision(
#         num_classes=cat_count, top_k=3).to(device)
#     recall_top3 = MulticlassRecall(
#         num_classes=cat_count, top_k=3).to(device)

#     # testing loop
#     for img, label in loader_test:
#         img = img.to(device)
#         # one-hot encode class labels
#         label = torch.argmax(label, dim=1)
#         label = label.to(device)

#         # predict
#         predicted = classifier(img)

#         # update metrics
#         predicted = predicted.to(device)
#         label = label.to(device)
#         accuracy.update(preds=predicted, target=label)
#         f1score.update(preds=predicted, target=label)
#         precision.update(preds=predicted, target=label)
#         recall.update(preds=predicted, target=label)
#         # update top3 metrics
#         accuracy_top3.update(preds=predicted, target=label)
#         f1score_top3.update(preds=predicted, target=label)
#         precision_top3.update(preds=predicted, target=label)
#         recall_top3.update(preds=predicted, target=label)

#     # calculate average metrics over all batches (single results in the container)
#     avg_acc = accuracy.compute()
#     avg_f1 = f1score.compute()
#     avg_precision = precision.compute()
#     avg_recall = recall.compute()

#     # calculate top3 metrics
#     avg_acc_top3 = accuracy_top3.compute()
#     avg_f1_top3 = f1score_top3.compute()
#     avg_precision_top3 = precision_top3.compute()
#     avg_recall_top3 = recall_top3.compute()

#     # write to tensorboard
#     # accuracy
#     tb_writer.add_scalar(
#         'validation/accuracy/top1',
#         avg_acc,
#         global_step=current_epoch
#     )
#     tb_writer.add_scalar(
#         'validation/accuracy/top3',
#         avg_acc_top3,
#         global_step=current_epoch
#     )

#     # f1
#     tb_writer.add_scalar(
#         'validation/f1/top1',
#         avg_f1,
#         global_step=current_epoch
#     )

#     tb_writer.add_scalar(
#         'validation/f1/top3',
#         avg_f1_top3,
#         global_step=current_epoch)

#     # precision
#     tb_writer.add_scalar(
#         'validation/precision/top1',
#         avg_precision,
#         global_step=current_epoch
#     )
#     # precision top 3
#     tb_writer.add_scalar(
#         'validation/precision/top3',
#         avg_precision_top3,
#         global_step=current_epoch
#     )

#     # recall
#     tb_writer.add_scalar(
#         'validation/recall/top1',
#         avg_recall,
#         global_step=current_epoch
#     )
#     # recall top 3
#     tb_writer.add_scalar(
#         'validation/recall/top3',
#         avg_recall_top3,
#         global_step=current_epoch)

#     # flush tensorboard
#     tb_writer.flush()


def main(
    root_path: str = ('.'),
    log_path: str = "./data/classification/logs",
    image_dir: str = "./data/imgs",
    label_pat: str = "./data/labels", 
    batch_size: int = 32,
    epochs: int = 100,
    num_workers: int = 8,
    img_size = 256,
    lr: float = 1e-4,
    cat_count: int = 156):
    
    #CONFIG PARSER
    parser = config_parser()
    
    

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size])
    ])
    # target transform
    target_transform = transforms.Compose([
        lambda x: torch.as_tensor(x),
        lambda x: F.one_hot(x.to(torch.int64), cat_count)
    ])



    
