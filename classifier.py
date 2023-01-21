import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.optim as optim
import wandb
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchvision import transforms 
from torch.utils.data import DataLoader
from parser_config import config_parser
from augmentation import  aug_transfrom
from tqdm import tqdm #te quiero demasio. taqadum
from segmentation_dataset import ClassificationDataset
from pathlib import Path


wandb.init(project="classifier-efficientnet")


def main(
    root_path: str = ('.'),
    log_path: str = "./data/classification/logs",
    npz_path: str = None,
    image_path: str = 'data/ISEAD/segmentation_masks/JPEGImages',
    label_path: str = 'data/ISEAD/segmentation_masks/JPEGImages/manual_annotated_data.csv', 
    batch_size: int = 32,
    epochs: int = 100,
    num_workers: int = 8,
    img_size = 256,
    lr: float = 1e-4):
    
    #CONFIG WANDB
    wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size
    }
    
    #CONFIG PARSER
    parser = config_parser()
    args = parser.parse_args()

    #LOADING DATA
    
    
    
    full_dataset = ClassificationDataset(
        unwanted_classes= args.unwanted_classes,
        unwanted_pics= args.unwanted_pics,
        npz_path= args.npz_path,
        label_path= args.label_path,
        image_path= args.image_path,
        one_hot = True,
        augmentation= aug_transfrom
    )
    num_classes = full_dataset._get_num_classes()
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_data, validation_data = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers)
    
    validation_dataloader = DataLoader(validation_data, 
                                       batch_size=batch_size, 
                                       shuffle=True,
                                       num_workers=num_workers)
    
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL = timm.create_model("efficientnet_b0", pretrained=True,
                          num_classes=num_classes, in_chans=3)
    MODEL.to(device)
    OPTIM = optim.Adam(MODEL.parameters(), lr=lr)
    criterion_classification = nn.CrossEntropyLoss()
    global_step = 0
    LOSSES = []
    
    for e in range(args.epochs):
        for img_train, label_train in tqdm(train_dataloader, total=len(train_dataloader)):
            img_train = img_train.to(device)
            label_train = label_train.to(device)
            # predict
            predicted = MODEL(img_train.float())
            # loss
            LOSS = criterion_classification(predicted, label_train)
            # backward
            LOSS.backward()
            # optimize
            OPTIM.step()
            # logging
            LOSSES.append(LOSS.data.item())
            
        # end of epoch
        # validate the model
        valid_classifier(
            model = MODEL,
            num_classes=num_classes,
            loader_test=validation_dataloader,
            device=device,
            current_epoch=e
        )

    save_path = Path(log_path, "classifier.pth")
    Path(log_path).mkdir(parents=True, exist_ok=True)
    torch.save(MODEL.state_dict(), save_path)
    
    
def valid_classifier(model, num_classes, loader_test, device, current_epoch):
    accuracy = MulticlassAccuracy(
        num_classes=num_classes, top_k=1).to(device)
    f1score = MulticlassF1Score(
        num_classes=num_classes).to(device)
    precision = MulticlassPrecision(
        num_classes=num_classes).to(device)
    recall = MulticlassRecall(num_classes=num_classes).to(device)

    accuracy_top3 = MulticlassAccuracy(
        num_classes=num_classes, top_k=3).to(device)
    f1score_top3 = MulticlassF1Score(
        num_classes=num_classes, top_k=3).to(device)
    precision_top3 = MulticlassPrecision(
        num_classes=num_classes, top_k=3).to(device)
    recall_top3 = MulticlassRecall(
        num_classes=num_classes, top_k=3).to(device)

    # testing loop
    for img, label in loader_test:
        #load
        img = img.to(device)
        label = label.to(device)
        # predict
        predicted = model(img.float())

        # update metrics
        predicted = predicted.to(device)
        accuracy.update(preds=predicted, target=label)
        f1score.update(preds=predicted, target=label)
        precision.update(preds=predicted, target=label)
        recall.update(preds=predicted, target=label)
        # update top3 metrics
        accuracy_top3.update(preds=predicted, target=label)
        f1score_top3.update(preds=predicted, target=label)
        precision_top3.update(preds=predicted, target=label)
        recall_top3.update(preds=predicted, target=label)

    # calculate average metrics over all batches (single results in the container)
    avg_acc = accuracy.compute()
    avg_f1 = f1score.compute()
    avg_precision = precision.compute()
    avg_recall = recall.compute()

    # calculate top3 metrics
    avg_acc_top3 = accuracy_top3.compute()
    avg_f1_top3 = f1score_top3.compute()
    avg_precision_top3 = precision_top3.compute()
    avg_recall_top3 = recall_top3.compute()
    
    wandb.log({"accuracy": avg_acc})
    wandb.log({"f1score": avg_f1})
    wandb.log({"precision": avg_precision})
    wandb.log({"recall": avg_recall})
    wandb.log({"accuracy top 3": avg_acc_top3})
    wandb.log({"f1score top 3": avg_f1_top3})
    wandb.log({"precision top 3": avg_precision_top3})
    wandb.log({"recalltop 3": avg_recall_top3})
    
if __name__ == "__main__":
    main()



    
