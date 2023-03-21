import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.optim as optim
import wandb
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from config.parser_config import config_parser
import albumentations as A
from augmentation import  aug_transform
from tqdm import tqdm #te quiero demasio. taqadum
from classification_dataset import ClassificationDataset
from pathlib import Path
from dataset_utils import stratified_split, edge_stratified_split

def main(
    run_name: str,
    root_path: str = ('.'),
    log_path: str = "./data/classification/logs",
    gan_dir: str = "./gan_generated_data/all_fake",
    gan_info: str = "./gan_generated_data/gan_info.csv",
    npz_path: str = None,
    image_path: str =  None,
    label_path: str = None,
    batch_size: int = 32,
    epochs: int = 100,
    num_workers: int = 8,
    img_size = 500,
    lr: float = 1e-4,
    is_resume_training: bool = False,
    given_augment = None):

    if given_augment == 'no augment':
        augment = None
    elif given_augment:
        augment = given_augment
    else:
        augment = aug_transform()
    
    trained_epochs = 0
    #LOADING DATA
    full_dataset = ClassificationDataset(
        one_hot = False,
        augmentation= augment,
        npz_path= npz_path,
        image_path= image_path,
        label_path= label_path,
        size = img_size)
    

    full_dataset._extend(gan_dir=gan_dir, gan_info=gan_info)

    get_cat_from_label = full_dataset._get_cat_from_label
    num_classes = full_dataset._get_num_classes()
    
    edge_classes = ["Gryllteiste","Schnatterente","Buchfink","unbestimmte Larusmöwe",
                        "Schmarotzer/Spatel/Falkenraubmöwe","Brandgans","Wasserlinie mit Großalgen",
                        "Feldlerche","Schmarotzerraubmöwe","Grosser Brachvogel","unbestimmte Raubmöwe",
                        "Turmfalke","Trauerseeschwalbe","unbestimmter Schwan",
                        "Sperber","Kiebitzregenpfeifer",
                        "Skua","Graugans","unbestimmte Krähe"]
    
    edge_labels = [full_dataset._get_label_from_cat(cat) for cat in edge_classes]
    
    train_data, train_set_labels, validation_data, test_set_labels = stratified_split(dataset = full_dataset, 
                                                                                            labels = full_dataset._labels,
                                                                                            fraction = 0.8,
                                                                                            random_state=0)        
    

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
    
    #CONFIG LOGS FOR SAVING LATER
    SAVE_PATH = Path(log_path, "classifier_{}.pth".format(run_name))
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    
    #MODEL
    MODEL = timm.create_model("efficientnet_b0", pretrained=True, num_classes= num_classes, in_chans= 3)
    MODEL.to(device)
    OPTIM = optim.Adam(MODEL.parameters(), lr=lr)
    criterion_classification = nn.CrossEntropyLoss()
    LOSSES = []
    
    # #CONFIG IF RESUMING
    # if wandb.run.resumed:
    #     MODEL = keras.models.load_model(wandb.restore("model-best.h5").name)
    # if is_resume_training:
    #     checkpoint = torch.load(SAVE_PATH)
    #     MODEL.load_state_dict(checkpoint['model_state_dict'])
    #     OPTIM.load_state_dict(checkpoint['optimizer_state_dict'])
    #     trained_epochs = checkpoint['trained_epochs']
    #     LOSSES = checkpoint['loss']
        
    ##CONFIG WANDB
    wandb.config = {
    "learning_rate": lr,
    "batch_size": batch_size,
    "run_name": run_name
    }
    
    is_last_epoch_flag = False
    
    for e in range(trained_epochs, epochs + trained_epochs):
        for img_train, label_train, cat_train in tqdm(train_dataloader, total=len(train_dataloader)):
            if e == trained_epochs:
                log_imgs = utils.make_grid(img_train)
                log_images = wandb.Image(log_imgs, caption="{}".format(run_name))
                wandb.log({"examples": log_images})
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
            
        if epochs + trained_epochs -1 == e:
            is_last_epoch_flag = True
        
        # end of epoch
        # validate the model
        valid_classifier(
            model = MODEL,
            num_classes=num_classes,
            loader_test=validation_dataloader,
            device=device,
            is_last_epoch_flag= is_last_epoch_flag,
            epoch = e,
            get_cat_from_label = get_cat_from_label)

    # torch.save(MODEL.state_dict(), SAVE_PATH)
    # torch.save({
    #         'trained_epochs': trained_epochs + epochs,
    #         'model_state_dict': MODEL.state_dict(),
    #         'optimizer_state_dict': OPTIM.state_dict(),
    #         'loss': LOSS,
    #         'name': run_name
    #         }, SAVE_PATH)
        edge_valid_classifier(
            model = MODEL,
            num_classes=num_classes,
            loader_test=validation_dataloader,
            device=device,
            is_last_epoch_flag= is_last_epoch_flag,
            epoch = e,
            get_cat_from_label = get_cat_from_label,
            edge_labels = edge_labels)
    

def edge_valid_classifier(model, num_classes, loader_test, device, is_last_epoch_flag, epoch, get_cat_from_label, edge_labels):
    edge_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    edge_f1score = MulticlassF1Score(num_classes=num_classes).to(device)
    edge_precision = MulticlassPrecision(num_classes=num_classes).to(device)
    edge_recall = MulticlassRecall(num_classes=num_classes).to(device)
    

    for img, label, cat in loader_test:
        mask = sum(label==i for i in edge_labels).bool()
        cat  = cat[mask]
        img = img[mask].to(device)
        label = label[mask].to(device)
        # predict
        predicted = model(img.float())
        predicted = predicted.to(device)
        edge_accuracy.update(preds=predicted, target=label)
        edge_f1score.update(preds=predicted, target=label)
        edge_precision.update(preds=predicted, target=label)
        edge_recall.update(preds=predicted, target=label)

        for i in range(img.shape[0]):
        #load
            top_3_label = torch.topk(predicted[i].flatten(), 3).indices
            top_3_cat = [get_cat_from_label(label) for label in top_3_label]
            edge_img_to_log = wandb.Image(img[i,...], 
                                caption="P_label:[1]-{}\n[2]-{}\n[3]-{} \n  T_label: {} \n  \n  P_cat: [1]-{}\n[2]-{}\n[3]-{} \n  T_cat: {}".format(top_3_label[0],
                                                                                                        top_3_label[1],
                                                                                                        top_3_label[2],
                                                                                                label[i],
                                                                                                top_3_cat[0],
                                                                                                top_3_cat[1],
                                                                                                top_3_cat[2],
                                                                                                cat[i]))
            wandb.log({"EDGE CASE": edge_img_to_log})
    edge_avg_acc = edge_accuracy.compute()
    edge_avg_f1 = edge_f1score.compute()
    edge_avg_precision = edge_precision.compute()
    edge_avg_recall = edge_recall.compute()
    wandb.log({"edge_accuracy": edge_avg_acc, 'epoch': epoch})
    wandb.log({"edge_f1score": edge_avg_f1, 'epoch': epoch})
    wandb.log({"edge_precision": edge_avg_precision, 'epoch': epoch})
    wandb.log({"edge_recall": edge_avg_recall, 'epoch': epoch})


    
    
def valid_classifier(model, num_classes, loader_test, device, is_last_epoch_flag, epoch, get_cat_from_label):
    accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    f1score = MulticlassF1Score(num_classes=num_classes).to(device)
    precision = MulticlassPrecision(num_classes=num_classes).to(device)
    recall = MulticlassRecall(num_classes=num_classes).to(device)

    accuracy_top3 = MulticlassAccuracy(num_classes=num_classes, top_k=3).to(device)
    f1score_top3 = MulticlassF1Score(num_classes=num_classes, top_k=3).to(device)
    precision_top3 = MulticlassPrecision(num_classes=num_classes, top_k=3).to(device)
    recall_top3 = MulticlassRecall(num_classes=num_classes, top_k=3).to(device)

    # testing loop
    for img, label, cat in loader_test:
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
        
        
        
            
            
        
        if is_last_epoch_flag:
            imgs_to_log = []
            for i in range(img.shape[0]):
                top_3_label = torch.topk(predicted[i].flatten(), 3).indices
                top_3_cat = [get_cat_from_label(label) for label in top_3_label]
                img_to_log = wandb.Image(img[i,...], 
                                  caption="P_label:[1]-{}\n[2]-{}\n[3]-{} \n  T_label: {} \n  \n  P_cat: [1]-{}\n[2]-{}\n[3]-{} \n  T_cat: {}".format(top_3_label[0],
                                                                                                           top_3_label[1],
                                                                                                           top_3_label[2],
                                                                                                    label[i],
                                                                                                    top_3_cat[0],
                                                                                                    top_3_cat[1],
                                                                                                    top_3_cat[2],
                                                                                                    cat[i]))

                imgs_to_log.append(img_to_log)
                
            wandb.log({"Predictions in last epoch": imgs_to_log})
        

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
    wandb.log({"accuracy": avg_acc, 'epoch': epoch})
    wandb.log({"f1score": avg_f1, 'epoch': epoch})
    wandb.log({"precision": avg_precision, 'epoch': epoch})
    wandb.log({"recall": avg_recall, 'epoch': epoch})
    wandb.log({"accuracy top 3": avg_acc_top3, 'epoch': epoch})
    wandb.log({"f1score top 3": avg_f1_top3, 'epoch': epoch})
    wandb.log({"precision top 3": avg_precision_top3, 'epoch': epoch})
    wandb.log({"recalltop 3": avg_recall_top3, 'epoch': epoch})
    
if __name__ == "__main__":
    #CONFIG PARSER
    parser = config_parser()
    args = parser.parse_args()
    
    wandb.init(project="classifier-efficientnet")
    if args.run_name:
        wandb.run.name = args.run_name
    
    main(epochs = args.epochs, 
         run_name = args.run_name,
         npz_path = args.npz_path,
         image_path = args.image_path,
         label_path = args.label_path,
         img_size = args.size,
         augment = args.augment)

def single_run(args, run_name, given_augment, project_name):
    wandb.init(project=project_name)
    
    wandb.run.name = run_name + ' | lr: {} |  epochs: {} | size: {}'.format(args.lr, args.epochs, args.size)
    
    wandb.config = {'epochs' : args.epochs, 
    'run_name' : args.run_name,
    'npz_path' :args.npz_path,
    'image_path' : args.image_path,
    'label_path' : args.label_path,
    'img_size' : args.size,
    'lr' : args.lr
    }
    
    main(epochs = args.epochs, 
         run_name = args.run_name,
         npz_path = args.npz_path,
         image_path = args.image_path,
         label_path = args.label_path,
         img_size = args.size,
         lr = args.lr,
         given_augment = given_augment)
    
    
    
    wandb.finish()