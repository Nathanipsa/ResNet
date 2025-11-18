import network as fn
import torch
import numpy as np
import os
import sys
import zipfile
import cv2
from sklearn.model_selection import train_test_split
from Logger import PrintLog
from Config import Config


def start_print_logging(filepath="prints.log", mode="w") -> PrintLog:
    logger = PrintLog(filepath, mode)
    sys.stdout = logger
    return logger


def main() -> int:
    logger = start_print_logging("output.txt")


    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")

    # -----------------------
    # Hyperparams / config -> adding the hyperparams in a Config class
    # -----------------------
    # num_training = 150000
    # learning_rate = 0.1
    # accumulation_steps = 2
    # batch_size = 64
    # model_save_path = './model/resnet_attention'
    # brestore = False
    # restore_iter = 137000

    # patience = 20000
    # min_delta = 0.1
    # best_accuracy = 0.0
    # patience_counter = 0
    # replace by Config()
    config = Config()

    if not config.brestore:
        config.restore_iter = 0

    # Don't change it
    # path = r'/home/kms0712w900/Desktop/project2/'
    path = sys.argv[1]

    print('load zip...')
    z_train = zipfile.ZipFile(path + 'train.zip', 'r')
    z_train_list = z_train.namelist()
    train_cls = fn.read_gt(path + 'train_gt.txt', len(z_train_list))

    print("Splitting the dataset into train : 190 000; test : 10 000")

    train_list_final, val_list, train_cls_final, val_cls = train_test_split(
        z_train_list,
        train_cls,
        test_size=0.05,  # 5% de 200k = 10k validation images
        random_state=42, # Reproductable split
        stratify=train_cls
    )

    z_test = zipfile.ZipFile(path + 'test.zip', 'r')
    z_test_list = z_test.namelist()
    test_cls = fn.read_gt(path + 'test_gt.txt', len(z_test_list))

    print(f"Training samples: {len(z_train_list)}")
    print(f"Test samples: {len(z_test_list)}")
    print(f"MEAN: {fn.MEAN}")
    print(f"STD: {fn.STD}")

    model = fn.ResNet().to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_training, eta_min=1e-6)

    if config.brestore:
        print('Model restored from file')
        model.load_state_dict(torch.load(config.model_save_path + '/model_%d.pt' % config.restore_iter))
        print(f'Restoring scheduler to iteration {config.restore_iter}...')
        for _ in range(config.restore_iter):
            scheduler.step()
        print('Scheduler restored successfully')

    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    if not os.path.isdir(config.model_save_path):
        os.makedirs(config.model_save_path)

    optimizer.zero_grad()

    for it in range(config.restore_iter, config.num_training + 1):
        batch_img, batch_cls = fn.Mini_batch_training_zip(
            z_train,
            train_list_final,
            train_cls_final,
            config.batch_size,
            augmentation=True
        )
        batch_img = np.transpose(batch_img, (0, 3, 1, 2))

        model.train()
        pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(DEVICE))
        cls_tensor = torch.tensor(batch_cls, dtype=torch.long).to(DEVICE)

        raw_loss = loss(pred, cls_tensor)
        train_loss = raw_loss / config.accumulation_steps
        train_loss.backward()

        if (it + 1) % config.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if it % 100 == 0:
            # print("it: %d   train loss: %.4f" % (it, train_loss.item()))
            print(f"it: {it}   raw loss: {raw_loss.item():.4f}   scaled loss: {train_loss.item():.4f}")


        if it % 500 == 0 and it != 0:
            print('Saving checkpoint...')
            torch.save(model.state_dict(), config.model_save_path + '/model_%d.pt' % it)

        if it % 1000 == 0 and it != 0:
            print('Test')
            model.eval()
            t1_count = 0
            t5_count = 0

            for itest in range(len(val_list)):
                img_temp = z_train.read(val_list[itest])
                img = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
                img = img.astype(np.float32)

                # Applying crop here too to test the same way we trained the model

                padding = 10
                img_padded = np.pad(img.astype(np.float32), ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
                h, w = img_padded.shape[:2]
                top = (h - 128) // 2
                left = (w - 128) // 2
                img_cropped = img_padded[top:top + 128, left:left + 128, :]

                test_img = img_cropped / 255.0
                test_img = (test_img - fn.MEAN) / fn.STD
                test_img = np.reshape(test_img, [1, 128, 128, 3])
                test_img = np.transpose(test_img, (0, 3, 1, 2))

                with torch.no_grad():
                    pred = model(torch.from_numpy(test_img.astype(np.float32)).to(DEVICE))

                pred = pred.cpu().numpy()
                pred = np.reshape(pred, 200)

                gt = val_cls[itest]

                for ik in range(5):
                    max_index = np.argmax(pred)

                    if int(gt) == int(max_index):
                        if ik == 0:
                            t1_count += 1
                        t5_count += 1
                    pred[max_index] = -9999

            t1_accuracy = t1_count / float(len(val_list)) * 100
            t5_accuracy = t5_count / float(len(val_list)) * 100

            print("top-1 : %.4f%%     top-5: %.4f%%\n" % (t1_accuracy, t5_accuracy))

            if t1_accuracy > config.best_accuracy + config.min_delta:
                config.best_accuracy = t1_accuracy
                config.patience_counter = 0
                print('Best model saved (acc: %.2f%%)' % config.best_accuracy)
                torch.save(model.state_dict(), config.model_save_path + '/best_model.pt')
            else:
                config.patience_counter += 1000
                print('No improvement (%d/%d)' % (config.patience_counter, config.patience))

            f = open(config.model_save_path + '/accuracy.txt', 'a+')
            f.write("iter: %d   top-1 : %.4f     top-5: %.4f\n" % (it, t1_accuracy, t5_accuracy))
            f.close()

            if config.patience_counter >= config.patience:
                print('Early stop / best: %.2f%%' % config.best_accuracy)
                break

    torch.save(model.state_dict(), config.model_save_path + '/final_model.pt')
    print('Training done / best: %.2f%%' % config.best_accuracy)

    logger.close()

    return 0


if __name__ == '__main__':
    main()
