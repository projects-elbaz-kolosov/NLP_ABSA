from utility import *
import torch
import torch.nn as nn
from model import BertClassifier
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

datadir = "../data/"
#trainfile = datadir + "traindata.csv"
devfile = datadir + "devdata.csv"

def initialize_model(train_dataloader, model, epochs=3, lr1=1e-5, lr2=6e-4):
    # initializes the optimizer and the scheduler
    param_groups = [{'params': model.bert.parameters(), 'lr': lr1},
                    {'params': model.classifier.parameters(), 'lr': lr2}]

    optimizer = AdamW(param_groups,
                      eps=1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return optimizer, scheduler


class Classifier:
    """The Classifier"""

    def __init__(self):
        class_weights = [0.2, 0.1, 0.7]
        weights = torch.tensor(class_weights).to(device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.model = BertClassifier(freeze_bert=False)
        self.model.to(device)
        self.best_acc = 0

    def train(self, train_path, val_path=devfile, epochs=3, evaluation=True):
        train_dataloader = final_data_preprocessing(train_path)

        optimizer, scheduler = initialize_model(train_dataloader, self.model, epochs=epochs)

        print("Start training...\n")
        for epoch_i in range(epochs):

            print(
                f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 60)

            t0_epoch, t0_batch = time.time(), time.time()

            total_loss, batch_loss, batch_counts = 0, 0, 0

            self.model.train()

            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                b_feat, b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                self.model.zero_grad()

                logits = self.model(b_feat, b_input_ids, b_attn_mask)

                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % len(train_dataloader) == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            if evaluation == True:
                val_dataloader = final_data_preprocessing(val_path, test=True)
                val_loss, val_accuracy = self.evaluate(val_dataloader)
                time_elapsed = time.time() - t0_epoch
                print(
                    f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-" * 60)
                print("\n")
                if val_accuracy > self.best_acc:
                    print('Saving model')
                    state = {
                        'net': self.model.state_dict()
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, './checkpoint/ckpt.pth')

                    self.best_acc = val_accuracy
                    print('Best dev accuracy so far : {} \n'.format(round(self.best_acc, 2)))

        PATH = './checkpoint/ckpt.pth'
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['net'])
        print("Training complete!")

    def evaluate(self, val_dataloader):
        self.model.eval()

        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_feat, b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_feat, b_input_ids, b_attn_mask)
            # Compute loss
            loss = self.loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()
            #            print(preds)
            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    def predict(self, test_path):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

        test_dataloader = final_data_preprocessing(test_path, test=True)

        self.model.eval()

        all_preds = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_feat, b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:3]

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_feat, b_input_ids, b_attn_mask)
            preds = torch.argmax(logits, dim=1).flatten()
            all_preds.append(preds)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_preds, dim=0)
        all_logits = all_logits.cpu().numpy()

        return list(map(assign_reverse_label, all_logits))
