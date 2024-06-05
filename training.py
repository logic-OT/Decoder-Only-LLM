from torchmetrics import Accuracy
from tqdm import tqdm
import torch
from model import Model
from config import config
from data_preparation import prepare_data,tokenizer


def train():  
    device = config['device']
    lr = config['learning_rate']
    num_batches = config['num_batches']

    dataset =  prepare_data()
    loader = torch.utils.data.DataLoader(dataset,batch_size=num_batches,num_workers=0,shuffle=False)

    model,vocab_size = Model()
    metric =  Accuracy(num_classes=vocab_size,task='multiclass').to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.01)

    print('Training Started')
    NUM_EPOCHS = 1

    for epoch in range(NUM_EPOCHS):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = len(loader)

        # Initialize tqdm progress bar
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}", leave=True) as pbar:
            for i, (x_batch, y_batch) in enumerate(loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(x_batch)
                # Flatten the outputs and y_batch tensors one dimension lower
                outputs = outputs.view(-1, outputs.shape[-1])
                y_batch = y_batch.view(-1)
                
                # Loss calculation
                loss = criterion(outputs, y_batch).to(device)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Metrics
                argmax_pred = outputs.argmax(axis=1)
                metric.update(argmax_pred, y_batch)

                # Print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # update every 10 mini-batches
                    accuracy = metric.compute().item()
                    epoch_accuracy += accuracy
                    pbar.set_postfix({'Loss': running_loss / (i + 1), 'Accuracy': accuracy})
                
                # Update the progress bar
                pbar.update(1)

                # Save model weights periodically
                if i % 10 == 9:
                    print('Saved weights after 10 steps as model_weights.pth')
                    torch.save(model.state_dict(), 'model_weights.pth')

        # Compute and print average loss and accuracy for the epoch
        avg_loss = running_loss / num_batches
        avg_accuracy = epoch_accuracy / (num_batches // 10)  # since we're summing accuracy every 10 batches
        print(f'Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

    print('Training Completed')


train()