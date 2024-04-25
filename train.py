import torch
import tqdm
import torch.nn as nn
from model import Model
import torch.optim as optim
from dataset import val_loader, train_loader, dataset
from utils import calc_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])

def val(model):
    val_loss = 0
    bleu_score = 0
    for idx, (imgs, captions) in tqdm(
        enumerate(val_loader), total=len(val_loader), leave=False
    ):
        with torch.no_grad():
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            
            opt = outputs.permute(1,0,2).argmax(2)
            inp = captions.permute(1,0)
            
            bleu_score += calc_bleu(inp,opt)
            val_loss += criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
    print("Validation loss: ", val_loss / len(val_loader))
    print("BLEU Score: ", bleu_score / len(val_loader))
  

def train():
    torch.backends.cudnn.benchmark = True
    load_model = False
    save_model = False
    train_CNN = False

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 1e-6
    num_epochs=100

    model = Model(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=3e-2)

    for name, param in model.Encoder.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN


    model.train()
    prev_loss = 100000

    for epoch in range(num_epochs):
        print("Epoch : ", epoch)

        total_loss = 0
        total_bleu_score = 0

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            
            opt = outputs.permute(1,0,2).argmax(2)
            inp = captions.permute(1,0)
            
            total_bleu_score += calc_bleu(inp,opt)
            total_loss += loss
            total_bleu_score += calc_bleu(inp,opt)
            

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
        
        
        total_loss = total_loss / len(train_loader)
        total_bleu_score = total_bleu_score / len(train_loader)
        if total_loss<prev_loss:
            prev_loss = total_loss
            torch.save(model.state_dict(), '/kaggle/working/caption_model.pth')
                
        print("Loss : ", total_loss)
        val(model)         