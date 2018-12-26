import torch
import time
from pathlib import Path

class Trainer:
    
    def __init__(self, model, criterion, optimizer, output_dir):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.current_epoch = 1

    def fit(self, dataset_iter, epochs, patience=0):
        start_time = time.time()
        best_epoch = 0
        best_loss = float("inf")

        for epoch in range(self.current_epoch, epochs + 1):
            self.current_epoch = epoch
            print('Epoch {} of {}'.format(self.current_epoch, epochs))

            # Train a single epoch
            loss = self.fit_epoch(dataset_iter)

            # Save a better model
            if loss < best_loss:
                best_epoch = epoch
                best_loss = loss
                epoch_dir = 'epoch_{}'.format(epoch)
                self.save(epoch_dir)

            # Early stopping
            if patience > 0:
                elapsed_epochs = self.current_epoch - best_epoch
                if elapsed_epochs == patience:
                    print('Stop training! No improvement after {} epochs'.format(passed_epochs))
                    self.restore_epoch(best_epoch)
                    break

        elapsed = time.time() - start_time
        hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
        print('Training ended after {}'.format(hms))
    

    def fit_epoch(self, dataset_iter):
        self.model.train()  # informs pytorch that we are going to train this model
        all_loss = []
        avg = lambda x: sum(x) / len(x)
        for i, batch in enumerate(dataset_iter, start=1):
            # the 5 basic steps
            self.model.zero_grad()
            pred = self.model(batch.words)
            loss = self.criterion(pred, batch.label)
            loss.backward()
            self.optimizer.step()
            
            # save loss per batch and print it
            all_loss.append(loss.item())
            print('Loss ({}/{}): {:.4f}'.format(i, len(dataset_iter), avg(all_loss)), end='\r')
        print('')
        return avg(all_loss)


    def eval(self, dataset_iter, return_preds=False):
        self.model.eval()  # informs pytorch that we are going to evaluate this model
        all_loss = []
        all_preds = []
        with torch.no_grad():  # don't keep track of gradients
            for i, batch in enumerate(dataset_iter, start=1):
                pred = self.model(batch.words)
                loss = self.criterion(pred, batch.label)
                all_loss.append(loss.item())
                all_preds.extend(pred.numpy())
        avg_loss = sum(all_loss) / len(all_loss)
        print('Loss: {:.4f}'.format(avg_loss))
        if return_preds:
            return avg_loss, all_preds
        return avg_loss


    def save(self, dirname):
        output_path = Path(self.output_dir, dirname)
        output_path.mkdir(exist_ok=True)
        print('Saving training state to {}'.format(output_path))

        model_path = Path(output_path, 'model.torch')
        model_state_dict = self.model.state_dict()
        optimizer_path = Path(output_path, 'optimizer.torch')
        optimizer_state_dict = self.optimizer.state_dict()

        torch.save(model_state_dict, str(model_path))
        torch.save(optimizer_state_dict, str(optimizer_path))

    def load(self, dirname):
        output_path = Path(self.output_dir, dirname)
        print('Loading training state from {}'.format(output_path))
        model_path = Path(output_path, 'model.torch')
        optimizer_path = Path(output_path, 'optimizer.torch')
        model_state_dict = torch.load(str(model_path), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        optimizer_state_dict = torch.load(str(optimizer_path), map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(optimizer_state_dict)


    def restore_epoch(self, epoch):
        epoch_dir = 'epoch_{}'.format(epoch)
        self.load(str(Path(self.output_dir, epoch_dir)))
