import torch
import config


LR_Reduce_No_Train_Improvement = config.LR_Reduce_No_Train_Improvement
LR_Reduce_No_Val_Improvement = config.LR_Reduce_No_Val_Improvement
EARLY_STOP_LR_TOO_LOW = config.EARLY_STOP_LR_TOO_LOW

# This learning rate scheduler works as follows: After 15 epochs of no improvement on the validation loss, the learning rate gets divided by a specified factor. 
# Training terminates if the learning rate has fallen below 4E-6, then the best model on the validation loss (with highest validation accuracy) will be chosen as the final model.
class LRScheduler():
    def __init__(self, optimizer, model, foldResultsModelPath, setting, initLR, divideLRfactor):
        self.optimizer = optimizer
        self.model = model
        self.foldResultsModelPath = foldResultsModelPath
        self.currentLR = initLR
        self.divideLRfactor = divideLRfactor

        self.noImprovement = 0

        if 'val' in setting:
            self.bestValue = -1
        else:
            self.bestValue = 1E4

    # either way you train without utilizing a validation data set, then instead of the later, everything will be performed on the training data set!
    def stepTrain(self, newTrainLoss, logger, epoch):
        self.epoch = epoch
        # Update learning rate
        if newTrainLoss >= self.bestValue:
            self.noImprovement += 1

            if self.noImprovement >= LR_Reduce_No_Train_Improvement:
                self.model.load_state_dict(torch.load(self.foldResultsModelPath + '/currentBestTrainModel.pt', map_location='cpu'))
                self.update_lr_by_divison(self.divideLRfactor)
                logger.info('### After '+str(LR_Reduce_No_Train_Improvement)+' no train loss reduction => Best model loaded and LR reduced to '+str(self.currentLR)+' !')
                if self.currentLR < EARLY_STOP_LR_TOO_LOW:
                    return True
                self.noImprovement = 0
                # save intermediate model
                # self.checkpoint = {'model': self.model.state_dict(),
                #                    'optimizer':self.optimizer.state_dict(),
                #                    'epoch': self.epoch-LR_Reduce_No_Train_Improvement}
                # torch.save(self.checkpoint, self.foldResultsModelPath + '/%sepochBestTrainModel.pt' %(str(self.epoch-LR_Reduce_No_Train_Improvement)))
        else:
            self.noImprovement = 0
            self.bestValue = newTrainLoss
            torch.save(self.model.state_dict(), self.foldResultsModelPath + '/currentBestTrainModel.pt')

        return False

    # when utilizing a validation data set as recommended/commonly suggested
    def stepTrainVal(self, newValScore, logger, epoch):
        self.epoch = epoch
        # Update learning rate
        if newValScore <= self.bestValue:
            self.noImprovement += 1

            if self.noImprovement >= LR_Reduce_No_Val_Improvement:
                self.model.load_state_dict(torch.load(self.foldResultsModelPath + '/currentBestValModel.pt', map_location='cpu'))
                self.update_lr_by_divison(self.divideLRfactor)
                logger.info('### After ' + str(LR_Reduce_No_Val_Improvement) + ' no val score improvement => Best model loaded and LR reduced to ' + str(self.currentLR) + ' !')
                if self.currentLR < EARLY_STOP_LR_TOO_LOW:
                    return True
                self.noImprovement = 0
                # save intermediate model
                # self.checkpoint = {'model': self.model.state_dict(),
                #                    'optimizer':self.optimizer.state_dict(),
                #                    'epoch': self.epoch-LR_Reduce_No_Val_Improvement}
                # torch.save(self.checkpoint, self.foldResultsModelPath + '/%sepochBestTrainModel.pt' %(str(self.epoch-LR_Reduce_No_Val_Improvement)))
        else:
            self.noImprovement = 0
            self.bestValue = newValScore
            torch.save(self.model.state_dict(), self.foldResultsModelPath + '/currentBestValModel.pt')

        return False

    # divides learning rate of network by 'factor'
    def update_lr_by_divison(self, factor):
        newLR = self.currentLR / factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = newLR
        self.currentLR = newLR

    # loads current model with highest validation accuracy into current model
    def loadBestValIntoModel(self):
        self.model.load_state_dict(torch.load(self.foldResultsModelPath + '/currentBestValModel.pt', map_location='cpu'))
