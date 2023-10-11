import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation

def countParam(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def getDiceScores(prediction, segBatch):
    """
    Compute mean dice scores of predicted foreground labels.
    NOTE: Dice scores of missing gt labels will be excluded and are thus represented by -1 value entries in returned dice score matrix!
    NOTE: Method changes prediction to 0/1 values in the binary case!
    :param prediction: BxCxHxW (if 2D) or BxCxHxWxD (if 3D) FloatTensor (care: prediction has not undergone any final activation!) (note: C=1 for binary segmentation task)
    :param segBatch: BxCxHxW (if 2D) or BxCxHxWxD (if 3D) FloatTensor (Onehot-Encoding) or Bx1xHxW (if 2D) or Bx1xHxWxD (if 3D) LongTensor
    :return: Numpy array containing BxC-1 (background excluded) dice scores
    """
    batchSize, amountClasses = prediction.size()[0], prediction.size()[1]

    if amountClasses == 1: # binary segmentation task, simulate sigmoid to get label results
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = 0
        prediction = prediction.squeeze(1)
        segBatch = segBatch.squeeze(1)
        amountClasses += 1
    else: # multi-label segmentation task
        prediction = prediction.argmax(1) # LongTensor without C-channel
        if segBatch.dtype == torch.float32:  # segBatch is onehot-encoded
            segBatch = segBatch.argmax(1)
        else:
            segBatch = segBatch.squeeze(1)

    prediction = prediction.view(batchSize, -1)
    segBatch = segBatch.view(batchSize, -1)

    labelDiceScores = np.zeros((batchSize, amountClasses-1), dtype=np.float32) - 1 # ignore background class
    for b in range(batchSize):
        currPred = prediction[b,:]
        currGT = segBatch[b,:]

        for c in range(1,amountClasses):
            classPred = (currPred == c).float()
            classGT = (currGT == c).float()

            if classGT.sum() != 0: # only evaluate label prediction when is also present in ground-truth
                labelDiceScores[b, c-1] = ((2. * (classPred * classGT).sum()) / (classGT.sum() + classPred.sum())).item()

    return labelDiceScores

def getMeanDiceScores(diceScores, logger):
    """
    Compute mean label dice scores of numpy dice score array (2d) (and its mean)
    :return: mean label dice scores with '-1' representing totally missing label (meanLabelDiceScores), mean overall dice score (meanOverallDice)
    """
    meanLabelDiceScores = np.ma.masked_where(diceScores == -1, diceScores).mean(0).data
    label_GT_occurrences = (diceScores != -1).sum(0)
    if (label_GT_occurrences == 0).any():
        logger.info('[# WARNING #] Label(s): ' + str(np.argwhere(label_GT_occurrences == 0).flatten() + 1) + ' not present at all in current dataset split!')
        meanLabelDiceScores[label_GT_occurrences == 0] = -1
    meanOverallDice = meanLabelDiceScores[meanLabelDiceScores != -1].mean()

    return meanLabelDiceScores, meanOverallDice

def printResultsForDiseaseModel(evaluatorID, allClassEvaluators, logger, saveResults, resultsPath, diseaseModels):
    logger.info('########## NOW: Detection (average precision) and segmentation accuracies (object-level dice): ##########')
    precisionsTub, avg_precisionTub, avg_dice_scoreTub, std_dice_scoreTub, min_dice_scoreTub, max_dice_scoreTub = allClassEvaluators[evaluatorID][0].score()  # tubuliresults
    precisionsGlom, avg_precisionGlom, avg_dice_scoreGlom, std_dice_scoreGlom, min_dice_scoreGlom, max_dice_scoreGlom = allClassEvaluators[evaluatorID][1].score()  # tubuliresults
    precisionsTuft, avg_precisionTuft, avg_dice_scoreTuft, std_dice_scoreTuft, min_dice_scoreTuft, max_dice_scoreTuft = allClassEvaluators[evaluatorID][2].score()  # tubuliresults
    precisionsVeins, avg_precisionVeins, avg_dice_scoreVeins, std_dice_scoreVeins, min_dice_scoreVeins, max_dice_scoreVeins = allClassEvaluators[evaluatorID][3].score()  # tubuliresults
    precisionsArtery, avg_precisionArtery, avg_dice_scoreArtery, std_dice_scoreArtery, min_dice_scoreArtery, max_dice_scoreArtery = allClassEvaluators[evaluatorID][4].score()  # tubuliresults
    precisionsLumen, avg_precisionLumen, avg_dice_scoreLumen, std_dice_scoreLumen, min_dice_scoreLumen, max_dice_scoreLumen = allClassEvaluators[evaluatorID][5].score()  # tubuliresults
    logger.info('DETECTION RESULTS MEASURED BY AVERAGE PRECISION:')
    logger.info('0.5    0.55    0.6    0.65    0.7    0.75    0.8    0.85    0.9 <- Thresholds')
    logger.info(str(np.round(precisionsTub, 4)) + ', Mean: ' + str(np.round(avg_precisionTub, 4)) + '  <-- Tubuli')
    logger.info(str(np.round(precisionsGlom, 4)) + ', Mean: ' + str(np.round(avg_precisionGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
    logger.info(str(np.round(precisionsTuft, 4)) + ', Mean: ' + str(np.round(avg_precisionTuft, 4)) + '  <-- Tuft')
    logger.info(str(np.round(precisionsVeins, 4)) + ', Mean: ' + str(np.round(avg_precisionVeins, 4)) + '  <-- Veins')
    logger.info(str(np.round(precisionsArtery, 4)) + ', Mean: ' + str(np.round(avg_precisionArtery, 4)) + '  <-- Artery (incl. lumen)')
    logger.info(str(np.round(precisionsLumen, 4)) + ', Mean: ' + str(np.round(avg_precisionLumen, 4)) + '  <-- Artery lumen')
    logger.info('SEGMENTATION RESULTS MEASURED BY OBJECT-LEVEL DICE SCORES:')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreTub, 4)) + ', Std: ' + str(np.round(std_dice_scoreTub, 4)) + ', Min: ' + str(np.round(min_dice_scoreTub, 4)) + ', Max: ' + str(np.round(max_dice_scoreTub, 4)) + '  <-- Tubuli')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreGlom, 4)) + ', Std: ' + str(np.round(std_dice_scoreGlom, 4)) + ', Min: ' + str(np.round(min_dice_scoreGlom, 4)) + ', Max: ' + str(np.round(max_dice_scoreGlom, 4)) + '  <-- Glomeruli (incl. tuft)')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreTuft, 4)) + ', Std: ' + str(np.round(std_dice_scoreTuft, 4)) + ', Min: ' + str(np.round(min_dice_scoreTuft, 4)) + ', Max: ' + str(np.round(max_dice_scoreTuft, 4)) + '  <-- Tuft')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreVeins, 4)) + ', Std: ' + str(np.round(std_dice_scoreVeins, 4)) + ', Min: ' + str(np.round(min_dice_scoreVeins, 4)) + ', Max: ' + str(np.round(max_dice_scoreVeins, 4)) + '  <-- Veins')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreArtery, 4)) + ', Std: ' + str(np.round(std_dice_scoreArtery, 4)) + ', Min: ' + str(np.round(min_dice_scoreArtery, 4)) + ', Max: ' + str(np.round(max_dice_scoreArtery, 4)) + '  <-- Artery (incl. lumen)')
    logger.info('Mean: ' + str(np.round(avg_dice_scoreLumen, 4)) + ', Std: ' + str(np.round(std_dice_scoreLumen, 4)) + ', Min: ' + str(np.round(min_dice_scoreLumen, 4)) + ', Max: ' + str(np.round(max_dice_scoreLumen, 4)) + '  <-- Artery lumen')

    if saveResults:
        figPath = resultsPath + '/QuantitativeResults'
        if not os.path.exists(figPath):
            os.makedirs(figPath)

        disease = diseaseModels[evaluatorID]

        np.save(figPath + '/' + disease + '_tubuliDice.npy', np.array(allClassEvaluators[evaluatorID][0].diceScores))
        np.save(figPath + '/' + disease + '_glomDice.npy', np.array(allClassEvaluators[evaluatorID][1].diceScores))
        np.save(figPath + '/' + disease + '_tuftDice.npy', np.array(allClassEvaluators[evaluatorID][2].diceScores))
        np.save(figPath + '/' + disease + '_veinsDice.npy', np.array(allClassEvaluators[evaluatorID][3].diceScores))
        np.save(figPath + '/' + disease + '_arteriesDice.npy', np.array(allClassEvaluators[evaluatorID][4].diceScores))
        np.save(figPath + '/' + disease + '_lumenDice.npy', np.array(allClassEvaluators[evaluatorID][5].diceScores))

        np.save(figPath + '/' + disease + '_detectionResults.npy', np.stack((precisionsTub, precisionsGlom, precisionsTuft, precisionsVeins, precisionsArtery, precisionsLumen)))

def convert_labelmap_to_rgb(labelmap):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    """
    n_classes = labelmap.max()
    colors = torch.tensor([[  0,   0,   0], # Black
                           [255,   0,   0], # Red
                           [  0, 255,   0], # Green
                           [  0,   0, 255], # Blue
                           [  0, 255, 255], # Cyan
                           [255,   0, 255], # Magenta
                           [255, 255,   0], # Yellow
                           [139,  69,  19], # Brown (saddlebrown)
                           [128,   0, 128], # Purple
                           [255, 140,   0], # Orange
                           [255, 255, 255]], dtype=torch.uint8) # White
    result = torch.zeros(size=(labelmap.size()[0], labelmap.size()[1], 3), dtype=torch.uint8)
    for i in range(1, n_classes+1):
        result[labelmap == i] = colors[i]

    return result.permute(2, 0, 1)

def getColorMapForLabelMap():
    return ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'brown', 'orange', 'purple', 'white']

def saveFigureResults(img, outputPrediction, postprocessedPrediction, finalPredictionRGB, GT, preprocessedGT, preprocessedGTrgb, fullResultPath, alpha=0.4):
    customColors = getColorMapForLabelMap()
    max_number_of_labels = len(customColors)
    assert outputPrediction.max() < max_number_of_labels, 'Too many labels -> Not enough colors available in custom colormap! Add some colors!'
    customColorMap = mpl.colors.ListedColormap(getColorMapForLabelMap())

    # avoid brown color (border visualization) in output for final GT and prediction
    postprocessedPrediction[postprocessedPrediction==7] = 0
    preprocessedGT[preprocessedGT==7] = 0

    # also dilate tubuli here
    postprocessedPrediction[binary_dilation(postprocessedPrediction==1)] = 1

    predictionMask = np.ma.masked_where(postprocessedPrediction == 0, postprocessedPrediction)

    plt.figure(figsize=(16, 8.1))
    plt.subplot(241)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(outputPrediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(postprocessedPrediction, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(finalPredictionRGB)
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(img[(img.shape[0]-outputPrediction.shape[0])//2:(img.shape[0]-outputPrediction.shape[0])//2+outputPrediction.shape[0],(img.shape[1]-outputPrediction.shape[1])//2:(img.shape[1]-outputPrediction.shape[1])//2+outputPrediction.shape[1],:])
    plt.imshow(predictionMask, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1, alpha = alpha)
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(GT, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(preprocessedGT, cmap=customColorMap, vmin = 0, vmax = max_number_of_labels-1)
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(preprocessedGTrgb)
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(fullResultPath)
    plt.close()

def getOneHotEncoding(imgBatch, labelBatch):
    """
    :param imgBatch: image minibatch (FloatTensor) to extract shape and device info for output
    :param labelBatch: label minibatch (LongTensor) to be converted to one-hot encoding
    :return: One-hot encoded label minibatch with equal size as imgBatch and stored on same device
    """
    if imgBatch.size()[1] != 1: # Multi-label segmentation otherwise binary segmentation
        labelBatch = labelBatch.unsqueeze(1)
        onehotEncoding = torch.zeros_like(imgBatch)
        onehotEncoding.scatter_(1, labelBatch, 1)
        return onehotEncoding
    return labelBatch