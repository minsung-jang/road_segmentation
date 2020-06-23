import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle

import modules.batch_module as bm
from model.u_net import soft_cldice_coeff

def getPrefixString(_sm, model_name):
    
    prefix = _sm.custom_prefix+'_'+model_name
    
    return prefix

def getDiceCoeff(y_true, y_pred, thresh=1.0) : ## avoid the case divisor == 0
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum(y_true_f * y_pred_f) ## do AND operation and get the total sum
    score = 2.0 * ( intersection + thresh ) / ( sum(y_true_f) + sum(y_pred_f) + thresh )
    score = round(score, 3)
    
    return score

def getConfusionMatrix(y_true, y_pred) :
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    tn, fp, fn, tp  = confusion_matrix(y_true_f, y_pred_f).ravel()
    
    ##     true / false    : match pred - actual
    ## postivie / negative : pred yes or no
    
    tpr = round (tp / ( tp + fn ), 3 ) ## among "actual yes" ( True Positive Rate, Sensitivity )
    fpr = round( fp / ( tn + fp ), 3 ) ## among "acutal no"  ( False Poistive Rate, Specificity )
    ppv = round( tp / ( tp + fp ), 3 ) ## among "pred yes"   ( Positive Predictive Value )
    npv = round( tn / ( tn + fn ), 3 ) ## among "pred no"    ( Negative Predictive Value )
 
    return tpr, fpr, ppv, npv


def plotPredictedImages(y_true, y_pred, img, score, path):
    
    frame_size = y_true.shape[0]
    thick = 5
    
    w = int(frame_size/3*4)
    size = (w,frame_size,3)
    begin = w - frame_size
    
    ## Vertical lines
    vertline = np.zeros((w,thick,3), dtype=int)
    vertline[0:begin,:,:]=255

    ## Image frames
    imgC = np.ones(size, dtype=int)*255
    true = np.ones(size, dtype=int)*255
    pred = np.ones(size, dtype=int)*255
    
    imgC[begin:w,:,:] = img
    true[begin:w,:,1] = true[(w-frame_size):w,:,1]-255*y_true
    true[begin:w,:,2] = true[(w-frame_size):w,:,2]-255*y_true
    pred[begin:w,:,1] = pred[(w-frame_size):w,:,1]-255*y_pred
    pred[begin:w,:,2] = pred[(w-frame_size):w,:,2]-255*y_pred
    
    ## Concatenation
    im_h = cv2.hconcat([vertline,imgC,vertline,true,vertline,pred,vertline])
    
    ## Horizontal line
    im_h[(begin-thick-1):(begin-1),:,:]=0

    horiline = np.zeros((thick,im_h.shape[1],3), dtype=int)
    im_h = cv2.vconcat([im_h,horiline])
    
    ## Put scores on the top of the image
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 

    # org 
    org = (int(im_h.shape[0]*0.6), int(im_h.shape[1]*0.07)) 

    # fontScale 
    fontScale = 3

    # Blue color in BGR 
    color = (0, 0, 0) 

    # Line thickness of 2 px 
    thickness = 10
    image = cv2.putText(im_h, 'Score:'+str(score), org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    
    cv2.imwrite(path, image)

def saveTestStatistics(_im, prefix, model, window_size, doResize=True, verbose=True):
    
    total_df = _im.total_df
    strategy_name = _im.strategy_name
    custom_prefix = _im.custom_prefix
    
    test_df = total_df[total_df['class']=='T']
    df_index = test_df.index
    test_df=test_df.reset_index(drop=True)
    
    ## Save test result
    test_string = './output/'+prefix+'_testStats'
    teststat_csv = test_string+'.csv'
    teststat_txt = test_string+'.txt'
    
    ## Log file
    output_file = open(teststat_txt,"w")
    output_file.write("Window Size: "+str(window_size)+'\n')
    output_file.write("Input Size: "+str(_im.input_img_size)+'\n')
    output_file.write("Elapsed time: "+str(_im.training_time)+'[sec] \n')
    if (doResize):
        output_file.write("Resize \n")
    else:
        output_file.write("Keep \n")
    output_file.close()
    print("Save a logfile: ", teststat_txt)
    
    tmp_df = pd.DataFrame(columns=['dice','TPR','FPR','PPV','NPV'])
    
    for i_row in range(len(test_df)):
        
        nID = test_df['nID'].iloc[i_row]
        y_pred, y_true=_im.predictSingleImage(model, nID, window_size, doResize, verbose)
        img = _im.loadSingleIMG(nID)
        ID = test_df['ID'].iloc[i_row]
        path = "./prediction/pred_"+ID+".png"
        
        dice_coeff = getDiceCoeff(y_true,y_pred)
        
        #tpr, fpr, ppv, npv = getConfusionMatrix(y_true, y_pred)
        tpr=fpr=ppv=npv=1
        
        tmp_df=tmp_df.append({'dice': dice_coeff, 'TPR':tpr, 'FPR':fpr, 'PPV': ppv, 'NPV': npv},ignore_index=True)
        
        plotPredictedImages(y_true, y_pred, img, dice_coeff, path)
       
    ## Concatenate dataframe
    stat_df = [ test_df, tmp_df]
    stat_df = pd.concat(stat_df, axis=1)
    stat_df.index=df_index
    stat_df.to_csv(teststat_csv, index=True, header=True)
    print("Save a dataframe: ", teststat_csv)
    
    return stat_df

def plotHistogram(stat_df, prefix):
    
    teststat = []
    stat_title = ['dice','TPR','FPR','PPV','NPV']
    plot_title = ['slDice Score','True Positive Rate (Sensitivity)', 'False Positive Rate (Specificity)',
                  'Postive Predictive Value ', 'Negative Predictive Value']
    clr_title = ['forestgreen', 'azure', 'tomato', 'navy','crimson']

    for i in range(len(stat_title)):
        title = stat_title[i]
        mean= round(np.mean(stat_df[title]),3)
        std= round(np.std(stat_df[title]),3)

        mean_str = "Mean : "+str(mean)
        std_str="Stddev: "+str(std)
        num_str="# images: "+str(len(stat_df))
        labels=[mean_str, std_str,num_str]
        handles = [Rectangle((0,0),0,0,color="white") for c in labels]

        bin_edges=list(np.linspace(0,1,11))
        if (std * 3 < 0.1 ):
            bin_edges = list(np.linspace(max(-0.2+round(mean,1),0), min(0.2+round(mean,1),1), 11))
        fig=plt.figure(figsize=[5,3], dpi=240)
        plt.hist(stat_df[stat_title[i]],
                 bins=bin_edges,
                 density=False,
                 histtype='bar',
                 color=clr_title[i],
                 edgecolor='black', alpha=0.7)

        plt.xlabel('Ratio')
        plt.xticks(bin_edges)
        plt.ylabel('Number of images')
        plt.title(plot_title[i])
        plt.legend(handles, labels,fontsize=12)
        fig.savefig('./plots/'+prefix+'fig_'+stat_title[i]+".png", dpi=fig.dpi)
    
