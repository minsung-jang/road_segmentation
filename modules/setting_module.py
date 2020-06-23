###############################################################################
##
##  setting_module : Read setting.param file
##  Copyright (C) 2020 Minsung Jang < msjang (at) iastate (dot) edu >
##
###############################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from os import path
from PIL import Image
from sklearn.model_selection import train_test_split

class SettingModule:
    
    total_df = pd.DataFrame()
    
    ## Input Data
    data_dirs=[]
    regions=[]
    img_key=""
    gt_key=""
    
    ## Strategy
    strategy_name = ""
    custom_prefix = ""
    
    ## Image
    input_img_size = 256
    n_resize = 1
    random_distortion = False
    
    ## Train and test datasets
    data_fraction = 1.0
    test_fraction = 0.5
    valid_fraction = 0.2
    random_seed = 0
    
    datalist_file = ""
    
    ## Learning paratmeters
    epoch = 100
    batch_size = 12
    es_patience = 10
    es_mindelta = 1e-2
    pl_factor = 0.05
    pl_patience = 10
    pl_mindelta = 1e-2
    
    ## Others
    training_time = 0.0
    
    ##### Functions: parse the list for each title
    def insertInputData(self, parse_list):
        line = 1
        self.custom_prefix = parse_list[line]
        data = parse_list[line+1]
        self.regions = parse_list[(line+2):]
        for region in self.regions:
            data_dir = data+"/"+region
            self.data_dirs.append(data_dir)
        
    def insertImageSettings(self, parse_list):
        line = 1
        self.input_img_size = int(parse_list[line])
        self.n_resize = int(parse_list[line+1])
        if ( parse_list[line+2] == ("True" or "true" or "yes" or "Yes") ):
            self.random_distortion = True
    
    def insertTrainTest(self, parse_list):
        line = 1
        self.data_fraction = float(parse_list[line])
        self.test_fraction = float(parse_list[line+1])
        self.valid_fraction = float(parse_list[line+2])
        self.random_seed = int(parse_list[line+3])
        
        self.custom_prefix = self.custom_prefix+str(self.random_seed)
        self.datalist_file = './output/'+self.custom_prefix+'_dataSummary.csv'
        
    def insertLearning(self, parse_list):
        line = 1
        self.epoch = int(parse_list[line])
        self.batch_size = int(parse_list[line+1])
        self.es_patience = int(parse_list[line+2])
        self.es_mindelta = float(parse_list[line+3])
        self.pl_factor = float(parse_list[line+4])
        self.pl_patience = int(parse_list[line+5])
        self.pl_mindelta = float(parse_list[line+6])

    ##### Parse the "setting.param"
    
    def parseSettingFile(self, setting_file):
        
        setting_txt_list = setting_file.read().splitlines()
        
        s_list = []

        ## parse the titles first
        ## : Input Data, Image Settings, Training and Test sets, Learning
        title_list = [] 
        
        for s in setting_txt_list:
            
            if ( s.find(':') > 0 ):
                content = s[(s.find(':')+1):].strip()
                title_list.append(content)
                
            if ( s.find('# ') > 0 ):
                s_list.append(title_list)
                title_list=[]
                title = s[(s.find('# ')):(s.find(' #')+2)]
                title_list.append(title)
        
        s_list.append(title_list)
        s_list = s_list[1:]
        
        ## Parse contents for each title
        self.insertInputData(s_list[0])
        
        self.insertImageSettings(s_list[1])
        
        self.insertTrainTest(s_list[2])
        
        self.insertLearning(s_list[3])
        
    def getInitSettings(self):
        
        setting_file_path = "./setting.param"
        
        try:
            if (not path.exists(setting_file_path)):
                raise NameError
            
        except NameError:
            print("There is no setting file")

        setting_file = open(setting_file_path,"r")
        print("Setting file is loaded ...", setting_file_path)
        print("\n")
        
        self.parseSettingFile(setting_file)
        
    def getTotalDataset(self):
        
        colname='filename'
        
        total_df = []
        
        for data_dir in self.data_dirs:

            filename = pd.DataFrame(os.listdir(data_dir),columns=[colname])
            img_id = filename[colname].map(lambda s: s.split('.')[0]) 
            img_id = img_id[img_id.values != '.ipynb'] ## Remove garbage namings
            img_id = img_id[img_id.values != '.']
            
            ## Convert "gt_id" to a dataframe "img_df"
            img_df = pd.DataFrame(columns=['ID'])
            img_df['ID']=img_id.values
            img_df.insert(1,'Region', self.data_dirs.index(data_dir))
            img_df = img_df.sample(n=int(np.ceil(len(img_df)*self.data_fraction)),
                                   random_state=self.random_seed)
            class_df = self.classifyTrainTest(img_df)
            total_df.append(class_df)
        
        ## Combine the lists from the different states by row
        total_df = pd.concat(total_df)
        total_df = total_df.reset_index(drop=True)
        total_df.insert(0,'nID',total_df.index)
        total_df.to_csv(self.datalist_file, index=False, header=True)
        
        return total_df
            
    def classifyTrainTest(self, img_df):
        
        train0_df, test_df = train_test_split(img_df, test_size=self.test_fraction,
                                              random_state=self.random_seed)
        train_df, valid_df = train_test_split(train0_df, test_size=self.valid_fraction,
                                              random_state=self.random_seed)

        train_df.insert(2,"class","L") 
        valid_df.insert(2,"class","V") 
        test_df.insert(2,"class","T")
        
        class_df = [train_df, valid_df, test_df]
        class_df = pd.concat(class_df)
        class_df = class_df.reset_index(drop=True)
        
        return class_df
        
    def loadClassfiedDataset(self):
        total_df=pd.DataFrame()
        data_list_file_exists=path.exists(self.datalist_file)

        if (data_list_file_exists):
            print("Try to load the existing file: " + self.datalist_file +"...\n")
            total_df = pd.read_csv(self.datalist_file)

        else:

            print("Try to generate a new combination of datasets: "+self.datalist_file+"...\n")
            total_df = self.getTotalDataset()

        for region in self.regions:
            region_idx = self.regions.index(region)
            region_df = total_df[total_df['Region']==region_idx]
            print("===== Region [", region, "] :", len(region_df),"images =====")
            print("--- Train # :", len(region_df[region_df['class']=='L']), "images")
            print("--- Valid # :", len(region_df[region_df['class']=='V']), "images")
            print("---  Test # :", len(region_df[region_df['class']=='T']), "images")
            print("\n")
            
        return total_df
        
    def plotImageSize(self):

        plot_title = ['Width','Height', 'W / H']
        clr_title = ['forestgreen','tomato', 'navy']
        n_plots = len(self.regions)
        
        print("Plotting the histrograms for the widths, height and their ratios...")
        
        for i in range(n_plots):
            region_df=self.total_df[self.total_df["Region"]==i]
            width_df=np.array(region_df['width'].values, dtype=np.float)
            height_df=np.array(region_df['height'].values, dtype=np.float)
            ratio_df = np.array(width_df / height_df, dtype=np.float)
            width_df=np.array(width_df, dtype=np.int)
            height_df=np.array(height_df, dtype=np.int)
            hist_list = [width_df, height_df, ratio_df]
            
            fig=plt.figure((i+1), figsize=(15,4), dpi=360)
            
            print("Plotting ", self.regions[i],"...")
            for k in range(len(hist_list)):
                title = plot_title[k]+" : "+self.regions[i]
                df = hist_list[k]
                mean = int(np.mean(df))
                std = int(np.std(df))
                max_ = max(df)
                min_ = min(df)
                
                if (k==2):
                    mean = round(np.mean(df),2)
                    std = round(np.std(df),2)
                    max_ = round(max(df),2)
                    min_ = round(min(df),2)
                    
                mean_str = "Mean : "+str(mean)
                max_str = "Max : "+str(max_)
                min_str = "Min : "+str(min_)
                std_str ="Stddev : "+str(std)
                num_str ="# images: "+str(len(region_df))
        
                labels = [mean_str, max_str, min_str, std_str, num_str]
                bin_edges = list(np.linspace(min_,max_,6))
                handles = [Rectangle((0,0),0,0,color="white") for c in labels]
        
                plt.subplot(1, len(hist_list), (k+1))
                plt.hist(df, bins=bin_edges,density=False,
                         color=clr_title[k],histtype='bar',edgecolor='black', alpha=0.7)
                plt.title(title)
                xlabel='Pixels'
                if (k==2):
                    xlabel='Ratio'
                plt.xlabel(xlabel)
                plt.xticks(bin_edges)
                plt.ylabel('Number of Images')
                plt.legend(handles, labels,fontsize=10)
            plt.show()
            fig.savefig("./plots/fig_"+self.regions[i]+"_WH_pixels.png", dpi=fig.dpi)
    
    def checkImageSize(self):
        print("Start to check the sizes of the images...")
        add_df = pd.DataFrame(columns=["width","height"])
        n_data = len(self.total_df)
        percent_count = 0

        for nID in range(n_data):

            img_id = self.total_df["ID"][nID]
            region_id = self.total_df["Region"][nID]
            gt_filepath = self.data_dirs[region_id]+'/{}'.format(img_id)+self.gt_key+'.png'

            mask = Image.open(gt_filepath)
            width, height = mask.size

            add_df = add_df.append({'width': int(width), 'height': int(height)},ignore_index=True)

            if ( nID > 1 and (nID+1) % int(np.ceil(len(self.total_df)/20)) == 0 ):
                percent_count += 5
                print(percent_count, "% (" , (nID+1), " / ", n_data, ") has been completed")

        if ( percent_count < 100 ):
            percent_count = 100
            print(percent_count, "% (" , (nID+1), " / ", n_data, ") has been completed")
            
        comb_df = [self.total_df, add_df]
        comb_df = pd.concat(comb_df, axis=1)
        
        self.total_df = comb_df
        
        self.plotImageSize()
    
    def __init__(self):
        
        self.getInitSettings()
        self.total_df = self.loadClassfiedDataset()
