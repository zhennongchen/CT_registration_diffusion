import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_name):
     
        self.file_name = file_name
        self.data = pd.read_excel(file_name, dtype = {'dataset_ID': str, 'case_ID': str})
      
    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        batch_list = np.asarray(c['batch'])
        dataset_id_list = np.asarray(c['dataset_ID'])
        case_id_list = np.asarray(c['case_ID'])
        image_folder_list = np.asarray(c['image_folder'])
        
        return batch_list, dataset_id_list, case_id_list, image_folder_list
