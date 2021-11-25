import pandas as pd
import numpy as np
import math
import sys
import os.path
import shapely.geometry
import shapely.ops
import matplotlib.pyplot as plt
import powerlaw as pl
from scipy import stats
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from multiprocessing import Pool
import sklearn
import scipy
sys.setrecursionlimit(100000)
from itertools import combinations
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.rm'] = 'Arial'


class MobilityDiversityFunctions:
    
    def __init__(self):
        pass
    
    
    @staticmethod
    def calculate_mobility_diversity(all_options, probabilities):
        """
        Compute mobility diversity based on all the options and the probabilities
        
        Parameters
        ----------
        all_options : list
            list of all the options (travel destinations)

        probabilities : list
            list of all the probabilities of going to the all_options

        Returns
        -------
        mobility diversity : a float value
        """
        return stats.entropy(probabilities, base=2)/np.log2(len(all_options))
    
    
    @staticmethod
    def return_probabilities(data_group, destination_column, expansion_column, all_options):
        """
        Compute the probabilities based on the column of expansion factor
        
        Parameters
        ----------
        data_group : dataframe
            dataframe with all the travel information

        destination_column : string
            name of the column that have the travel destination
            
        expansion_column : string
            name of the column that have the expansion factor of each travel
            
        all_options : list
            list of all the options (unique travel destinations)

        Returns
        -------
        probabilities : list of probabilities 
        """
        counts = data_group.groupby(destination_column)[expansion_column].count()\
                                .reindex(all_options, fill_value=0).reset_index(name='count')
        counts['prob'] = counts['count']/counts['count'].sum()
        return counts['prob'].values
    
    @staticmethod
    def calculate_mobility_diversity_by_sampling(all_options, data_group, allow_duplicates=False, expansion_column = 'FAT_EXP', destination_column='TRAJECTORY', percentage_sample=0.8, simulations=1000, fixed_sample_size=None, expand=True):
        """
        Compute the mobility diversity using sampling technique
        
        Parameters
        ----------
        
        all_options : list
            list of all the options (unique travel destinations)
        
        data_group : dataframe
            dataframe with all the travel information
        
        allow_duplicates : Boolean
            allow the sampling to choose more than once the same travel
        
        percentage_sample : float
            value between 0 and 1 to establish the fraction of travels that will be randomly chosen from the data
        
        simulations : int
            number of times that the mobility diversity will be computed
        
        get_min_ : int
            establish the minimal of travels that can be selected
       
        destination_column : string
            name of the column that have the travel destination
            
        expansion_column : string
            name of the column that have the expansion factor of each travel

        Returns
        -------
        mobility diversity : list of float values
        """
        if expand:
            # depending on your data you have to make sure that the expansion factor is positive and different from Nan
            # data_group[expansion_column] = data_group[expansion_column].abs().fillna(0)
            # In this way, you will treat the travels as an integer value
            data_group = data_group.loc[data_group.index.repeat(data_group[expansion_column])].copy()
        
        entropies_ = []
        
        sample_size = len(data_group)
        if fixed_sample_size != None and fixed_sample_size < sample_size:
            sample_size = fixed_sample_size      
        else:
            sample_size = int(number_travels*percentage_sample)
        
        for simulation in range(simulations):
            get_sample = data_group.sample(n=sample_size, replace=allow_duplicates)[[destination_column,expansion_column]].copy()
            probabilities = MobilityDiversityFunctions.return_probabilities(data_group, destination_column, expansion_column, all_options)
            entropies_.append(MobilityDiversityFunctions.calculate_mobility_diversity(all_options, probabilities))
            del get_sample, probabilities

        return entropies_
    
    
    @staticmethod
    def calculate_mobility_diversity_fast_shuffling_int_expansion_factor(all_options, data_group, expand=False, get_min_ = None, expansion_column = 1, destination_column='TRAJECTORY', percentage_sample=0.8, simulations=1000, replace_false=False, print_evolution=False):
        """
        Compute the mobility diversity using sampling technique
        
        Parameters
        ----------
        
        all_options : list
            list of all the options (unique travel destinations)
        
        data_group : dataframe
            dataframe with all the travel information
        
        expand : Boolean
            allow the dataframe to be expanded to the number of travels considering the expansion factor
        
        allow_duplicates : Boolean
            allow the sampling to choose more than once the same travel
        
        percentage_sample : float
            value between 0 and 1 to establish the fraction of travels that will be randomly chosen from the data
        
        simulations : int
            number of times that the mobility diversity will be computed
        
        get_min_ : int
            establish the minimal of travels that can be selected
       
        destination_column : string
            name of the column that have the travel destination
            
        expansion_column : string
            name of the column that have the expansion factor of each travel
            
        print_evolution : Boolean
            allow printing the number of simulations that were executed 

        Returns
        -------
        mobility diversity : list of float values
        """
        
        if expand:
            # depending on your data you have to make sure that the expansion factor is positive and different from Nan
            # data_group[expansion_column] = data_group[expansion_column].abs().fillna(1.0)
            # In this way, you will treat the travels as an integer value
            data_group = data_group.loc[data_group.index.repeat(data_group[expansion_column])].copy()
        
        number_people = len(data_group)
        if get_min_ != None and get_min_ < number_people:
            number_people = get_min_
               
        sample_size_ = int(number_people*percentage_sample)
        all_loc = sklearn.utils.shuffle(data_group[destination_column].values)
        
        entropies_ = []
        
        for time in range(simulations):
            
            if print_evolution and time % 1000 == 0:
                print(time)

            np.random.shuffle(all_loc)
            destinations = pd.DataFrame(all_loc[:sample_size_])[0].value_counts(normalize=True)\
            .reindex(all_options, fill_value=0).values
            entropies_.append(MobilityDiversityFunctions.calculate_mobility_diversity(all_options, destinations))

            
        return entropies_
    

    
    
   
    

