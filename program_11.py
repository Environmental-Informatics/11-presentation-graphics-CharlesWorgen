#!/bin/env python
"""
2020/05/06
Modified by Charles Huang

Lab11 - Presentation Graphics

"""
#
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    #Clip data with provided parameter
    DataDF = DataDF.loc[startDate:endDate]
    #Record the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
       
    return( DataDF, MissingValues )

def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    #Reads in csv file from lab 10
    DataDF = pd.read_csv(fileName, header=0, delimiter=' ', parse_dates=['Date'],index_col=['Date'])
    
    return( DataDF )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    #Extract month from index into new column Month for grouping
    MoDataDF['Month'] = MoDataDF.index.month
    #Calculate mean value by month group
    MonthlyAverages = MoDataDF.groupby('Month').mean()
    
    return( MonthlyAverages )
    
# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }

    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    csvName = {"Annual":"Annual_Metrics.csv","Monthly":"Monthly_Metrics.csv"}  
    
    colorbyriver = {"Wildcat": 'red',"Tippe": 'blue'} # set color for each river
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    
    # process input datasets
    for file in fileName.keys():
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        
        # clip to last five year
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '2014-10-01', '2019-09-30' )
        
        #ploting daily flow for both streams for the last 5 years
        plt.figure(1)
        plt.plot(DataDF[file]['Discharge'], label=riverName[file], color=colorbyriver[file])   
    #Add in label,title,legend, etc.
    plt.xlabel('Date')
    plt.ylabel('Discharge (cfs)')
    plt.title('Daily flow for the last 5 years')
    plt.legend()
    #Ouput image
    plt.savefig('Daily.png', dpi=96)
    plt.show()
 
      
    #Importannual Annual metrics from csv
    DataDF['Annual'] = ReadMetrics(csvName['Annual'])
    DataDF['Annual'] = DataDF['Annual'].groupby('Station') # Separated by river
    
    #Annual coefficient of variation
    plt.figure(2)
    for river,df in DataDF['Annual']:
        plt.scatter(df.index.values,df['Coeff Var'].values, label=riverName[river],
                    color=colorbyriver[river])        
    #Add in label,title,legend, etc.
    plt.xlabel('Date')
    plt.ylabel('Coefficient of Variation')
    plt.title('Annual coefficient of variation')
    plt.legend()   
    #Ouput image
    plt.savefig('A_CV.png', dpi=96)
    plt.show()

    #Annual TQmean
    plt.figure(3)
    for river,df in DataDF['Annual']:
        plt.scatter(df.index.values,df['Tqmean'].values, label=riverName[river],
                    color=colorbyriver[river])
    #Add in label,title,legend, etc.
    plt.xlabel('Date')
    plt.ylabel('T-Q mean')
    plt.title('Annual T-Q mean')
    plt.legend()
    #Ouput image
    plt.savefig('A_TQ.png', dpi=96)
    plt.show() 
    
    #Annual R-B index
    plt.figure(4)
    for river,df in DataDF['Annual']:
        plt.scatter(df.index.values,df['R-B Index'].values, label=riverName[river],
                    color=colorbyriver[river])
    #Add in label,title,legend, etc.
    plt.xlabel('Date')
    plt.ylabel('R-B index')
    plt.title('Annual Richards-Baker Flashiness Index')
    plt.legend()
    #Ouput image
    plt.savefig('A_RB.png', dpi=96)
    plt.show() 
    
    #Importannual Monthly metrics from csv
    DataDF['Monthly'] = ReadMetrics(csvName['Monthly'])
    # Separated by river
    WCMo = DataDF['Monthly'][DataDF['Monthly']['Station'] == 'Wildcat']
    TPMo = DataDF['Monthly'][DataDF['Monthly']['Station'] == 'Tippe']
    
    #Calculate monthly average using function above
    WCMA = GetMonthlyAverages(WCMo)
    TPMA = GetMonthlyAverages(TPMo)
    
    #Average annual monthly flow
    plt.figure(5)
    plt.plot(WCMA['Mean Flow'],color = 'red')
    plt.plot(TPMA['Mean Flow'],color = 'blue')
    #Add in label,title,legend, etc.
    plt.xlabel('Month')
    plt.ylabel('Average monthly flow discharge (cfs)')
    plt.title('Average annual monthly flow')
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    #Ouput image
    plt.savefig('AvgMo.png', dpi=96)
    plt.show()
    
    #Return period of annual peak flow events
    #Select peak flow from annual metrics dataframe
    PFDF = DataDF['Annual'][['Peak Flow','Station']]
    
    for river,df in PFDF:
        #Sort Peak Flow from highest to lowest value
        pf_order = df.sort_values('Peak Flow', ascending=False)
        #Create ranks and assign each of these a rank from 1 to N
        #where rank 1 is the highest event, and rank N is the lowest event
        ranks = []
        for i in range(len(pf_order)):
            ranks.append((i)/(len(pf_order)+1))
        
        #Plotting
        plt.figure(6)        
        plt.scatter(ranks,pf_order['Peak Flow'].values, label=riverName[river],
                    color=colorbyriver[river])
    #Add in label,title,legend, etc.
    plt.xlabel('Exceedence Probability')
    plt.ylabel('Peak Discharge (cfs)')
    plt.title('Return Period of Annual Peak Flow Events')
    plt.legend()
    #Ouput image
    plt.savefig('Re_Pe.png', dpi=96)
    plt.show()