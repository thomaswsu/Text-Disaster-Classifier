import pandas
from pandas import ExcelFile, read_csv
import train
import runpy

""" You need to install pandas and xlrd to handle modern Excel files
    You also need to import SKLearn.
"""

if __name__ == "__main__":
    portionOfDataSet = -1
    while not(0 < portionOfDataSet <= 1):
        portionOfDataSet = float(input("Enter the portion of data set that you want to use. \
For example, 0.1 -> use 10% of the data: "))
        if not(0 < portionOfDataSet <= 1):
            print("Invalid portion selected. Must enter between 0 and 1")
    
    test_ratio = -1
    while not(0 < test_ratio <= 1):
        test_ratio = float(input("Enter the ratio of data do be used for testing vs training. \
For example, 0.2 would mean that 20% of the data would be used for testing and \
80% of the data would be used for training: "))
    test_ratio = 0.2 # fraction of data to for testing. Ex: 80% of data is used for training, 20% for verifying

    training_data = pandas.read_csv("Training Data.csv", low_memory=False, encoding='utf-8-sig')
    # test1(training_data, fractionOfDataSet, test_ratio)
    train.test2(training_data, 5, portionOfDataSet)


""" Sources:
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
https://stackabuse.com/text-classification-with-python-and-scikit-learn/
More sources in other files
"""






