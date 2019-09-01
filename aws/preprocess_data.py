import json
import numpy as np
import os
import pandas as pd
import pickle

class Preprocessor(object):
    """
    Performs all the data preprocessing required to conver the data into
    a google cloud friendly json file.
    """

    def __init__(self):
        self.non_null_indices = [
            0, 1, 2, 3, 4, 5, 12, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
        ]
        self.categorical_features = [
            'merchant_category',
            'merchant_group',
            'name_in_email',
            'status_last_archived_0_24m',
            'status_2nd_last_archived_0_24m',
            'status_3rd_last_archived_0_24m',
            'status_max_archived_0_6_months',
            'status_max_archived_0_12_months',
            'status_max_archived_0_24_months',
        ]


    def _handle_dummy_vars(self, data, data_directory):
        '''
        Creates dummy variables for all the categorical data.
        '''
        non_cat_features = [
            val for val in data.columns if val not in self.categorical_features
        ]
        cat_data = data.loc[:, self.categorical_features]
        non_cat_data = data.loc[:, non_cat_features]

        encoder_path = os.path.join(data_directory, 'dummy_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)

        dummified_cat_data = pd.DataFrame(data=encoder.transform(cat_data).todense())
        dummy_data = pd.concat((non_cat_data, dummified_cat_data), axis=1)

        return dummy_data


    def preprocess(self, data_directory):
        """
        Transforms a pandas dataframe into a numpy array that can be fed
        into the XGBoost model.

        Args:
            data_directory: filepath where the data.csv and encoder are stored.
        """
        data_filepath = os.path.join(data_directory, 'dataset.csv')
        dataset = pd.read_csv(data_filepath, sep=';')

        # Remove columns with excessive NULL ratios
        non_null_data = dataset.iloc[:, self.non_null_indices]

        # Handle imputation: All 0 fills since that is the mode value for the columns
        # with null values.
        imputed_dataset = non_null_data.fillna(0)

        # Dummify Categorical data
        dummy_data = self._handle_dummy_vars(imputed_dataset, data_directory)

        # Convert True/False to 0/1
        dummy_data['has_paid'] = dummy_data['has_paid'] * 1

        # Set uuid as the index, drop the labels, and convert to numpy array
        dataset = dummy_data.set_index('uuid').drop('default', axis=1).values

        self.dataset = dataset


    def convert_to_API_friendly_format(self, data_directory, subset=True):
        '''
        Converts the numpy array of prediction data an API friendly format.

        Args:
            data_directory: the filepath to where we're going to store the json.
                Also, needs to be where data.csv and the 1-hot-encoder are stored.
        '''
        try:
            dataset = self.dataset
        except:
            self.preprocess(data_directory)
            dataset = self.dataset

        if subset:
            output_filepath = os.path.join(data_directory, 'api_demo_dataset.txt')
        else:
            output_filepath = os.path.join(data_directory, 'api_full_dataset.txt')

        with open(output_filepath, 'w') as f:
            if subset:
                for row in dataset[:10]:
                    payload = {'payload': list(row)}
                    json.dump(payload, f)
                    f.write('\n\n')

            else:
                for row in dataset:
                    payload = {'payload': list(row)}
                    json.dump(payload, f)
                    f.write('\n\n')
