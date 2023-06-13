#Import packages
import requests
import pandas as pd
import numpy as np
import io

#be_datahive API Wrapper
class be_datahive:
    BASE_URL = "https://be-server.herokuapp.com"
    DATASIZE = 460000
    DEFAULT_LIMIT = 500
    DEFAULT_OFFSET = 0
    DEFAULT_FEATURE_COLS = ["grna", "pam_sequence", "sequence", "full_context_sequence", "full_context_sequence_padded", "grna_sequence_match", "cell", "base_editor", "energy_1", "energy_2", "energy_3", "energy_4", "energy_5", "energy_6", "energy_7", "energy_8", "energy_9", "energy_10", "energy_11", "energy_12", "energy_13", "energy_14", "energy_15", "energy_16", "energy_17", "energy_18", "energy_19", "energy_20", "energy_21", "energy_22", "energy_23", "energy_24", "free_energy", "melt_temperature_grna", "melt_temperature_target"]
    DEFAULT_TARGET_COL = 'efficiency_full_grna_calculated'
    
    def __init__(self):
        pass

    #Request handling
    def get(self, endpoint, params=None):
        """
        Request handling
        @params:
            endpoint     - Required  : endpoint name (str)
            params       - Required  : request parameters (dict)
        """
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}")
        
        return response.json()
    
    #Returns study data
    def get_studies(self, id=None, year=None, base_editor=None):
        """
        Returns study data
        @params:
            id           - Required  : id (int)
            year         - Required  : publication year (int)
            base_editor  - Required  : base editors covered (str)
        """
        params = {'id': id, 'year': year, 'base_editor': base_editor}
        return pd.DataFrame(self.get('studies', params))

    # Returns efficiency data
    def get_efficiency(self, max_rows=None, limit=None, offset=None, **kwargs):
        """
        Returns efficiency data
        @params:
            max_rows     - Required  : max data rows to get sample (int)
            limit        - Required  : request limit (int)
            offset       - Required  : request offset (int)
            kwargs       - Required  : request parameters (dict)
        """
        return self._get_data_batch('efficiency', max_rows, limit, offset, **kwargs)

    # Returns bystander data    
    def get_bystander(self, max_rows=None, limit=None, offset=None, **kwargs):
        """
        Returns bystander data
        @params:
            max_rows     - Required  : max data rows to get sample (int)
            limit        - Required  : request limit (int)
            offset       - Required  : request offset (int)
            kwargs       - Required  : request parameters (dict)
        """
        return self._get_data_batch('bystander', max_rows, limit, offset, **kwargs)

    # Batch data request handling
    def _get_data_batch(self, endpoint, max_rows, limit, offset, **kwargs):
        """
        Batch data request handling
        @params:
            endpoint     - Required  : endpoint name (str)
            max_rows     - Required  : max data rows to get sample (int)
            limit        - Required  : request limit (int)
            offset       - Required  : request offset (int)
            kwargs       - Required  : request parameters (dict)
        """
        limit = self.DEFAULT_LIMIT if limit is None else limit
        offset = self.DEFAULT_OFFSET if offset is None else offset
        data = []
        while True:
            params = {**kwargs, 'limit': limit, 'offset': offset}
            batch = self.get(endpoint, params)
            if not batch:
                break
            data.extend(batch)
            offset += limit

            if max_rows == None:
                print(f"Downloaded {offset/self.DATASIZE*100}% of the database")
            else:
                print(f"Downloaded {offset/max_rows*100}% of the request")

            if max_rows!= None and offset >= max_rows:
                break
        
        data = pd.DataFrame(data)
        data.replace({None: np.nan}, inplace=True)

        return data

    #Convert byte arrays into numpy
    def _convert_to_numpy_array(self, df,columns):
        """
        Convert byte arrays into numpy
        @params:
            df           - Required  : data frame (pd.Dataframe)
            columns      - Required  : columns to be converted (list)
        """
        for c in columns:
            val_list = []
            for val in df[c].values:
                # Convert bytes to BytesIO
                buf1 = io.BytesIO(val.encode('latin-1'))
                arr = np.load(buf1, allow_pickle=True)
                val_list.append(arr)
            
            temp_df = pd.DataFrame({c: val_list})
            df.drop(c,inplace=True, axis=1)
            df = pd.concat([df,temp_df],axis=1)

        return df

    #Convert dataframe numpy arrays for efficiency models
    def get_efficiency_ml_arrays(self, df, encoding='raw', target_col=None, clean=True):
        """
        Convert dataframe numpy arrays for efficiency models
        @params:
            df           - Required  : data frame (pd.Dataframe)
            encoding     - Required  : encoding standard (str) | 'raw', 'one-hot', or 'hilbert-curve'
            target_col   - Required  : model target (str)
            clean        - Required  : clean up dataframe by replacing None and NaN (bool)

        """
        target_col = self.DEFAULT_TARGET_COL if target_col is None else target_col

        # Select encodings
        if encoding == 'raw':
            feature_cols = self.DEFAULT_FEATURE_COLS
        elif encoding == 'one-hot':
            encoding_cols = [c for c in df.columns if c.startswith("one_hot")] 
            purged_features = [c for c in self.DEFAULT_FEATURE_COLS if f"one_hot_{c}" not in encoding_cols]
            feature_cols = list(purged_features) + list(encoding_cols)

            #Convert byte arrays into numpy
            df = self._convert_to_numpy_array(df, encoding_cols)
        elif encoding == 'hilbert-curve':
            encoding_cols = [c for c in df.columns if c.startswith("hilbert_curve")] 
            purged_features = [c for c in self.DEFAULT_FEATURE_COLS if f"hilbert_curve_{c}" not in encoding_cols]
            feature_cols = list(purged_features) + list(encoding_cols)

            #Convert byte arrays into numpy
            df = self._convert_to_numpy_array(df, encoding_cols)
        else:
            raise ValueError(f'Unknown encoding: {encoding}')

        #Tidy up dataframe
        if clean:
            #Drop all rows that have no data
            df.dropna(subset=[target_col],axis=0, how='all', inplace=True)
            df.replace(np.nan,0, inplace=True)
            
        features = df[feature_cols].to_numpy()
        target = df[target_col].to_numpy()
        return features, target

    # Convert dataframe numpy arrays for bystander models
    def get_bystander_ml_arrays(self, df, encoding='raw', bystander_type = 'edited', clean=True):
        """
        Convert dataframe numpy arrays for bystander models
        @params:
            df               - Required  : data frame (pd.Dataframe)
            encoding         - Required  : encoding standard (str) | 'raw', 'one-hot', or 'hilbert-curve'
            bystander_type   - Required  : bystander task either 'edited' or 'outcome' (str)
            clean            - Required  : clean up dataframe by replacing None and NaN (bool)
        """
        # Select bystander type
        if bystander_type == 'edited':
            targets_cols = [c for c in df.columns if c.startswith('Position') and ' ' not in c]   
        elif bystander_type == 'outcome':
            targets_cols = [c for c in df.columns if not c.startswith("Position_")]
        else:
            raise ValueError(f'Unknown bystander type: {bystander_type}')

        # Select encodings
        if encoding == 'raw':
            feature_cols = self.DEFAULT_FEATURE_COLS
        elif encoding == 'one-hot':
            encoding_cols = [c for c in df.columns if c.startswith("one_hot")] 
            purged_features = [c for c in self.DEFAULT_FEATURE_COLS if f"one_hot_{c}" not in encoding_cols]
            feature_cols = list(purged_features) + list(encoding_cols)

            #Convert byte arrays into numpy
            df = self._convert_to_numpy_array(df, encoding_cols)
        elif encoding == 'hilbert-curve':
            encoding_cols = [c for c in df.columns if c.startswith("hilbert_curve")] 
            purged_features = [c for c in self.DEFAULT_FEATURE_COLS if f"hilbert_curve_{c}" not in encoding_cols]
            feature_cols = list(purged_features) + list(encoding_cols)

            #Convert byte arrays into numpy
            df = self._convert_to_numpy_array(df, encoding_cols)
        else:
            raise ValueError(f'Unknown encoding: {encoding}')

        #Tidy up dataframe
        if clean:
            #Drop all rows that have no data
            df.dropna(subset=targets_cols,axis=0, how='all', inplace=True)
            df.replace(np.nan,0, inplace=True)

        features = df[feature_cols].to_numpy()
        target = df[targets_cols].to_numpy()
        return features, target

###############
#Example
###############
# api = be_datahive()
# efficiency_data = api.get_efficiency(limit=10000, offset=0)
# features, target = api.get_efficiency_ml_arrays(efficiency_data, encoding='one-hot', clean=True)
# bystander_data = api.get_bystander(max_rows=10000, limit=10000, offset=0)
# features, target = api.get_bystander_ml_arrays(bystander_data, encoding='one-hot', clean=True)