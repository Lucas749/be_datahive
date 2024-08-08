#Import packages
import requests
import pandas as pd
import numpy as np
import io
import pickle

#be_datahive API Wrapper
class be_datahive:
    BASE_URL = "https://be-server.herokuapp.com"
    DATASIZE = 460000
    DEFAULT_LIMIT = 250
    DEFAULT_OFFSET = 0
    RAW_FEATURE_COLS = ["grna", "pam_sequence", "sequence", "full_context_sequence", "full_context_sequence_padded"]
    DEFAULT_FEATURE_COLS = ["grna_sequence_match", "energy_1", "energy_2", "energy_3", "energy_4", "energy_5", "energy_6", "energy_7", "energy_8", "energy_9", "energy_10", "energy_11", "energy_12", "energy_13", "energy_14", "energy_15", "energy_16", "energy_17", "energy_18", "energy_19", "energy_20", "energy_21", "energy_22", "energy_23", "energy_24", "free_energy", "melt_temperature_grna", "melt_temperature_target"]
    DEFAULT_TARGET_COL = 'efficiency_full_grna_calculated'
    REQUEST_TIMEOUT = 500 # Set request timeout to 500 seconds

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
        response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)

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

    def get_available_base_editors(self, df):
        """
        List all available base editors values
        @params:
            df           - Required  : data frame (pd.Dataframe)
        """
        values = df['base_editor'].unique()
        
        return values
    
    def get_available_cells(self, df):
        """
        List all available cell values
        @params:
            df           - Required  : data frame (pd.Dataframe)
        """
        values = df['cell'].unique()
        
        return values

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

        #Handling limit based on max_rows
        if max_rows != None:
            if max_rows < limit:
                limit = max_rows

        #Save initial offset for progress calculations
        if offset != None:
            initial_offset = offset
        else:
            initial_offset = self.DEFAULT_OFFSET

        data = []
        while True:
            params = {**kwargs, 'limit': limit, 'offset': offset}
            batch = self.get(endpoint, params)
            if not batch:
                break
            data.extend(batch)
            offset += limit

            if max_rows is None:
                total_to_download = self.DATASIZE - initial_offset
                percentage = ((offset - initial_offset) / total_to_download) * 100
                print(f"Downloaded {min(percentage, 100):.2f}% of the database")
            else:
                total_to_download = max_rows
                percentage = ((offset - initial_offset) / total_to_download) * 100
                print(f"Downloaded {min(percentage, 100):.2f}% of the request")

            if max_rows!= None and offset >= max_rows:
                break
        
        data = pd.DataFrame(data)
        data.replace({None: np.nan}, inplace=True)

        return data

    #Convert byte arrays into numpy
    def _convert_to_numpy_array(self, df, columns):
        """
        Convert byte arrays into numpy
        @params:
            df           - Required  : data frame (pd.Dataframe)
            columns      - Required  : columns to be converted (list)
        """
        for c in columns:
            val_list = []
            for val in df[c].values:
                if isinstance(val, (float, int)):
                    # If the value is already a number, just append it
                    val_list.append(np.array([val]))
                elif isinstance(val, np.ndarray):
                    # If the value is already a numpy array, ensure it's C-contiguous
                    val_list.append(np.ascontiguousarray(val))
                else:
                    try:
                        # Try to load the data as a pickle
                        buf1 = io.BytesIO(val.encode('latin-1'))
                        arr = np.load(buf1, allow_pickle=True)
                    except Exception as e:
                        # If loading as pickle fails, handle it as raw byte data
                        try:
                            arr = np.frombuffer(val.encode('latin-1'), dtype=np.uint8)
                        except AttributeError:
                            # If encoding fails, it might already be bytes
                            arr = np.frombuffer(val, dtype=np.uint8)
                    val_list.append(arr)

            temp_df = pd.DataFrame({c: val_list})
            df.reset_index(inplace=True, drop=True)
            df = df.drop(c, axis=1)
            df = pd.concat([df, temp_df], axis=1)
        return df
    
    # Flatten nested arrays and update feature columns
    def _flatten_features(self, features, feature_cols):
        """
        Flatten nested arrays and update feature columns
        @params:
            features      - Required  : array (np.array)
            feature_cols  - Required  : list of feature column names (list)
        """
        flat_features = []
        updated_feature_cols = []
        
        for i in range(features.shape[0]):
            flat_row = []
            for j, col in enumerate(features[i]):
                if isinstance(col, np.ndarray):
                    flat_row.extend(col)
                    # Update feature column names with original name and indices
                    updated_feature_cols.extend([f"{feature_cols[j]}_{k}" for k in range(len(col))])
                else:
                    flat_row.append(col)
                    if i == 0:
                        updated_feature_cols.append(feature_cols[j])
            flat_features.append(flat_row)
            
        return np.array(flat_features), updated_feature_cols

    #Convert dataframe numpy arrays for efficiency models
    def get_efficiency_ml_arrays(self, df, encoding='raw', target_col=None, clean=True, flatten=True, base_editor=None, cell=None):
        """
        Convert dataframe numpy arrays for efficiency models
        @params:
            df           - Required  : data frame (pd.Dataframe)
            encoding     - Required  : encoding standard (str) | 'raw', 'one-hot', or 'hilbert-curve'
            target_col   - Required  : model target (str)
            clean        - Optional  : clean up dataframe by replacing None and NaN (bool)
            flatten      - Optional  : flatten nested arrays (bool)
            base_editor  - Optional  : subset by base editor (str or list of str)
            cell         - Optional  : subset by cell (str or list of str)
        """
        target_col = self.DEFAULT_TARGET_COL if target_col is None else target_col

        # Subset the dataframe by base_editor or cell if provided
        if base_editor is not None:
            if isinstance(base_editor, str):
                base_editor = [base_editor]
            df = df[df['base_editor'].isin(base_editor)]
        
        if cell is not None:
            if isinstance(cell, str):
                cell = [cell]
            df = df[df['cell'].isin(cell)]

        # Select encodings
        if encoding == 'raw':
            feature_cols = self.RAW_FEATURE_COLS + self.DEFAULT_FEATURE_COLS

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
            

        data_info = {'features': feature_cols, 'target': target_col}
        features = df[feature_cols].to_numpy()
        target = df[target_col].to_numpy()

        #Flatten numpy arrays
        if flatten:
            # Flatten all features that contain arrays and get updated columns
            features, updated_feature_cols = self._flatten_features(features, feature_cols)
            data_info['features'] = updated_feature_cols 

        return features, target, data_info

    # Convert dataframe numpy arrays for bystander models
    def get_bystander_ml_arrays(self, df, encoding='raw', bystander_type = 'edited', clean=True, flatten=True, base_editor=None, cell=None):
        """
        Convert dataframe numpy arrays for bystander models
        @params:
            df               - Required  : data frame (pd.Dataframe)
            encoding         - Required  : encoding standard (str) | 'raw', 'one-hot', or 'hilbert-curve'
            bystander_type   - Required  : bystander task either 'edited' or 'outcome' (str)
            clean            - Optional  : clean up dataframe by replacing None and NaN (bool)
            flatten          - Optional  : flatten nested arrays (bool)
            base_editor      - Optional  : subset by base editor (str or list of str)
            cell             - Optional  : subset by cell (str or list of str)
        """
        # Select bystander type
        if bystander_type == 'edited':
            targets_cols = [c for c in df.columns if c.startswith('Position') and ' ' not in c]   
        elif bystander_type == 'outcome':
            targets_cols = [c for c in df.columns if not c.startswith("Position_")]
        else:
            raise ValueError(f'Unknown bystander type: {bystander_type}')

        # Subset the dataframe by base_editor or cell if provided
        if base_editor is not None:
            if isinstance(base_editor, str):
                base_editor = [base_editor]
            df = df[df['base_editor'].isin(base_editor)]
        
        if cell is not None:
            if isinstance(cell, str):
                cell = [cell]
            df = df[df['cell'].isin(cell)]

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
        
        data_info = {'features': feature_cols, 'target': targets_cols}
        features = df[feature_cols].to_numpy()
        target = df[targets_cols].to_numpy()

        #Flatten numpy arrays
        if flatten:
            # Flatten all features that contain arrays and get updated columns
            features, updated_feature_cols = self._flatten_features(features, feature_cols)
            data_info['features'] = updated_feature_cols 

        return features, target, data_info

###############
#Example
###############
# api = be_datahive()
# efficiency_data = api.get_efficiency(limit=10000, offset=0)
# features, target = api.get_efficiency_ml_arrays(efficiency_data, encoding='one-hot', clean=True)
# bystander_data = api.get_bystander(max_rows=10000, limit=10000, offset=0)
# features, target = api.get_bystander_ml_arrays(bystander_data, encoding='one-hot', clean=True)