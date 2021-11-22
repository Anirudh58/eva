# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd

from src.planner.load_data_plan import LoadDataPlan
from src.executor.abstract_executor import AbstractExecutor
from src.storage.storage_engine import StorageEngine
from src.models.storage.batch import Batch
from src.configuration.configuration_manager import ConfigurationManager


class LoadCSVDataExecutor(AbstractExecutor):

    def __init__(self, node: LoadDataPlan):
        super().__init__(node)
        config = ConfigurationManager()
        self.path_prefix = config.get_value('storage', 'path_prefix')

    def validate(self):
        pass

    def exec(self):
        """
        Read the input CSV file and store in the database
        """

        # Get the path to the CSV file
        csv_file_path = os.path.join(self.path_prefix, self.node.file_path)

        # Read the CSV file
        meta_df = pd.read_csv(csv_file_path)
        meta_df_len = len(meta_df)

        # Create a batch
        batch = Batch(meta_df)

        # Store the batch
        StorageEngine.write(self.node.table_metainfo, batch)

        # Return the number of frames loaded
        df_yield = pd.DataFrame({
                'Meta': str(self.node.file_path),
                'Num_Rows': meta_df_len,
            })

        yield df_yield


