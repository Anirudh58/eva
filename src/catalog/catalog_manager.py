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

from src.catalog.database import init_db
from src.catalog.df_schema import DataFrameSchema
from src.catalog.models.df_column import DataFrameColumn
from src.catalog.models.df_metadata import DataFrameMetadata
from src.configuration.configuration_manager import ConfigurationManager
from src.configuration.dictionary import CATALOG_DIR
from src.utils.logging_manager import LoggingLevel
from src.utils.logging_manager import LoggingManager


class CatalogManager(object):
    _instance = None
    _catalog = None
    _catalog_dictionary = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CatalogManager, cls).__new__(cls)

            cls._instance.bootstrap_catalog()

        return cls._instance

    def bootstrap_catalog(self):

        eva_dir = ConfigurationManager().get_value("core", "location")
        output_url = os.path.join(eva_dir, CATALOG_DIR)
        LoggingManager().log("Bootstrapping catalog" + str(output_url),
                             LoggingLevel.INFO)
        init_db()
        # # Construct output location
        # catalog_dir_url = os.path.join(eva_dir, "catalog")
        #
        # # Get filesystem path
        # catalog_os_path = urlparse(catalog_dir_url).path
        #
        # # Check if catalog exists
        # if os.path.exists(catalog_os_path):
        #     # Load catalog if it exists
        #     load_catalog_dataframes(catalog_dir_url,
        #     self._catalog_dictionary)
        # else:
        #     # Create catalog if it does not exist
        #     create_catalog_dataframes(
        #         catalog_dir_url, self._catalog_dictionary)

    def get_bindings(self, database_name, table_name=None, column_name=None):
        metadata_id = DataFrameMetadata.get_id_from_name(database_name)
        table_id = None
        column_id = None
        if column_name is not None:
            column_id = DataFrameColumn.get_id_from_metadata_id_and_name(
                metadata_id,
                column_name)
        return metadata_id, table_id, column_id

    def get_metadata(self, metadata_id, col_id_list=[]):
        metadata = DataFrameMetadata.get(metadata_id)
        if len(col_id_list) > 0:
            df_columns = DataFrameColumn.get_by_metadata_id_and_id_in(
                col_id_list,
                metadata_id)
            metadata.set_schema(
                DataFrameSchema(metadata.get_name(), df_columns))
        return metadata

    # def create_dataset(self, dataset_name: str):
    #
    #     dataset_catalog_entry = \
    #         self._catalog_dictionary.get(DATASET_DATAFRAME_NAME)
    #
    #     dataset_df = \
    #         load_dataframe(dataset_catalog_entry.get_dataframe_file_url())
    #
    #     dataset_df.show(10)
    #
    #     next_row_id = get_next_row_id(dataset_df, DATASET_DATAFRAME_NAME)
    #
    #     row_1 = [next_row_id, dataset_name]
    #     rows = [row_1]
    #
    #     append_rows(dataset_catalog_entry, rows)
