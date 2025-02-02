# coding=utf-8
# Copyright 2018-2022 EVA
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
import pandas as pd

from eva.catalog.catalog_manager import CatalogManager
from eva.executor.abstract_executor import AbstractExecutor
from eva.executor.executor_utils import ExecutorError
from eva.models.storage.batch import Batch
from eva.parser.table_ref import TableInfo
from eva.plan_nodes.drop_plan import DropPlan
from eva.storage.storage_engine import StorageEngine
from eva.utils.logging_manager import logger


class DropExecutor(AbstractExecutor):
    def __init__(self, node: DropPlan):
        super().__init__(node)

    def validate(self):
        pass

    def exec(self):
        """Drop table executor"""
        catalog_manager = CatalogManager()
        if len(self.node.table_infos) > 1:
            logger.exception("Drop supports only single table")
        table_info: TableInfo = self.node.table_infos[0]

        if not catalog_manager.check_table_exists(
            table_info.table_name, table_info.database_name
        ):
            err_msg = "Table: {} does not exist".format(table_info)
            if self.node.if_exists:
                logger.warn(err_msg)
                return Batch(pd.DataFrame([err_msg]))
            else:
                logger.exception(err_msg)
                raise ExecutorError(err_msg)

        try:
            table_obj = catalog_manager.get_table_catalog_entry(
                table_info.table_name, table_info.database_name
            )
            storage_engine = StorageEngine.factory(table_obj)
        except RuntimeError as err:
            raise ExecutorError(str(err))
        storage_engine.drop(table=table_obj)

        success = catalog_manager.delete_table_catalog_entry(table_obj)

        if not success:
            err_msg = "Failed to drop {}".format(table_info)
            logger.exception(err_msg)
            raise ExecutorError(err_msg)

        yield Batch(
            pd.DataFrame(
                {"Table Successfully dropped: {}".format(table_info.table_name)},
                index=[0],
            )
        )
