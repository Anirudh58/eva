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
import unittest

from mock import patch, MagicMock

from src.optimizer.optimizer_utils import bind_dataset


class OptimizerUtilsTest(unittest.TestCase):

    @patch('src.optimizer.optimizer_utils.CatalogManager')
    def test_bind_dataset(self, mock):
        video = MagicMock()
        catalog = mock.return_value
        actual = bind_dataset(video)
        catalog.get_dataset_metadata.assert_called_with(video.database_name,
                                                        video.table_name)
        self.assertEqual(actual, catalog.get_dataset_metadata.return_value)
