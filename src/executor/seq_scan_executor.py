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
from typing import Iterator
import pandas as pd

from src.models.storage.batch import Batch
from src.executor.abstract_executor import AbstractExecutor
from src.planner.seq_scan_plan import SeqScanPlan
from src.expression.function_expression import FunctionExpression

class SequentialScanExecutor(AbstractExecutor):
    """
    Applies predicates to filter the frames which satisfy the condition
    Arguments:
        node (AbstractPlan): The SequentialScanPlan

    """

    def __init__(self, node: SeqScanPlan):
        super().__init__(node)
        self.predicate = node.predicate
        self.project_expr = node.columns

    def validate(self):
        pass

    def exec(self) -> Iterator[Batch]:

        child_executor = self.children[0]
        for batch in child_executor.exec():
            # We do the predicate first
            if self.predicate is not None:
                outcomes = self.predicate.evaluate(batch)
                required_frame_ids = []
                for i, outcome in enumerate(outcomes):
                    if outcome:
                        required_frame_ids.append(i)

                batch = batch[required_frame_ids]

            # Then do project
            if len(self.project_expr) > 0:
                new_batch = self.project_expr[0].evaluate(batch)
            for expr in self.project_expr[1:]:
                temp_batch = expr.evaluate(batch)
                new_batch = new_batch.merge_column_wise(temp_batch)

            yield new_batch
