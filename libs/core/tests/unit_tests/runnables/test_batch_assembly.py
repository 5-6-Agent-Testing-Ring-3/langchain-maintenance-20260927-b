"""Batch assembly preserves the order of successes and exceptions."""

import asyncio
from typing import cast

import pytest

from langchain_core.runnables import RunnableLambda


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("size", [0, 1, 32])
@pytest.mark.parametrize("failure_stride", [0, 1, 3])
async def test_sequence_batch_output_order(
    *, use_async: bool, size: int, failure_stride: int
) -> None:
    inputs = list(range(size))
    errors = {
        value: ValueError(str(value))
        for value in inputs
        if failure_stride and value % failure_stride == 0
    }

    def process(value: int) -> int:
        if value in errors:
            raise errors[value]
        return value

    runnable = RunnableLambda(process) | RunnableLambda(lambda value: value * 2)
    if use_async:
        outputs = await runnable.abatch(inputs, return_exceptions=True)
    else:
        outputs = await asyncio.to_thread(
            runnable.batch, inputs, return_exceptions=True
        )

    assert outputs == [errors.get(value, value * 2) for value in inputs]
    assert inputs == list(range(size))


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("size", [0, 1, 32])
async def test_retry_batch_exhausted_output_order(
    *, use_async: bool, size: int
) -> None:
    inputs = list(range(size))
    errors = [ValueError(str(value)) for value in inputs]
    attempts = [0] * size

    def fail(value: int) -> int:
        attempts[value] += 1
        raise errors[value]

    runnable = RunnableLambda(fail).with_retry(
        stop_after_attempt=2, wait_exponential_jitter=False
    )
    if use_async:
        outputs = await runnable.abatch(inputs, return_exceptions=True)
    else:
        outputs = await asyncio.to_thread(
            runnable.batch, inputs, return_exceptions=True
        )

    assert cast("list[int | Exception]", outputs) == errors
    assert attempts == [2] * size
    assert inputs == list(range(size))
