"""Eval for diagnosing flow runs that crash due to concurrency lease renewal failure.

Based on real user issues:
- https://github.com/PrefectHQ/prefect/issues/19068
- https://github.com/PrefectHQ/prefect/issues/18839

When a flow run holds a concurrency slot, it must periodically renew the lease.
If renewal fails (network issues, API problems, timeout), Prefect crashes the run
to prevent over-allocation.
"""

import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

import httpx
import pytest
from prefect import flow
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.actions import GlobalConcurrencyLimitCreate
from prefect.client.schemas.objects import FlowRun
from prefect.concurrency.sync import concurrency
from pydantic_ai import Agent


@pytest.fixture
async def crashed_flow_run(prefect_client: PrefectClient) -> FlowRun:
    """Create a flow run that crashes due to lease renewal failure.

    Simulates a network failure during lease renewal - a common real-world
    scenario that causes flow runs to crash with:
    "Concurrency lease renewal failed - slots are no longer reserved"
    """
    limit_name = f"database-connections-{uuid4().hex[:8]}"
    await prefect_client.create_global_concurrency_limit(
        concurrency_limit=GlobalConcurrencyLimitCreate(name=limit_name, limit=1)
    )

    from prefect.concurrency import _leases

    original_renewal_loop = _leases._lease_renewal_loop

    async def failing_renewal_loop(lease_id, lease_duration):
        """Simulate network failure during lease renewal."""
        raise httpx.ConnectError("Simulated network failure during lease renewal")

    _leases._lease_renewal_loop = failing_renewal_loop

    try:

        @flow(name=f"db-sync-job-{uuid4().hex[:8]}")
        def db_sync_job():
            with concurrency(limit_name, occupy=1, strict=True):
                time.sleep(0.5)  # Give the failure time to propagate
            return "done"

        try:
            db_sync_job(return_state=True)
        except BaseException:
            # CancelledError is a BaseException, not Exception
            pass

    finally:
        _leases._lease_renewal_loop = original_renewal_loop

    runs = await prefect_client.read_flow_runs()
    return runs[0]


async def test_diagnoses_lease_renewal_failure(
    simple_agent: Agent,
    crashed_flow_run: FlowRun,
    evaluate_response: Callable[[str, str], Awaitable[None]],
) -> None:
    """Test agent identifies concurrency lease renewal failure as crash cause."""
    assert crashed_flow_run.state.type.value == "CRASHED", (
        f"Expected CRASHED but got {crashed_flow_run.state.type.value}"
    )

    prompt = f"My flow run '{crashed_flow_run.name}' crashed. What happened?"

    async with simple_agent:
        result = await simple_agent.run(prompt)

    await evaluate_response(
        "Does the agent identify that the crash was due to concurrency lease "
        "renewal failure? Should mention 'lease' or 'concurrency'.",
        result.output,
    )
