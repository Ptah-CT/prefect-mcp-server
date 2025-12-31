"""Eval for diagnosing flow runs that crash due to concurrency lease renewal failure.

Based on real user issues:
- https://github.com/PrefectHQ/prefect/issues/19068
- https://github.com/PrefectHQ/prefect/issues/18839

When a flow run holds a concurrency slot, it must periodically renew the lease.
If renewal fails (network issues, API problems, timeout), Prefect crashes the run
to prevent over-allocation. This is a common production issue that's hard to diagnose
without understanding Prefect's internal lease renewal mechanism.
"""

import time
from collections.abc import Awaitable, Callable
from unittest.mock import patch
from uuid import uuid4

import pytest
from httpx import Request, Response
from prefect import flow
from prefect._internal.concurrency.cancellation import CancelledError
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.actions import GlobalConcurrencyLimitCreate
from prefect.client.schemas.objects import FlowRun
from prefect.concurrency.sync import concurrency
from prefect.exceptions import PrefectHTTPStatusError
from pydantic_ai import Agent

# Minimum lease duration is 60s, renewal happens at 45s (0.75 * 60)
LEASE_DURATION = 60
WAIT_FOR_RENEWAL = 50  # Wait past the 45s renewal point


@pytest.fixture
async def crashed_flow_run(prefect_client: PrefectClient) -> FlowRun:
    """Create a flow run that crashes due to lease renewal failure.

    Actually triggers a real lease renewal failure by:
    1. Creating a concurrency limit
    2. Running a flow that acquires a slot with strict=True
    3. Patching the renewal to fail on second attempt
    4. Waiting for the renewal to fail and crash the flow
    """
    limit_name = f"database-connections-{uuid4().hex[:8]}"
    await prefect_client.create_global_concurrency_limit(
        concurrency_limit=GlobalConcurrencyLimitCreate(
            name=limit_name,
            limit=1,
            slot_decay_per_second=0,
        )
    )

    # Track renewal attempts
    renewal_count = [0]
    flow_run_id = [None]

    @flow(name=f"db-sync-job-{uuid4().hex[:8]}")
    def db_sync_job():
        from prefect.context import get_run_context

        ctx = get_run_context()
        flow_run_id[0] = ctx.flow_run.id

        with concurrency(
            limit_name,
            occupy=1,
            lease_duration=LEASE_DURATION,
            strict=True,  # This makes it crash on lease renewal failure
        ):
            # Patch renewal to fail on second attempt (first renewal is immediate)
            original_renew = type(get_client()).renew_concurrency_lease

            async def failing_renew(self, *args, **kwargs):
                renewal_count[0] += 1
                if renewal_count[0] > 1:
                    request = Request("POST", "http://test/renew")
                    response = Response(404, json={"detail": "Lease not found"})
                    raise PrefectHTTPStatusError.from_httpx_error(
                        __import__("httpx").HTTPStatusError(
                            "Not found", request=request, response=response
                        )
                    )
                return await original_renew(self, *args, **kwargs)

            with patch.object(
                type(get_client()), "renew_concurrency_lease", failing_renew
            ):
                # Wait for renewal to be attempted at t=45s
                time.sleep(WAIT_FOR_RENEWAL)

        return "done"

    # Run the flow - it will crash due to lease renewal failure
    try:
        db_sync_job(return_state=True)
    except (CancelledError, Exception):
        pass  # Expected - the flow crashes

    if flow_run_id[0] is None:
        pytest.fail("Flow run ID not captured")

    return await prefect_client.read_flow_run(flow_run_id[0])


@pytest.mark.timeout(120)  # Allow time for the 50s wait
async def test_diagnoses_lease_renewal_failure(
    simple_agent: Agent,
    crashed_flow_run: FlowRun,
    evaluate_response: Callable[[str, str], Awaitable[None]],
) -> None:
    """Test agent identifies concurrency lease renewal failure as crash cause."""
    # Verify the flow actually crashed
    assert crashed_flow_run.state.type.value == "CRASHED", (
        f"Expected CRASHED state but got {crashed_flow_run.state.type.value}"
    )

    prompt = (
        f"My flow run '{crashed_flow_run.name}' crashed unexpectedly. What happened?"
    )

    async with simple_agent:
        result = await simple_agent.run(prompt)

    await evaluate_response(
        "Does the agent correctly identify that the flow run crashed due to "
        "concurrency lease renewal failure? The response should mention "
        "'lease renewal' or 'concurrency slot' and explain that the run was "
        "terminated because the lease could not be renewed.",
        result.output,
    )
