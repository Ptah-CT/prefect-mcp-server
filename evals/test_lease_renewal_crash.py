"""Eval for diagnosing flow runs that crash due to concurrency lease renewal failure.

Based on real user issues:
- https://github.com/PrefectHQ/prefect/issues/19068
- https://github.com/PrefectHQ/prefect/issues/18839

When a flow run holds a concurrency slot, it must periodically renew the lease.
If renewal fails (network issues, API problems, timeout), Prefect crashes the run
to prevent over-allocation. This is a common production issue that's hard to diagnose
without understanding Prefect's internal lease renewal mechanism.
"""

from collections.abc import Awaitable, Callable
from uuid import uuid4

import pytest
from prefect import flow, get_run_logger
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.actions import GlobalConcurrencyLimitCreate
from prefect.client.schemas.objects import FlowRun, GlobalConcurrencyLimit
from prefect.states import Crashed
from pydantic_ai import Agent

LEASE_RENEWAL_ERROR = (
    "Concurrency lease renewal failed - slots are no longer reserved. "
    "Terminating execution to prevent over-allocation."
)


@pytest.fixture
async def concurrency_limit(prefect_client: PrefectClient) -> GlobalConcurrencyLimit:
    """Create a global concurrency limit."""
    limit_name = f"database-connections-{uuid4().hex[:8]}"
    await prefect_client.create_global_concurrency_limit(
        concurrency_limit=GlobalConcurrencyLimitCreate(
            name=limit_name,
            limit=1,
        )
    )
    return await prefect_client.read_global_concurrency_limit_by_name(limit_name)


@pytest.fixture
async def crashed_flow_run(
    prefect_client: PrefectClient,
    concurrency_limit: GlobalConcurrencyLimit,
) -> FlowRun:
    """Create a flow run that crashed due to lease renewal failure.

    The flow logs show it acquired a concurrency slot before crashing,
    matching the real user experience from GitHub issues.
    """

    @flow(name=f"db-sync-job-{uuid4().hex[:8]}")
    def db_sync_job():
        logger = get_run_logger()
        logger.info(f"Acquired concurrency slot for '{concurrency_limit.name}'")
        logger.info("Starting database sync operation")
        logger.info("Processing batch 1 of 5")
        logger.info("Processing batch 2 of 5")
        return "completed"

    state = db_sync_job(return_state=True)
    flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)

    # Simulate the crash that occurs when lease renewal fails
    crashed_state = Crashed(message=LEASE_RENEWAL_ERROR)
    await prefect_client.set_flow_run_state(
        flow_run_id=flow_run.id,
        state=crashed_state,
        force=True,
    )

    return await prefect_client.read_flow_run(flow_run.id)


async def test_diagnoses_lease_renewal_failure(
    simple_agent: Agent,
    crashed_flow_run: FlowRun,
    evaluate_response: Callable[[str, str], Awaitable[None]],
) -> None:
    """Test agent identifies concurrency lease renewal failure as crash cause."""
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
