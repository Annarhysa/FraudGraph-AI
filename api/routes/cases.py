from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import services

router = APIRouter(prefix="/cases", tags=["cases"])


class CreateCaseRequest(BaseModel):
    transaction_id: str
    analyst: str = "analyst"
    priority: str = "MEDIUM"
    notes: str = ""


class UpdateCaseRequest(BaseModel):
    status: str | None = None
    notes: str | None = None
    priority: str | None = None


@router.get("")
def list_cases():
    from fraudgraph import cases as case_store

    return {"cases": case_store.list_cases()}


@router.post("")
def create_case(body: CreateCaseRequest):
    from fraudgraph import cases as case_store

    df = services.get_alerts_df()
    if body.transaction_id not in set(df["transaction_id"]):
        raise HTTPException(404, f"No transaction with id {body.transaction_id}")
    if body.priority not in case_store.PRIORITIES:
        raise HTTPException(400, f"priority must be one of {case_store.PRIORITIES}")
    case_id = case_store.create_case(body.transaction_id, body.analyst, body.priority, body.notes)
    return case_store.get_case(case_id)


@router.patch("/{case_id}")
def update_case(case_id: str, body: UpdateCaseRequest):
    from fraudgraph import cases as case_store

    if case_store.get_case(case_id) is None:
        raise HTTPException(404, f"No case with id {case_id}")
    if body.status and body.status not in case_store.STATUSES:
        raise HTTPException(400, f"status must be one of {case_store.STATUSES}")
    if body.priority and body.priority not in case_store.PRIORITIES:
        raise HTTPException(400, f"priority must be one of {case_store.PRIORITIES}")
    case_store.update_case(case_id, status=body.status, notes=body.notes, priority=body.priority)
    return case_store.get_case(case_id)
