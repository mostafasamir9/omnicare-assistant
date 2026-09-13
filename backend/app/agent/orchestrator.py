import json
from typing import Any

from google.genai import types

from backend.app.agent.prompts import SYSTEM_PROMPT
from backend.app.agent.router import deterministic_route
from backend.app.guardrails import (
    detect_injection,
    sanitize_user_input,
    wrap_untrusted,
)
from backend.app.llm.client import llm
from backend.app.rag.retriever import retrieve_with_confidence
from backend.app.schemas import (
    ChatResponse,
    Citation,
    ToolCallRecord,
)
from backend.app.session import session_store
from backend.app.tools.claims import (
    extract_claim_id,
    get_claim_status,
    submit_claim,
)


# ------------------------------------------------------------------
# Tool declarations exposed to Gemini
# ------------------------------------------------------------------
def _get_claim_status_decl() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="get_claim_status",
        description="Look up the status of an existing claim by claim ID.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "claim_id": types.Schema(
                    type=types.Type.STRING,
                    description="Claim ID, e.g. CLM-1001",
                ),
            },
            required=["claim_id"],
        ),
    )


def _preview_claim_decl() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name="preview_claim",
        description=(
            "Prepare a new insurance claim for user confirmation. "
            "Call this BEFORE any submission. Returns a preview the user must confirm."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "policy_id": types.Schema(
                    type=types.Type.STRING,
                    description="Policy ID, e.g. POL-001",
                ),
                "claim_type": types.Schema(
                    type=types.Type.STRING,
                    enum=["auto", "home", "life", "health"],
                ),
                "incident_date": types.Schema(
                    type=types.Type.STRING,
                    description="Date of incident in YYYY-MM-DD format",
                ),
                "description": types.Schema(
                    type=types.Type.STRING,
                    description="Short description of what happened",
                ),
                "amount_estimate": types.Schema(
                    type=types.Type.NUMBER,
                    description="Estimated claim amount in USD",
                ),
            },
            required=[
                "policy_id",
                "claim_type",
                "incident_date",
                "description",
                "amount_estimate",
            ],
        ),
    )


def _build_tools() -> list[types.Tool]:
    return [
        types.Tool(
            function_declarations=[
                _get_claim_status_decl(),
                _preview_claim_decl(),
            ]
        )
    ]


# ------------------------------------------------------------------
# Context block for RAG
# ------------------------------------------------------------------
def _build_context_block(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations:
        lines.append(f"[{c.doc_id} | {c.section} | p.{c.page}]\n{c.snippet}")
    return wrap_untrusted("\n\n".join(lines))


# ------------------------------------------------------------------
# History -> Gemini contents
# ------------------------------------------------------------------
def _history_to_contents(session_id: str) -> list[types.Content]:
    contents: list[types.Content] = []
    for turn in session_store.history(session_id)[-10:]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=turn["content"])],
            )
        )
    return contents


# ------------------------------------------------------------------
# Deterministic fast path (no LLM call)
# ------------------------------------------------------------------
def _fast_path_status(
    session_id: str, user_id: str, message: str
) -> ChatResponse | None:
    route = deterministic_route(message)
    if route != "status":
        return None

    claim_id = extract_claim_id(message)
    if not claim_id:
        return None

    if not session_store.add_tool_call(session_id):
        return ChatResponse(
            session_id=session_id,
            reply="Tool call limit reached for this session.",
            refused=True,
        )

    result = get_claim_status(claim_id, user_id)

    record = ToolCallRecord(
        name="get_claim_status",
        args={"claim_id": claim_id},
        result=result.model_dump() if result else None,
        ok=result is not None,
    )

    if result:
        reply = (
            f"Claim {result.claim_id} is currently **{result.status}**.\n"
            f"- Filed: {result.filed_date}\n"
            f"- Last update: {result.last_update}\n"
            f"- Adjuster: {result.adjuster}\n"
            f"- Notes: {result.notes}"
        )
    else:
        reply = f"I couldn't find a claim with ID {claim_id}. Please double-check the ID."

    session_store.append(session_id, "assistant", reply)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        tool_calls=[record],
    )


# ------------------------------------------------------------------
# Main chat handler
# ------------------------------------------------------------------
def handle_chat(session_id: str, user_id: str, message: str) -> ChatResponse:
    # 1. Injection guard
    if detect_injection(message):
        return ChatResponse(
            session_id=session_id,
            reply=(
                "I can't help with that request. If you have a question about "
                "your policy or claim, I'm happy to assist."
            ),
            refused=True,
        )

    # 2. Sanitize
    clean = sanitize_user_input(message)
    session_store.append(session_id, "user", clean)

    if not session_store.add_tokens(session_id, max(1, len(clean) // 4)):
        return ChatResponse(
            session_id=session_id,
            reply="Session token budget exceeded. Please start a new session.",
            refused=True,
        )

    # 3. Fast path: claim status by ID
    fast = _fast_path_status(session_id, user_id, clean)
    if fast is not None:
        return fast

    # 4. RAG retrieval
    citations, confidence = retrieve_with_confidence(clean)
    context_block = _build_context_block(citations)

    # 5. Build contents for Gemini
    contents = _history_to_contents(session_id)

    if context_block:
        contents.insert(
            max(0, len(contents) - 1),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"<policy_context>\n{context_block}\n</policy_context>"
                    )
                ],
            ),
        )

    # 6. Call the model with tools
    tools = _build_tools()
    response = llm.generate(
        contents=contents,
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        temperature=0.2,
    )

    tool_records: list[ToolCallRecord] = []
    function_calls = llm.function_calls_from_response(response)

    # 7. Handle function calls
    if function_calls:
        model_content = response.candidates[0].content
        contents.append(model_content)

        for fc in function_calls:
            name = fc.name
            try:
                args = dict(fc.args) if fc.args else {}
            except Exception:
                try:
                    args = json.loads(type(fc.args).to_json(fc.args)) if fc.args else {}
                except Exception:
                    args = {}

            if name == "get_claim_status":
                if not session_store.add_tool_call(session_id):
                    return ChatResponse(
                        session_id=session_id,
                        reply="Tool call limit reached.",
                        refused=True,
                    )
                res = get_claim_status(args.get("claim_id", ""), user_id)
                tool_records.append(
                    ToolCallRecord(
                        name=name,
                        args=args,
                        result=res.model_dump() if res else None,
                        ok=res is not None,
                    )
                )
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response={
                                    "result": res.model_dump()
                                    if res
                                    else {"error": "not found"}
                                },
                            )
                        ],
                    )
                )

            elif name == "preview_claim":
                action = {
                    "type": "submit_claim",
                    "user_id": user_id,
                    "payload": args,
                }
                action_id = session_store.stash_pending(session_id, action)

                reply = (
                    "Here's a summary of the claim I'm about to submit:\n\n"
                    f"- **Policy**: {args.get('policy_id')}\n"
                    f"- **Type**: {args.get('claim_type')}\n"
                    f"- **Incident date**: {args.get('incident_date')}\n"
                    f"- **Estimated amount**: ${float(args.get('amount_estimate', 0)):,.2f}\n"
                    f"- **Description**: {args.get('description')}\n\n"
                    "Please confirm to submit, or tell me what to change."
                )

                session_store.append(session_id, "assistant", reply)

                return ChatResponse(
                    session_id=session_id,
                    reply=reply,
                    tool_calls=tool_records,
                    pending_action={**action, "action_id": action_id},
                )

        # Follow-up call after tool responses
        followup = llm.generate(
            contents=contents,
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
            temperature=0.2,
        )
        reply = llm.text_from_response(followup) or "I'm not sure how to help with that."
        session_store.append(session_id, "assistant", reply)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            tool_calls=tool_records,
        )

    # 8. Plain answer
    reply = llm.text_from_response(response) or "I'm not sure how to help with that."
    if citations:
        reply += "\n\n_Sources: " + ", ".join(
            f"{c.doc_id} §{c.section} (p.{c.page})" for c in citations
        ) + "_"

    session_store.append(session_id, "assistant", reply)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        citations=citations,
        tool_calls=tool_records,
    )


# ------------------------------------------------------------------
# Confirmation handler
# ------------------------------------------------------------------
def handle_confirm(
    session_id: str, user_id: str, action_id: str, confirm: bool
) -> ChatResponse:
    pending = session_store.pop_pending(action_id)

    if not pending:
        return ChatResponse(
            session_id=session_id,
            reply="That confirmation has expired or is invalid.",
            refused=True,
        )

    if pending.get("session_id") != session_id or pending.get("user_id") != user_id:
        return ChatResponse(
            session_id=session_id,
            reply="Confirmation does not match this session.",
            refused=True,
        )

    if not confirm:
        return ChatResponse(
            session_id=session_id,
            reply=(
                "No problem — I won't submit the claim. "
                "Tell me what you'd like to change."
            ),
        )

    if not session_store.add_tool_call(session_id):
        return ChatResponse(
            session_id=session_id,
            reply="Tool call limit reached.",
            refused=True,
        )

    payload = pending["payload"]
    result = submit_claim(
        policy_id=payload["policy_id"],
        user_id=user_id,
        claim_type=payload["claim_type"],
        incident_date=payload["incident_date"],
        description=payload["description"],
        amount_estimate=float(payload["amount_estimate"]),
    )

    record = ToolCallRecord(
        name="submit_claim",
        args=payload,
        result=result.model_dump(),
        ok=result.status != "rejected",
    )

    if result.status != "rejected":
        reply = (
            f"✅ {result.message}\n\n"
            f"Claim ID: **{result.claim_id}**\n"
            "You'll receive a confirmation email shortly."
        )
    else:
        reply = f"❌ {result.message}"

    session_store.append(session_id, "assistant", reply)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        tool_calls=[record],
    )