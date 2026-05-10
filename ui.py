"""
CloudDash Support — Streamlit Web UI (Bonus)
Run: streamlit run ui.py
Requires the API to be running on localhost:8000
"""
import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CloudDash Support",
    page_icon="☁️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
.agent-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 4px;
}
.badge-triage { background: #E8F4FD; color: #1565C0; }
.badge-technical_support { background: #E8F5E9; color: #2E7D32; }
.badge-billing { background: #FFF3E0; color: #E65100; }
.badge-escalation { background: #FCE4EC; color: #B71C1C; }

.citation-box {
    background: #F8F9FA;
    border-left: 3px solid #4285F4;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
    margin-top: 6px;
}
.ticket-box {
    background: #FFF3E0;
    border: 1px solid #FF9800;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 8px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trace_id" not in st.session_state:
    st.session_state.trace_id = None

# ── Header ─────────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])
with col1:
    st.title("☁️ CloudDash Support")
    st.caption("AI-powered multi-agent customer support")
with col2:
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.trace_id = None
        st.rerun()

# ── Health check ───────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

if not check_api_health():
    st.error("⚠️ API is not reachable. Make sure the server is running on " + API_BASE)
    st.code("python main.py", language="bash")
    st.stop()

# ── Start conversation if needed ───────────────────────────────────────────

def start_conversation():
    r = requests.post(f"{API_BASE}/conversations", json={}, timeout=30)
    r.raise_for_status()
    data = r.json()
    st.session_state.conversation_id = data["conversation_id"]
    st.session_state.trace_id = data["trace_id"]
    st.session_state.messages.append({
        "role": "assistant",
        "content": data["message"],
        "agent": data["agent"],
        "citations": [],
        "ticket_id": None,
        "handover": False,
    })

if st.session_state.conversation_id is None:
    start_conversation()

# ── Render chat history ────────────────────────────────────────────────────

AGENT_LABELS = {
    "triage": ("🎯 Triage Agent", "badge-triage"),
    "technical_support": ("🔧 Technical Support", "badge-technical_support"),
    "billing": ("💳 Billing Agent", "badge-billing"),
    "escalation": ("🚨 Escalation Agent", "badge-escalation"),
}

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            agent_key = msg.get("agent", "triage")
            label, badge_class = AGENT_LABELS.get(agent_key, ("🤖 Agent", "badge-triage"))
            st.markdown(
                f'<span class="agent-badge {badge_class}">{label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(msg["content"])

            # Citations
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 {len(citations)} KB article(s) referenced"):
                    for c in citations:
                        st.markdown(
                            f'<div class="citation-box">📄 <strong>[{c["article_id"]}]</strong> '
                            f'{c["title"]} <em>({c["category"]})</em> — '
                            f'Relevance: {c["relevance_score"]:.0%}</div>',
                            unsafe_allow_html=True,
                        )

            # Ticket
            if msg.get("ticket_id"):
                st.markdown(
                    f'<div class="ticket-box">🎫 <strong>Support Ticket Created:</strong> '
                    f'{msg["ticket_id"]} — A human specialist will follow up via email.</div>',
                    unsafe_allow_html=True,
                )

            if msg.get("handover"):
                st.info("🔄 Conversation was handed over to the above agent.")

# ── Chat input ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Describe your CloudDash issue..."):
    # Add user message to UI immediately
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "agent": None,
        "citations": [],
        "ticket_id": None,
        "handover": False,
    })

    # Send to API
    with st.spinner("CloudDash Support is processing your message..."):
        try:
            r = requests.post(
                f"{API_BASE}/conversations/{st.session_state.conversation_id}/messages",
                json={"message": prompt},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["message"],
                "agent": data["agent"],
                "citations": data.get("citations", []),
                "ticket_id": data.get("ticket_id"),
                "handover": data.get("handover_occurred", False),
            })
        except requests.exceptions.Timeout:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⚠️ The request timed out. Please try again.",
                "agent": "triage",
                "citations": [],
                "ticket_id": None,
                "handover": False,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}. Please ensure the API is running.",
                "agent": "triage",
                "citations": [],
                "ticket_id": None,
                "handover": False,
            })

    st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Session Info")
    st.code(f"ID: {st.session_state.conversation_id or 'N/A'}", language=None)
    if st.session_state.trace_id:
        st.code(f"Trace: {st.session_state.trace_id}", language=None)

    st.divider()
    st.subheader("🧪 Test Scenarios")
    scenarios = {
        "🔔 Alerts not firing (AWS)": "My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday. I'm on the Pro plan.",
        "📋 Upgrade + SSO issue": "I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue I reported last week has been resolved?",
        "💳 Double charge + refund": "I've been charged twice for April. I need an immediate refund and I want to speak to a manager.",
        "❓ Unknown integration": "Does CloudDash support integration with Datadog for cross-platform alerting?",
        "🔑 Reset API key": "How do I reset my API key?",
    }
    for label, scenario_text in scenarios.items():
        if st.button(label, use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": scenario_text,
                "agent": None,
                "citations": [],
                "ticket_id": None,
                "handover": False,
            })
            with st.spinner("Processing..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/conversations/{st.session_state.conversation_id}/messages",
                        json={"message": scenario_text},
                        timeout=60,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["message"],
                        "agent": data["agent"],
                        "citations": data.get("citations", []),
                        "ticket_id": data.get("ticket_id"),
                        "handover": data.get("handover_occurred", False),
                    })
                except Exception as e:
                    st.error(str(e))
            st.rerun()

    st.divider()
    st.caption("CloudDash Support v1.0 • Multi-Agent System")
