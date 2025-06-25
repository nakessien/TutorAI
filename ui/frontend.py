import streamlit as st
import requests
import json
import time
import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import hashlib

# Page configuration
st.set_page_config(
    page_title="TutorAI - Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://127.0.0.1:8000"
RESPONSE_STYLES = {
    "balanced": "🔄 Balanced",
    "detailed_policy": "📋 Detailed Policy",
    "practical_guide": "📝 Practical Guide"
}

ADMIN_PASSWORD = "admin123"  # 在实际部署中应该从环境变量读取


# Custom CSS
def load_custom_css():
    """Load custom CSS for better UI"""
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }

    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }

    .ai-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
    }

    /* Version switcher */
    .version-switcher {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin: 10px 0;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 8px;
    }

    /* Status indicators */
    .status-online {
        color: #4caf50;
        font-weight: bold;
    }

    .status-offline {
        color: #f44336;
        font-weight: bold;
    }

    /* Admin panel styling */
    .admin-panel {
        border: 2px solid #ff9800;
        border-radius: 8px;
        padding: 1rem;
        background-color: #fff3e0;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .chat-message {
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# Session state initialization
def init_session_state():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'admin_mode' not in st.session_state:
        st.session_state.admin_mode = False

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"

    if 'api_online' not in st.session_state:
        st.session_state.api_online = False


# API Functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_online = True
            return True
        else:
            st.session_state.api_online = False
            return False
    except:
        st.session_state.api_online = False
        return False


def ask_question(question: str, user_id: str, style: Optional[str] = None):
    """Send question to API"""
    try:
        payload = {
            "question": question,
            "user_id": user_id
        }
        if style:
            payload["style"] = style

        with st.spinner("🤔 Thinking... (this may take a minute for complex questions)"):
            response = requests.post(
                f"{API_BASE_URL}/api/ask",
                json=payload,
                timeout=180  # 增加到 180 秒
            )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ReadTimeout:
        st.error("⏱️ Request timed out. The model is taking longer than expected. Please try a simpler question or wait a moment and try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def regenerate_answer(question: str, user_id: str, session_id: str, current_styles: List[str]):
    """Regenerate answer with different style"""
    try:
        payload = {
            "question": question,
            "user_id": user_id,
            "session_id": session_id,
            "current_styles": current_styles
        }

        with st.spinner("🔄 Generating alternative response... (this may take a moment)"):
            response = requests.post(
                f"{API_BASE_URL}/api/regenerate",
                json=payload,
                timeout=180  # 增加到 180 秒
            )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Regeneration failed: {response.status_code}")
            return None

    except requests.exceptions.ReadTimeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def record_preference(user_id: str, session_id: str, question: str,
                      chosen_style: str, generation_sequence: List[str],
                      interaction_time: float):
    """Record user preference"""
    try:
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "question": question,
            "chosen_style": chosen_style,
            "generation_sequence": generation_sequence,
            "interaction_time": interaction_time
        }

        response = requests.post(
            f"{API_BASE_URL}/api/preference",
            json=payload,
            timeout=10
        )

        return response.status_code == 200

    except:
        return False


def get_system_status():
    """Get system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_admin_statistics():
    """Get admin statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/admin/statistics", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_user_data(user_id: str):
    """Get user data"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/admin/user/{user_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# UI Components
def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.title("🎓 TutorAI")

        # API Status
        if st.session_state.api_online:
            st.markdown('<p class="status-online">🟢 API Online</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-offline">🔴 API Offline</p>', unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        if st.button("💬 Chat", use_container_width=True,
                     type="primary" if st.session_state.current_page == "chat" else "secondary"):
            st.session_state.current_page = "chat"
            st.rerun()

        if st.session_state.admin_mode:
            if st.button("🛠️ Admin Panel", use_container_width=True,
                         type="primary" if st.session_state.current_page == "admin" else "secondary"):
                st.session_state.current_page = "admin"
                st.rerun()

        st.markdown("---")

        # Admin access (hidden button)
        if not st.session_state.admin_mode:
            if st.button("🔐", help="Admin Access", key="hidden_admin"):
                show_admin_login()
        else:
            if st.button("🚪 Exit Admin", use_container_width=True):
                st.session_state.admin_mode = False
                st.session_state.current_page = "chat"
                st.rerun()

        # User info
        st.markdown("---")
        st.caption(f"User ID: {st.session_state.user_id[:8]}...")

        if st.button("🔄 Refresh", use_container_width=True):
            check_api_health()
            st.rerun()


def show_admin_login():
    """Show admin login modal"""
    password = st.text_input("Admin Password:", type="password", key="admin_password")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", use_container_width=True):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_mode = True
                st.session_state.current_page = "admin"
                st.success("Admin access granted!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid password!")

    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def render_chat_page():
    """Render main chat interface"""
    st.title("💬 Academic Policy Assistant")
    st.markdown("Ask me anything about academic policies and procedures!")

    # Check API health
    if not st.session_state.api_online:
        check_api_health()

    if not st.session_state.api_online:
        st.error("🔴 API is offline. Please check the backend service.")
        return

    # Sample questions
    with st.expander("💡 Sample Questions"):
        sample_questions = [
            "What are the requirements for transferring majors?",
            "How do I apply for academic appeal?",
            "What is the deadline for course registration?",
            "What documents do I need for graduation?",
            "How do I request an official transcript?"
        ]

        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"📝 {question}", key=f"sample_{i}", use_container_width=True):
                    handle_user_question(question)

    # Question input
    st.markdown("### Ask Your Question")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Type your question here:",
            height=100,
            placeholder="Enter your academic policy question...",
            key="question_input"
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            submitted = st.form_submit_button("🚀 Ask", type="primary", use_container_width=True)

        with col2:
            style_preference = st.selectbox(
                "Response Style (optional):",
                options=[None] + list(RESPONSE_STYLES.keys()),
                format_func=lambda x: "Auto-detect preferred style" if x is None else RESPONSE_STYLES[x],
                key="style_preference"
            )

    if submitted and question.strip():
        handle_user_question(question.strip(), style_preference)
    elif submitted:
        st.warning("Please enter a question first.")

    # Display conversation history
    render_conversation_history()


def handle_user_question(question: str, style: Optional[str] = None):
    """Handle user question submission"""
    start_time = time.time()

    # Call API
    result = ask_question(question, st.session_state.user_id, style)

    if result:
        # Create conversation item
        conversation_item = {
            "question": question,
            "session_id": result["session_id"],
            "responses": [{
                "answer": result["answer"],
                "style": result["style"],
                "generation_time": result["generation_time"],
                "metadata": result["metadata"]
            }],
            "current_response_index": 0,
            "generation_sequence": [result["style"]],
            "context_used": result["context_used"],
            "sources_count": result["sources_count"],
            "timestamp": datetime.now(),
            "interaction_start_time": start_time
        }

        # Add to conversation history
        st.session_state.conversation_history.append(conversation_item)
        st.rerun()


def render_conversation_history():
    """Render conversation history"""
    if not st.session_state.conversation_history:
        st.info("👋 Start a conversation by asking a question above!")
        return

    st.markdown("### 💬 Conversation History")

    # Reverse order to show newest first
    for i, item in enumerate(reversed(st.session_state.conversation_history)):
        real_index = len(st.session_state.conversation_history) - 1 - i
        render_conversation_item(item, real_index)


def render_conversation_item(item: Dict[str, Any], index: int):
    """Render a single conversation item"""
    with st.container():
        st.markdown("---")

        # Question
        st.markdown("**🤔 Question:**")
        st.markdown(f'<div class="chat-message user-message">{item["question"]}</div>',
                    unsafe_allow_html=True)

        # Current response
        current_response = item["responses"][item["current_response_index"]]
        current_style = current_response["style"]

        # Response header with style indicator
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**🤖 Answer ({RESPONSE_STYLES.get(current_style, current_style)}):**")

        with col2:
            # Version switcher if multiple responses
            if len(item["responses"]) > 1:
                render_version_switcher(item, index)

        # Response content
        st.markdown(f'<div class="chat-message ai-message">{current_response["answer"]}</div>',
                    unsafe_allow_html=True)

        # Action buttons
        render_action_buttons(item, index)

        # Metadata
        with st.expander("ℹ️ Response Details"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Generation Time", f"{current_response['generation_time']:.2f}s")

            with col2:
                st.metric("Context Used", "Yes" if item["context_used"] else "No")

            with col3:
                st.metric("Sources", item["sources_count"])

            with col4:
                st.metric("Versions", len(item["responses"]))


def render_version_switcher(item: Dict[str, Any], index: int):
    """Render version switcher for multiple responses"""
    total_versions = len(item["responses"])
    current_index = item["current_response_index"]

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("←", key=f"prev_{index}", disabled=(current_index == 0)):
            item["current_response_index"] = current_index - 1
            st.rerun()

    with col2:
        st.markdown(f'<div class="version-switcher">{current_index + 1}/{total_versions}</div>',
                    unsafe_allow_html=True)

    with col3:
        if st.button("→", key=f"next_{index}", disabled=(current_index == total_versions - 1)):
            item["current_response_index"] = current_index + 1
            st.rerun()


def render_action_buttons(item: Dict[str, Any], index: int):
    """Render action buttons for conversation item"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Regenerate button
        if st.button("🔄 Try Different Style", key=f"regen_{index}", use_container_width=True):
            handle_regeneration(item, index)

    with col2:
        # Helpful feedback
        if st.button("👍 This is helpful", key=f"helpful_{index}", use_container_width=True):
            handle_preference_feedback(item, index, True)

    with col3:
        # Copy answer
        if st.button("📋 Copy Answer", key=f"copy_{index}", use_container_width=True):
            current_response = item["responses"][item["current_response_index"]]
            st.code(current_response["answer"])

    with col4:
        # Export conversation
        if st.button("💾 Export", key=f"export_{index}", use_container_width=True):
            export_conversation_item(item)


def handle_regeneration(item: Dict[str, Any], index: int):
    """Handle answer regeneration"""
    result = regenerate_answer(
        question=item["question"],
        user_id=st.session_state.user_id,
        session_id=item["session_id"],
        current_styles=item["generation_sequence"]
    )

    if result:
        # Add new response
        new_response = {
            "answer": result["answer"],
            "style": result["style"],
            "generation_time": result["generation_time"],
            "metadata": result["metadata"]
        }

        item["responses"].append(new_response)
        item["current_response_index"] = len(item["responses"]) - 1
        item["generation_sequence"].append(result["style"])

        st.success(f"Generated new response in {RESPONSE_STYLES.get(result['style'], result['style'])} style!")
        st.rerun()


def handle_preference_feedback(item: Dict[str, Any], index: int, is_helpful: bool):
    """Handle preference feedback"""
    current_response = item["responses"][item["current_response_index"]]
    interaction_time = time.time() - item["interaction_start_time"]

    success = record_preference(
        user_id=st.session_state.user_id,
        session_id=item["session_id"],
        question=item["question"],
        chosen_style=current_response["style"],
        generation_sequence=item["generation_sequence"],
        interaction_time=interaction_time
    )

    if success:
        st.success("Thank you for your feedback! 🎯")
    else:
        st.warning("Could not record preference.")


def export_conversation_item(item: Dict[str, Any]):
    """Export conversation item"""
    current_response = item["responses"][item["current_response_index"]]

    export_data = {
        "question": item["question"],
        "answer": current_response["answer"],
        "style": current_response["style"],
        "timestamp": item["timestamp"].isoformat(),
        "metadata": {
            "generation_time": current_response["generation_time"],
            "context_used": item["context_used"],
            "sources_count": item["sources_count"],
            "total_versions": len(item["responses"])
        }
    }

    st.download_button(
        label="📁 Download as JSON",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name=f"tutor_ai_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def render_admin_page():
    """Render admin panel"""
    st.title("🛠️ Admin Panel")

    if not st.session_state.admin_mode:
        st.error("Access denied. Please log in as admin.")
        return

    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 System Status", "📈 Statistics", "👥 User Management", "⚙️ Settings"])

    with tab1:
        render_system_status()

    with tab2:
        render_statistics()

    with tab3:
        render_user_management()

    with tab4:
        render_settings()


def render_system_status():
    """Render system status"""
    st.markdown("### System Status")

    status = get_system_status()

    if status:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Status", status["status"])

        with col2:
            st.metric("Model Status", status["model_status"])

        with col3:
            st.metric("Total Chunks", status["total_chunks"])

        with col4:
            st.metric("Uptime", f"{status['uptime']:.1f}s")

        # Performance stats
        if "performance_stats" in status:
            st.markdown("### Performance Statistics")
            perf_stats = status["performance_stats"]

            col1, col2, col3 = st.columns(3)

            with col1:
                if "total_requests" in perf_stats:
                    st.metric("Total Requests", perf_stats["total_requests"])

            with col2:
                if "session_stats" in perf_stats:
                    session_stats = perf_stats["session_stats"]
                    st.metric("Active Sessions", session_stats.get("active_sessions", 0))

            with col3:
                if "llm_stats" in perf_stats:
                    llm_stats = perf_stats["llm_stats"]
                    if "average_generation_time" in llm_stats:
                        st.metric("Avg Generation Time", f"{llm_stats['average_generation_time']:.2f}s")
    else:
        st.error("Could not retrieve system status")


def render_statistics():
    """Render statistics"""
    st.markdown("### System Statistics")

    stats = get_admin_statistics()

    if stats and "statistics" in stats:
        data = stats["statistics"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Users", data.get("total_users", 0))

        with col2:
            st.metric("Total Interactions", data.get("total_interactions", 0))

        with col3:
            st.metric("Users with Conflicts", data.get("users_with_conflicts", 0))

        with col4:
            avg_time = data.get("avg_interaction_time", 0)
            st.metric("Avg Interaction Time", f"{avg_time:.2f}s")

        # Style distribution
        if "style_distribution" in data:
            st.markdown("### Style Distribution")
            style_dist = data["style_distribution"]

            for style, info in style_dist.items():
                st.metric(
                    RESPONSE_STYLES.get(style, style),
                    f"{info['count']} ({info['percentage']:.1f}%)"
                )
    else:
        st.error("Could not retrieve statistics")


def render_user_management():
    """Render user management"""
    st.markdown("### User Management")

    # User lookup
    user_id_input = st.text_input("User ID to lookup:")

    if st.button("🔍 Lookup User") and user_id_input:
        user_data = get_user_data(user_id_input)

        if user_data and "user_data" in user_data:
            data = user_data["user_data"]

            st.markdown(f"#### User: {user_id_input}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Interactions", data.get("total_interactions", 0))

            with col2:
                avg_time = data.get("avg_interaction_time", 0)
                st.metric("Avg Interaction Time", f"{avg_time:.2f}s")

            with col3:
                first_interaction = data.get("first_interaction")
                if first_interaction:
                    st.metric("First Interaction", first_interaction[:10])

            # Style distribution
            if "style_distribution" in data:
                st.markdown("#### Style Preferences")
                style_dist = data["style_distribution"]

                for style, count in style_dist.items():
                    st.metric(RESPONSE_STYLES.get(style, style), count)

            # Preference profile
            if "preference_profile" in data and data["preference_profile"]:
                profile = data["preference_profile"]
                st.markdown("#### Preference Profile")
                st.json(profile)
        else:
            st.error("User not found or no data available")

    # Current session user info
    st.markdown("### Current Session User")
    if st.button("🔍 Show My Data"):
        user_data = get_user_data(st.session_state.user_id)
        if user_data:
            st.json(user_data)


def render_settings():
    """Render settings"""
    st.markdown("### System Settings")

    # Document management
    st.markdown("#### Document Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reload Documents", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/api/admin/reload-documents", timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Documents reloaded successfully!")
                    if "statistics" in result:
                        stats = result["statistics"]
                        st.info(f"Total chunks: {stats.get('total_chunks', 0)}")
                else:
                    st.error("Failed to reload documents")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if st.button("🗑️ Reset User Preferences", use_container_width=True):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/admin/reset-user/{st.session_state.user_id}",
                    timeout=10
                )
                if response.status_code == 200:
                    st.success("User preferences reset successfully!")
                else:
                    st.error("Failed to reset preferences")
            except Exception as e:
                st.error(f"Error: {e}")

    # System info
    st.markdown("#### System Information")
    st.info(f"Current User ID: {st.session_state.user_id}")
    st.info(f"API Base URL: {API_BASE_URL}")
    st.info(f"Admin Mode: {st.session_state.admin_mode}")


# Keyboard shortcuts handler
def handle_keyboard_shortcuts():
    """Handle keyboard shortcuts (admin access)"""
    # This is a simplified approach since Streamlit doesn't have native keyboard event handling
    # In a real implementation, you might use JavaScript components
    pass


# Main application
def main():
    """Main application"""
    # Load custom CSS
    load_custom_css()

    # Initialize session state
    init_session_state()

    # Check API health on startup
    if 'initial_health_check' not in st.session_state:
        check_api_health()
        st.session_state.initial_health_check = True

    # Handle keyboard shortcuts
    handle_keyboard_shortcuts()

    # Render sidebar
    render_sidebar()

    # Render main content based on current page
    if st.session_state.current_page == "chat":
        render_chat_page()
    elif st.session_state.current_page == "admin":
        render_admin_page()


if __name__ == "__main__":
    main()