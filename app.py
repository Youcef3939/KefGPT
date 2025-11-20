import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="KefGPT",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

from rag_engine import RAGEngine
from quiz_engine import QuizEngine
from utils.session_manager import SessionManager

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "quiz_engine" not in st.session_state:
    st.session_state.quiz_engine = None
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()

def check_models():
    models_path = Path("models")
    llm_path = models_path / "llm_models"
    embedding_path = models_path / "embeddings"
    
    errors = []
    if not llm_path.exists() or not any(llm_path.glob("*.gguf")):
        errors.append("LLM model not found. please place your model in /models/llm_models/")
    if not embedding_path.exists():
        errors.append("embedding model will be downloaded on first use")
    
    return errors

def main():
    st.title("📚 KefGPT")
    st.caption("Lightweight, offline learning assistant with quiz generation")
    
    model_errors = check_models()
    if model_errors:
        for error in model_errors:
            st.warning(error)
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        data_path = Path("data/pdfs")
        if data_path.exists():
            courses = [d.name for d in data_path.iterdir() if d.is_dir()]
            if courses:
                selected_course = st.selectbox("Select Course", courses)
                st.session_state.selected_course = selected_course
            else:
                st.warning("no courses found. Create folders in data/pdfs/")
                selected_course = None
        else:
            st.warning("data/pdfs/ directory not found")
            selected_course = None
        
        st.divider()
        
        with st.expander("➕ Add New Course"):
            new_course_name = st.text_input("New Course Name")
            if st.button("Create Course"):
                if new_course_name:
                    new_course_path = data_path / new_course_name
                    if not new_course_path.exists():
                        new_course_path.mkdir(parents=True)
                        st.success(f"Created course: {new_course_name}")
                        st.rerun()
                    else:
                        st.error("Course already exists")
                else:
                    st.error("Please enter a course name")

        if selected_course:
            with st.expander("📤 Upload PDFs"):
                uploaded_files = st.file_uploader(
                    "Upload PDF files", 
                    type="pdf", 
                    accept_multiple_files=True
                )
                if uploaded_files:
                    course_dir = data_path / selected_course
                    for uploaded_file in uploaded_files:
                        save_path = course_dir / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    st.success(f"Uploaded {len(uploaded_files)} files to {selected_course}")
        
        st.divider()
        
        use_gpu = st.checkbox("🚀 Enable GPU Acceleration", value=True, help="Check this if you have a GPU to speed up responses")
        
        if st.button("🔄 Initialize Engines", type="primary"):
            if selected_course:
                with st.spinner("Initializing RAG engine..."):
                    try:
                        with st.spinner("Loading models and initializing..."):
                            n_gpu = -1 if use_gpu else 0
                            st.session_state.rag_engine = RAGEngine(
                                course_name=selected_course, 
                                auto_ingest=True,
                                n_gpu_layers=n_gpu
                            )
                            st.session_state.quiz_engine = QuizEngine(st.session_state.rag_engine)
                        st.success(f"Engines initialized for course: {selected_course}!")
                    except Exception as e:
                        st.error(f"Error initializing: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.error("Please select a course first")
        
        st.divider()
        
        if st.button("📜 View History"):
            st.session_state.show_history = True
    
    tab1, tab2 = st.tabs(["💬 Ask KefGPT", "📝 Quiz Mode"])
    
    with tab1:
        if st.session_state.rag_engine is None:
            st.info("👈 Please initialize the engines from the sidebar first")
        else:
            st.header("Ask KefGPT")
            st.caption("Ask questions about your course materials")
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Ask a question about your course..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.rag_engine.query(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            st.session_state.session_manager.add_qa(
                                question=prompt,
                                answer=response,
                                course=selected_course
                            )
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with tab2:
        if st.session_state.quiz_engine is None:
            st.info("👈 Please initialize the engines from the sidebar first")
        else:
            st.header("Quiz Mode")
            st.caption("Generate quizzes from your course materials")
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_questions = st.number_input("Number of Questions", min_value=5, max_value=20, value=10)
                topic = st.text_input("Topic (optional)", placeholder="Leave empty for general quiz")
            
            if st.button("🎲 Generate Quiz", type="primary"):
                with st.spinner("Generating quiz..."):
                    try:
                        quiz = st.session_state.quiz_engine.generate_quiz(
                            num_questions=num_questions,
                            topic=topic if topic else None
                        )
                        st.session_state.current_quiz = quiz
                        st.session_state.quiz_answers = {}
                        st.success("Quiz generated!")
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")
            
            if "current_quiz" in st.session_state:
                st.divider()
                st.subheader("Quiz Questions")
                
                quiz = st.session_state.current_quiz
                user_answers = {}
                
                for i, question_data in enumerate(quiz["questions"], 1):
                    st.markdown(f"**Question {i}:** {question_data['question']}")
                    options = question_data['options']
                    user_answer = st.radio(
                        f"Select an answer for Question {i}",
                        options,
                        key=f"q_{i}",
                        label_visibility="collapsed",
                        index=None
                    )
                    user_answers[i] = user_answer
                
                st.session_state.quiz_answers = user_answers
                
                if st.button("Submit Quiz"):
                    score = st.session_state.quiz_engine.score_quiz(
                        quiz=quiz,
                        user_answers=user_answers
                    )
                    
                    st.session_state.quiz_score = score
                    st.balloons()
                    
                    st.session_state.session_manager.add_quiz_result(
                        course=selected_course,
                        quiz=quiz,
                        score=score
                    )
                
                if "quiz_score" in st.session_state:
                    score = st.session_state.quiz_score
                    st.divider()
                    st.subheader("Quiz Results")
                    st.metric("Score", f"{score['correct']}/{score['total']} ({score['percentage']:.1f}%)")
                    
                    with st.expander("View Correct Answers"):
                        for i, question_data in enumerate(quiz["questions"], 1):
                            correct = question_data['correct_answer']
                            user_ans = user_answers.get(i, "Not answered")
                            is_correct = user_ans == correct
                            status = "✅" if is_correct else "❌"
                            st.markdown(f"{status} **Q{i}:** Correct answer is **{correct}**")
                            if not is_correct:
                                st.caption(f"Your answer: {user_ans}")

    if st.session_state.get("show_history", False):
        st.sidebar.divider()
        with st.sidebar:
            st.subheader("Session History")
            history = st.session_state.session_manager.get_history()
            
            if history.get("qa_history"):
                st.write("**Q&A History:**")
                for qa in history["qa_history"][-5:]:  
                    with st.expander(f"Q: {qa['question'][:50]}..."):
                        st.write(f"**Q:** {qa['question']}")
                        st.write(f"**A:** {qa['answer']}")
                        st.caption(f"Course: {qa.get('course', 'N/A')} | {qa['timestamp']}")
            
            if history.get("quiz_results"):
                st.write("**Quiz Results:**")
                for quiz_result in history["quiz_results"][-5:]:  
                    score = quiz_result['score']
                    st.write(f"**{quiz_result['course']}:** {score['correct']}/{score['total']} ({score['percentage']:.1f}%)")
                    st.caption(quiz_result['timestamp'])
            
            if st.button("Close History"):
                st.session_state.show_history = False

if __name__ == "__main__":
    main()