import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knowledge_base.document_loader import DocumentLoader
from knowledge_base.vector_store_manager import VectorStoreManager
from rag_engine.financial_rag import FinancialRAG
from data_manager.test_set_manager import TestSetManager
from eval_engine.hallucination_evaluator import HallucinationEvaluator

# Page Config
st.set_page_config(page_title="金融 RAG 评估系统 (Financial RAG Evaluation System)", layout="wide")

# Session State Initialization
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'rag_engine' not in st.session_state:
    st.session_state['rag_engine'] = None

def main():
    st.title("面向金融场景的 RAG 大模型幻觉评估系统 (Financial RAG Evaluation System)")

    # --- Sidebar ---
    with st.sidebar:
        st.header("配置 (Configuration)")
        
        # API Key Input with auto-detection from env
        default_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        api_key = st.text_input("API Key (OpenAI / DashScope)", value=default_api_key, type="password")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            # Also set DashScope key if needed by other libs, though we use OpenAI interface
            os.environ["DASHSCOPE_API_KEY"] = api_key
            
        # Provider Selection
        provider = st.selectbox("模型服务商 (Provider)", ["OpenAI", "Aliyun DashScope (Qwen)", "Other"], index=1)

        if provider == "OpenAI":
            default_base_url = "https://api.openai.com/v1"
            default_chat_model = "gpt-3.5-turbo"
            default_embed_model = "text-embedding-ada-002"
        elif provider == "Aliyun DashScope (Qwen)":
            default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            default_chat_model = "qwen-plus"
            default_embed_model = "text-embedding-v1"
        else:
            default_base_url = "http://localhost:8000/v1"
            default_chat_model = "meta-llama/Llama-2-7b-chat-hf"
            default_embed_model = "text-embedding-ada-002"

        # Add Base URL and Model Name configuration
        base_url = st.text_input("Base URL", value=default_base_url)
        chat_model_name = st.text_input("Chat Model Name", value=default_chat_model)
        embed_model_name = st.text_input("Embedding Model Name", value=default_embed_model)
        
        if st.button("更新配置 & 重置系统 (Update Config & Reset)"):
            st.session_state['vector_store'] = None
            st.session_state['rag_engine'] = None
            st.rerun()
        
        st.info("请确保已设置 API Key。如果环境变量中存在 DASHSCOPE_API_KEY，系统会自动加载。")
        
        st.divider()
        st.write("系统状态 (System Status):")
        if st.session_state['vector_store']:
            st.success("向量库: 就绪 (Ready)")
        else:
            st.warning("向量库: 未初始化 (Not Initialized)")

    # --- Main Tabs ---
    tab1, tab2, tab3 = st.tabs(["问财·探索 (Chat)", "数据管理 (Data)", "评测看板 (Eval)"])

    # --- Tab 1: Chat ---
    with tab1:
        st.header("金融 RAG 对话 (Financial RAG Chat)")
        
        # Initialize RAG if ready
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("请在侧边栏输入 OpenAI API Key。")
        else:
            if st.session_state['vector_store'] is None:
                 # Initialize with default path and optional base_url
                 st.session_state['vector_store'] = VectorStoreManager(
                     base_url=base_url if base_url else None,
                     model_name=embed_model_name,
                     api_key=api_key
                 )
            
            if st.session_state['rag_engine'] is None:
                st.session_state['rag_engine'] = FinancialRAG(
                    st.session_state['vector_store'],
                    model_name=chat_model_name if chat_model_name else "gpt-3.5-turbo",
                    base_url=base_url if base_url else None,
                    timeout=120,  # Increased timeout
                    api_key=api_key
                )

            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("请输入金融相关问题 (Question)..."):
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("正在生成回答 (Generating answer)..."):
                        try:
                            response = st.session_state['rag_engine'].generate_answer(prompt)
                            answer = response['answer']
                            sources = response['source_documents']
                            
                            st.markdown(answer)
                            
                            with st.expander("参考来源 (Sources)"):
                                for i, doc in enumerate(sources):
                                    st.write(f"**来源 (Source) {i+1}:** {doc[:200]}...")
                                    
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

    # --- Tab 2: Data Management ---
    with tab2:
        st.header("数据管理 (Data Management)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("知识库上传 (Knowledge Base Upload)")
            uploaded_file = st.file_uploader("上传金融研报 (PDF/TXT)", type=['pdf', 'txt'])
            if uploaded_file and st.button("处理并添加到知识库 (Process & Add to KB)"):
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("Embedding 需要 API Key。")
                else:
                    with st.spinner("正在处理文档 (Processing document)..."):
                        # Save temp file
                        temp_path = os.path.join("data", uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load and Process
                        loader = DocumentLoader()
                        try:
                            content = loader.load_file(temp_path)
                            
                            if st.session_state['vector_store'] is None:
                                st.session_state['vector_store'] = VectorStoreManager(
                                    base_url=base_url if base_url else None,
                                    model_name=embed_model_name,
                                    api_key=api_key
                                )
                                
                            vsm = st.session_state['vector_store']
                            chunks = vsm.text_splitter(content)
                            vsm.add_documents(chunks)
                            st.success(f"成功添加 {len(chunks)} 个切片到知识库。")
                            
                            # Clean up temp file
                            os.remove(temp_path)
                        except Exception as e:
                            st.error(f"处理文件出错: {e}")

        with col2:
            st.subheader("测试集管理 (Test Set Management)")
            mgr = TestSetManager()
            
            # Add new case
            with st.form("add_case_form"):
                st.write("添加新测试用例 (Add New Test Case)")
                q = st.text_input("问题 (Question)")
                gt = st.text_area("标准答案 (Ground Truth - Optional)")
                submitted = st.form_submit_button("添加用例 (Add Case)")
                if submitted and q:
                    mgr.add_case(q, gt)
                    st.success("用例已添加! (Case added!)")
            
            # Display cases
            cases = mgr.get_all_cases()
            if cases:
                df_cases = pd.DataFrame(cases).rename(columns={
                    'question': '问题 (Question)', 
                    'ground_truth': '标准答案 (Ground Truth)', 
                    'id': 'ID'
                })
                st.dataframe(df_cases)
            else:
                st.info("未找到测试用例 (No test cases found).")

    # --- Tab 3: Evaluation ---
    with tab3:
        st.header("评测看板 (Evaluation Dashboard)")
        
        if st.button("运行测试集评测 (Run Evaluation on Test Set)"):
            if not os.environ.get("OPENAI_API_KEY"):
                st.error("需要 API Key。")
            elif not st.session_state['rag_engine']:
                st.error("RAG 引擎未初始化，请先访问对话 (Chat) 标签页。")
            else:
                mgr = TestSetManager()
                cases = mgr.get_all_cases()
                if not cases:
                    st.warning("没有可评测的测试用例。")
                else:
                    evaluator = HallucinationEvaluator(
                        model_name=chat_model_name if chat_model_name else "gpt-3.5-turbo",
                        base_url=base_url if base_url else None,
                        timeout=120,
                        api_key=api_key
                    )
                    with st.spinner(f"正在评测 {len(cases)} 个用例..."):
                        results = evaluator.run_batch_eval(cases, st.session_state['rag_engine'])
                        scores = evaluator.calculate_score(results)
                    
                    st.success("评测完成! (Evaluation Complete!)")
                    
                    # Metrics Display
                    m1, m2 = st.columns(2)
                    m1.metric("平均信实度 (Avg Faithfulness)", f"{scores['faithfulness']:.2f}")
                    m2.metric("平均答案相关性 (Avg Answer Relevance)", f"{scores['relevance']:.2f}")
                    
                    # Radar Chart
                    df_scores = pd.DataFrame(dict(
                        r=[scores['faithfulness'], scores['relevance']],
                        theta=['信实度 (Faithfulness)', '相关性 (Relevance)']
                    ))
                    fig = px.line_polar(df_scores, r='r', theta='theta', line_close=True, range_r=[0,1])
                    st.plotly_chart(fig)
                    
                    # Detailed Results
                    st.subheader("详细结果 (Detailed Results)")
                    df_results = pd.DataFrame(results).rename(columns={
                        'question': '问题 (Question)', 
                        'answer': '回答 (Answer)', 
                        'faithfulness': '信实度 (Faithfulness)', 
                        'relevance': '相关性 (Relevance)', 
                        'ground_truth': '标准答案 (Ground Truth)', 
                        'id': 'ID'
                    })
                    st.dataframe(df_results)

if __name__ == "__main__":
    main()
