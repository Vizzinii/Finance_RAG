import os
import sys
import json

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_manager import AppConfigManager
from data_manager.test_set_manager import TestSetManager
from eval_engine.hallucination_evaluator import HallucinationEvaluator
from eval_engine.prompt_manager import PromptTemplateManager
from eval_engine.result_exporter import (
    build_export_payload,
    flatten_results_for_csv,
    summarize_error_buckets
)
from knowledge_base.document_loader import DocumentLoader
from knowledge_base.vector_store_manager import VectorStoreManager
from rag_engine.financial_rag import FinancialRAG


st.set_page_config(page_title="金融幻觉评测系统", layout="wide")

PROVIDER_LABELS = {
    "OpenAI": "OpenAI",
    "Aliyun DashScope (Qwen)": "阿里云 DashScope (Qwen)",
    "Other": "其他"
}

RESULT_COLUMN_LABELS = {
    "id": "编号",
    "question": "问题",
    "candidate_answer": "候选回答",
    "expected_label": "真实标签",
    "predicted_label": "预测标签",
    "verdict": "判定结果",
    "reason": "判定原因",
    "source_model": "来源模型",
    "is_correct": "是否判对"
}

LABEL_DISPLAY_MAP = {
    "positive": "阳性（有幻觉）",
    "negative": "阴性（无幻觉）"
}

VERDICT_DISPLAY_MAP = {
    "supported": "证据支持",
    "hallucinated": "存在幻觉",
    "uncertain": "证据不足",
    "contradicted": "与证据矛盾",
    "insufficient_evidence": "证据不足"
}

BOOL_DISPLAY_MAP = {
    True: "是",
    False: "否"
}

SAMPLE_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_eval_dataset.json")
)
CONFIG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config", "app_config.json")
)
APP_CONFIG_MANAGER = AppConfigManager(CONFIG_FILE_PATH)
DEFAULT_APP_CONFIG = APP_CONFIG_MANAGER.load_config()
DEFAULT_RUNTIME_CONFIG = APP_CONFIG_MANAGER.get_runtime_config(DEFAULT_APP_CONFIG)


if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "rag_engine" not in st.session_state:
    st.session_state["rag_engine"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "eval_results" not in st.session_state:
    st.session_state["eval_results"] = []
if "eval_metrics" not in st.session_state:
    st.session_state["eval_metrics"] = {}
if "last_eval_mode" not in st.session_state:
    st.session_state["last_eval_mode"] = "overall"
if "last_eval_dataset_name" not in st.session_state:
    st.session_state["last_eval_dataset_name"] = ""
if "provider" not in st.session_state:
    st.session_state["provider"] = DEFAULT_RUNTIME_CONFIG["provider"]
if "base_url" not in st.session_state:
    st.session_state["base_url"] = DEFAULT_RUNTIME_CONFIG["base_url"]
if "chat_model_name" not in st.session_state:
    st.session_state["chat_model_name"] = DEFAULT_RUNTIME_CONFIG["chat_model_name"]
if "embedding_model_name" not in st.session_state:
    st.session_state["embedding_model_name"] = DEFAULT_RUNTIME_CONFIG["embedding_model_name"]
if "vector_store_directory" not in st.session_state:
    st.session_state["vector_store_directory"] = DEFAULT_RUNTIME_CONFIG["vector_store_directory"]
if "retrieval_top_k" not in st.session_state:
    st.session_state["retrieval_top_k"] = DEFAULT_RUNTIME_CONFIG["retrieval_top_k"]
if "api_key" not in st.session_state:
    st.session_state["api_key"] = DEFAULT_RUNTIME_CONFIG["api_key"]
if "overall_prompt" not in st.session_state:
    st.session_state["overall_prompt"] = HallucinationEvaluator.DEFAULT_OVERALL_PROMPT.strip()
if "claim_extraction_prompt" not in st.session_state:
    st.session_state["claim_extraction_prompt"] = (
        HallucinationEvaluator.DEFAULT_CLAIM_EXTRACTION_PROMPT.strip()
    )
if "claim_verification_prompt" not in st.session_state:
    st.session_state["claim_verification_prompt"] = (
        HallucinationEvaluator.DEFAULT_CLAIM_VERIFICATION_PROMPT.strip()
    )
if "active_prompt_template" not in st.session_state:
    st.session_state["active_prompt_template"] = PromptTemplateManager.DEFAULT_TEMPLATE_NAME
if "config_status_message" not in st.session_state:
    st.session_state["config_status_message"] = ""
if "_pending_runtime_config" not in st.session_state:
    st.session_state["_pending_runtime_config"] = None


pending_runtime_config = st.session_state.get("_pending_runtime_config")
if pending_runtime_config:
    st.session_state["provider"] = pending_runtime_config["provider"]
    st.session_state["base_url"] = pending_runtime_config["base_url"]
    st.session_state["chat_model_name"] = pending_runtime_config["chat_model_name"]
    st.session_state["embedding_model_name"] = pending_runtime_config["embedding_model_name"]
    st.session_state["api_key"] = pending_runtime_config["api_key"]
    st.session_state["vector_store_directory"] = pending_runtime_config["vector_store_directory"]
    st.session_state["retrieval_top_k"] = pending_runtime_config["retrieval_top_k"]
    st.session_state["_pending_runtime_config"] = None


def apply_prompt_template(prompts):
    st.session_state["overall_prompt"] = prompts["overall_prompt"]
    st.session_state["claim_extraction_prompt"] = prompts["claim_extraction_prompt"]
    st.session_state["claim_verification_prompt"] = prompts["claim_verification_prompt"]


def apply_runtime_config(config):
    st.session_state["provider"] = config["provider"]
    st.session_state["base_url"] = config["base_url"]
    st.session_state["chat_model_name"] = config["chat_model_name"]
    st.session_state["embedding_model_name"] = config["embedding_model_name"]
    st.session_state["api_key"] = config["api_key"]
    st.session_state["vector_store_directory"] = config["vector_store_directory"]
    st.session_state["retrieval_top_k"] = config["retrieval_top_k"]


def get_runtime_config():
    return {
        "provider": st.session_state["provider"],
        "base_url": st.session_state["base_url"],
        "chat_model_name": st.session_state["chat_model_name"],
        "embedding_model_name": st.session_state["embedding_model_name"],
        "api_key": st.session_state["api_key"],
        "vector_store_directory": st.session_state["vector_store_directory"],
        "retrieval_top_k": st.session_state["retrieval_top_k"]
    }


def apply_provider_preset():
    provider = st.session_state["provider"]
    preset = APP_CONFIG_MANAGER.get_provider_presets()[provider]
    st.session_state["base_url"] = preset["base_url"]
    st.session_state["chat_model_name"] = preset["chat_model_name"]
    st.session_state["embedding_model_name"] = preset["embedding_model_name"]


def format_label(label):
    return LABEL_DISPLAY_MAP.get(label, label)


def format_verdict(verdict):
    return VERDICT_DISPLAY_MAP.get(verdict, verdict)


def format_provider(provider):
    return PROVIDER_LABELS.get(provider, provider)


def localize_results_dataframe(df_results):
    localized = df_results.copy()
    for column in ["expected_label", "predicted_label"]:
        if column in localized.columns:
            localized[column] = localized[column].map(
                lambda value: format_label(value) if pd.notna(value) else value
            )
    if "verdict" in localized.columns:
        localized["verdict"] = localized["verdict"].map(
            lambda value: format_verdict(value) if pd.notna(value) else value
        )
    if "is_correct" in localized.columns:
        localized["is_correct"] = localized["is_correct"].map(
            lambda value: BOOL_DISPLAY_MAP.get(value, value) if pd.notna(value) else value
        )
    return localized.rename(columns=RESULT_COLUMN_LABELS)


def export_results_as_json(results, metrics, mode: str, dataset_name: str, runtime_config):
    sanitized_config = dict(runtime_config)
    if "api_key" in sanitized_config:
        sanitized_config["api_key"] = "***"
    payload = build_export_payload(
        results=results,
        metrics=metrics,
        mode=mode,
        dataset_name=dataset_name,
        config_snapshot=sanitized_config
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_results_as_csv(results):
    flattened = flatten_results_for_csv(results)
    if not flattened:
        return ""
    return pd.DataFrame(flattened).to_csv(index=False)


def ensure_vector_store(base_url: str, embed_model_name: str, api_key: str, persist_directory: str):
    if st.session_state["vector_store"] is None:
        st.session_state["vector_store"] = VectorStoreManager(
            persist_directory=persist_directory,
            base_url=base_url or None,
            model_name=embed_model_name,
            api_key=api_key
        )
    return st.session_state["vector_store"]


def ensure_rag_engine(
    base_url: str,
    chat_model_name: str,
    embed_model_name: str,
    api_key: str,
    persist_directory: str,
    retrieval_top_k: int
):
    vector_store = ensure_vector_store(base_url, embed_model_name, api_key, persist_directory)
    if st.session_state["rag_engine"] is None:
        st.session_state["rag_engine"] = FinancialRAG(
            vector_store,
            model_name=chat_model_name or None,
            base_url=base_url or None,
            timeout=120,
            api_key=api_key,
            retrieval_top_k=retrieval_top_k
        )
    return st.session_state["rag_engine"]


def render_eval_metrics(metrics):
    if not metrics:
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("准确率", f"{metrics['accuracy']:.2f}")
    col2.metric("精确率", f"{metrics['precision']:.2f}")
    col3.metric("召回率", f"{metrics['recall']:.2f}")
    col4.metric("F1", f"{metrics['f1']:.2f}")
    col5.metric("不确定占比", f"{metrics['uncertain_rate']:.2f}")


def render_eval_results(results, mode: str):
    if not results:
        return

    st.subheader("样本结果")
    df_results = pd.DataFrame(results)
    visible_columns = [
        "id",
        "question",
        "candidate_answer",
        "expected_label",
        "predicted_label",
        "verdict",
        "reason",
        "source_model",
        "is_correct"
    ]
    visible_columns = [column for column in visible_columns if column in df_results.columns]
    localized_results = localize_results_dataframe(df_results[visible_columns])
    st.dataframe(localized_results, use_container_width=True)

    if mode == "claim":
        st.subheader("Claim 级明细")
        for result in results:
            with st.expander(f"样本 {result['id']} - {format_verdict(result['verdict'])}"):
                st.write("问题：", result["question"])
                st.write("候选回答：", result["candidate_answer"])
                st.write("判定原因：", result["reason"])
                st.write("Claim 统计：")
                st.json(result.get("claim_counts", {}))
                for claim_result in result.get("claim_results", []):
                    st.text(
                        f"{format_verdict(claim_result['verdict'])} | {claim_result['claim']} | "
                        f"置信度: {claim_result['confidence']:.2f}"
                    )
                    if claim_result.get("evidence"):
                        st.text("证据： " + " | ".join(claim_result["evidence"]))
    else:
        st.subheader("证据摘要")
        for result in results:
            with st.expander(f"样本 {result['id']} - {format_verdict(result['verdict'])}"):
                st.write("问题：", result["question"])
                st.write("候选回答：", result["candidate_answer"])
                st.write("判定原因：", result["reason"])
                if result.get("evidence"):
                    st.text("证据： " + " | ".join(result["evidence"]))
                if result.get("unsupported_parts"):
                    st.text("缺乏支持的部分： " + " | ".join(result["unsupported_parts"]))


def render_error_analysis(results, mode: str):
    if not results:
        return

    buckets = summarize_error_buckets(results)
    st.subheader("错误分析")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("错误样本数", len(buckets["incorrect"]))
    col2.metric("不确定样本数", len(buckets["uncertain"]))
    col3.metric("假阳性", len(buckets["false_positive"]))
    col4.metric("假阴性", len(buckets["false_negative"]))

    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
        ["错误样本", "不确定样本", "误分类样本"]
    )

    with analysis_tab1:
        if not buckets["incorrect"]:
            st.info("当前评测中没有错误样本。")
        else:
            for result in buckets["incorrect"]:
                with st.expander(
                    f"样本 {result['id']} | 真实标签 {format_label(result['expected_label'])} | "
                    f"预测标签 {format_label(result['predicted_label'])}"
                ):
                    st.write("问题：", result["question"])
                    st.write("候选回答：", result["candidate_answer"])
                    st.write("判定原因：", result["reason"])
                    if mode == "claim":
                        for claim_result in result.get("claim_results", []):
                            st.text(
                                f"{format_verdict(claim_result['verdict'])} | {claim_result['claim']} | "
                                f"置信度: {claim_result['confidence']:.2f}"
                            )
                    else:
                        if result.get("unsupported_parts"):
                            st.text("缺乏支持的部分： " + " | ".join(result["unsupported_parts"]))
                        if result.get("evidence"):
                            st.text("证据： " + " | ".join(result["evidence"]))

    with analysis_tab2:
        if not buckets["uncertain"]:
            st.info("当前评测中没有不确定样本。")
        else:
            for result in buckets["uncertain"]:
                with st.expander(f"样本 {result['id']} | 证据不足"):
                    st.write("问题：", result["question"])
                    st.write("候选回答：", result["candidate_answer"])
                    st.write("判定原因：", result["reason"])
                    if mode == "claim":
                        st.json(result.get("claim_counts", {}))
                        for claim_result in result.get("claim_results", []):
                            st.text(
                                f"{format_verdict(claim_result['verdict'])} | {claim_result['claim']} | "
                                f"置信度: {claim_result['confidence']:.2f}"
                            )
                    else:
                        if result.get("evidence"):
                            st.text("证据： " + " | ".join(result["evidence"]))

    with analysis_tab3:
        misclassified = buckets["false_positive"] + buckets["false_negative"]
        if not misclassified:
            st.info("当前评测中没有假阳性或假阴性样本。")
        else:
            rows = [
                {
                    "id": result["id"],
                    "expected_label": result["expected_label"],
                    "predicted_label": result["predicted_label"],
                    "verdict": result["verdict"],
                    "reason": result["reason"]
                }
                for result in misclassified
            ]
            localized_misclassified = localize_results_dataframe(pd.DataFrame(rows))
            st.dataframe(localized_misclassified, use_container_width=True)


def main():
    st.title("金融场景 RAG 幻觉评测系统")
    prompt_manager = PromptTemplateManager()
    test_set_manager = TestSetManager()
    app_config = APP_CONFIG_MANAGER.load_config()
    provider_presets = APP_CONFIG_MANAGER.get_provider_presets(app_config)

    if not provider_presets:
        st.error("配置文件中未找到可用的模型服务商预设。")
        st.stop()

    if st.session_state["provider"] not in provider_presets:
        st.session_state["provider"] = next(iter(provider_presets))
        apply_provider_preset()

    with st.sidebar:
        st.header("系统配置")

        st.caption(f"配置文件：{CONFIG_FILE_PATH}")

        st.selectbox(
            "模型服务商",
            list(provider_presets.keys()),
            key="provider",
            on_change=apply_provider_preset,
            format_func=format_provider
        )
        st.text_input("接口地址", key="base_url")
        st.text_input("对话模型名称", key="chat_model_name")
        st.text_input("向量模型名称", key="embedding_model_name")
        st.text_input(
            "API Key",
            key="api_key",
            type="password",
            help="点击“保存配置”后会直接写入配置文件，仅建议在本地演示环境使用。"
        )
        st.text_input("向量库目录", key="vector_store_directory")
        st.number_input("检索 Top K", min_value=1, max_value=20, step=1, key="retrieval_top_k")

        config_col1, config_col2 = st.columns(2)
        if config_col1.button("保存配置", use_container_width=True):
            saved_config = APP_CONFIG_MANAGER.save_runtime_config(get_runtime_config())
            st.session_state["vector_store"] = None
            st.session_state["rag_engine"] = None
            st.session_state["config_status_message"] = "运行配置已保存。"
            st.session_state["_pending_runtime_config"] = APP_CONFIG_MANAGER.get_runtime_config(saved_config)
            st.rerun()
        if config_col2.button("重新加载", use_container_width=True):
            reloaded_config = APP_CONFIG_MANAGER.load_config()
            st.session_state["vector_store"] = None
            st.session_state["rag_engine"] = None
            st.session_state["config_status_message"] = "已从配置文件重新加载运行配置。"
            st.session_state["_pending_runtime_config"] = APP_CONFIG_MANAGER.get_runtime_config(reloaded_config)
            st.rerun()

        if st.session_state["config_status_message"]:
            st.success(st.session_state["config_status_message"])
            st.session_state["config_status_message"] = ""

        with st.expander("评测 Prompt 配置", expanded=False):
            template_names = prompt_manager.list_templates()
            selected_template = st.selectbox(
                "Prompt 模板",
                options=template_names,
                index=template_names.index(st.session_state["active_prompt_template"])
                if st.session_state["active_prompt_template"] in template_names
                else 0
            )

            col1, col2 = st.columns(2)
            if col1.button("加载模板", use_container_width=True):
                prompts = prompt_manager.load_template(selected_template)
                apply_prompt_template(prompts)
                st.session_state["active_prompt_template"] = selected_template
                st.rerun()
            if col2.button("恢复默认", use_container_width=True):
                apply_prompt_template(prompt_manager.get_default_prompts())
                st.session_state["active_prompt_template"] = PromptTemplateManager.DEFAULT_TEMPLATE_NAME
                st.rerun()

            save_template_name = st.text_input(
                "将当前 Prompt 另存为",
                value=st.session_state["active_prompt_template"]
            )

            st.session_state["overall_prompt"] = st.text_area(
                "整体判定 Prompt",
                value=st.session_state["overall_prompt"],
                height=260
            )
            st.session_state["claim_extraction_prompt"] = st.text_area(
                "Claim 拆解 Prompt",
                value=st.session_state["claim_extraction_prompt"],
                height=220
            )
            st.session_state["claim_verification_prompt"] = st.text_area(
                "Claim 核验 Prompt",
                value=st.session_state["claim_verification_prompt"],
                height=260
            )

            if st.button("保存 Prompt 模板", use_container_width=True):
                saved_name = prompt_manager.save_template(
                    save_template_name,
                    {
                        "overall_prompt": st.session_state["overall_prompt"],
                        "claim_extraction_prompt": st.session_state["claim_extraction_prompt"],
                        "claim_verification_prompt": st.session_state["claim_verification_prompt"]
                    }
                )
                st.session_state["active_prompt_template"] = saved_name
                st.success(f"已保存 Prompt 模板：{saved_name}")

        if st.button("重置运行状态"):
            st.session_state["vector_store"] = None
            st.session_state["rag_engine"] = None
            st.session_state["messages"] = []
            st.session_state["eval_results"] = []
            st.session_state["eval_metrics"] = {}
            st.session_state["last_eval_mode"] = "overall"
            st.session_state["last_eval_dataset_name"] = ""
            apply_prompt_template(prompt_manager.get_default_prompts())
            st.session_state["active_prompt_template"] = PromptTemplateManager.DEFAULT_TEMPLATE_NAME
            st.rerun()

        st.divider()
        st.write("系统状态")
        if st.session_state["vector_store"]:
            st.success("向量库已就绪")
        else:
            st.warning("向量库尚未初始化")

    runtime_config = get_runtime_config()
    base_url = runtime_config["base_url"]
    chat_model_name = runtime_config["chat_model_name"]
    embed_model_name = runtime_config["embedding_model_name"]
    vector_store_directory = runtime_config["vector_store_directory"]
    retrieval_top_k = int(runtime_config["retrieval_top_k"])
    api_key = runtime_config["api_key"].strip()

    tab1, tab2, tab3 = st.tabs(["问答", "数据", "评测"])

    with tab1:
        st.header("RAG 问答")
        if not api_key:
            st.error("请先在左侧边栏填写 API Key。")
        else:
            rag_engine = ensure_rag_engine(
                base_url,
                chat_model_name,
                embed_model_name,
                api_key,
                vector_store_directory,
                retrieval_top_k
            )

            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("请输入金融问题..."):
                st.chat_message("user").markdown(prompt)
                st.session_state["messages"].append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("正在生成回答..."):
                        try:
                            response = rag_engine.generate_answer(prompt)
                            answer = response["answer"]
                            sources = response["source_documents"]

                            st.markdown(answer)
                            with st.expander("参考证据"):
                                for index, doc in enumerate(sources, start=1):
                                    st.write(f"证据 {index}：{doc[:300]}...")

                            st.session_state["messages"].append(
                                {"role": "assistant", "content": answer}
                            )
                        except Exception as exc:
                            st.error(f"运行出错：{exc}")

    with tab2:
        st.header("数据管理")
        col1, col2 = st.columns(2)
        manager = test_set_manager

        with col1:
            st.subheader("知识库上传")
            kb_file = st.file_uploader(
                "上传 PDF、TXT、DOC 或 DOCX 文件",
                type=["pdf", "txt", "doc", "docx"],
                key="kb_file"
            )
            if kb_file and st.button("处理并加入知识库"):
                if not api_key:
                    st.error("向量化需要先提供 API Key。")
                else:
                    with st.spinner("正在处理文档..."):
                        temp_path = os.path.join("data", kb_file.name)
                        try:
                            with open(temp_path, "wb") as file:
                                file.write(kb_file.getbuffer())

                            content = DocumentLoader().load_file(temp_path)
                            vector_store = ensure_vector_store(
                                base_url,
                                embed_model_name,
                                api_key,
                                vector_store_directory
                            )
                            chunks = vector_store.text_splitter(content)
                            vector_store.add_documents(chunks)
                            st.success(f"已向知识库添加 {len(chunks)} 个文本切片。")
                        except Exception as exc:
                            st.error(f"文档处理失败：{exc}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

            st.subheader("导入评测数据集")
            dataset_file = st.file_uploader(
                "上传评测数据集 JSON",
                type=["json"],
                key="dataset_file"
            )
            if dataset_file and st.button("导入数据集 JSON"):
                try:
                    manager.import_json_text(dataset_file.getvalue().decode("utf-8"))
                    st.success("评测数据集导入成功。")
                except Exception as exc:
                    st.error(f"评测数据集导入失败：{exc}")

            sample_col1, sample_col2 = st.columns(2)
            if sample_col1.button("加载示例数据集", use_container_width=True):
                try:
                    manager.import_json_file(SAMPLE_DATASET_PATH)
                    st.success("示例数据集已加载到当前工作区。")
                except Exception as exc:
                    st.error(f"示例数据集加载失败：{exc}")
            if os.path.exists(SAMPLE_DATASET_PATH):
                with open(SAMPLE_DATASET_PATH, "rb") as sample_file:
                    sample_col2.download_button(
                        "下载示例 JSON",
                        data=sample_file.read(),
                        file_name="sample_eval_dataset.json",
                        mime="application/json",
                        use_container_width=True
                    )

            dataset = manager.get_dataset()
            st.caption(
                f"数据集：{dataset.get('dataset_name', '')} | "
                f"知识库版本：{dataset.get('kb_version', '') or '未填写'} | "
                f"样本数：{len(dataset.get('samples', []))}"
            )

        with col2:
            st.subheader("新增评测样本")
            with st.form("add_case_form"):
                question = st.text_input("问题")
                candidate_answer = st.text_area("候选回答")
                label = st.selectbox(
                    "标签",
                    ["negative", "positive"],
                    format_func=format_label
                )
                source_model = st.text_input("来源模型", value="manual")
                source_type = st.text_input("来源类型", value="manual")
                ground_truth = st.text_area("标准答案（可选）")
                reference_docs = st.text_area(
                    "参考证据（每行一条，可选）"
                )
                notes = st.text_area("备注（可选）")
                submitted = st.form_submit_button("添加样本")
                if submitted:
                    try:
                        manager.add_case(
                            question=question,
                            candidate_answer=candidate_answer,
                            label=label,
                            source_model=source_model,
                            source_type=source_type,
                            ground_truth=ground_truth,
                            reference_docs=[
                                line.strip()
                                for line in reference_docs.splitlines()
                                if line.strip()
                            ],
                            notes=notes
                        )
                        st.success("样本已添加。")
                    except Exception as exc:
                        st.error(f"添加样本失败：{exc}")

        samples = manager.get_all_cases()
        st.subheader("当前评测样本")
        if samples:
            df_cases = pd.DataFrame(samples)
            localized_cases = df_cases.copy()
            if "label" in localized_cases.columns:
                localized_cases["label"] = localized_cases["label"].map(format_label)
            localized_cases = localized_cases.rename(
                columns={
                    "id": "编号",
                    "question": "问题",
                    "candidate_answer": "候选回答",
                    "label": "标签",
                    "source_model": "来源模型",
                    "source_type": "来源类型",
                    "reference_docs": "参考证据",
                    "ground_truth": "标准答案",
                    "notes": "备注"
                }
            )
            st.dataframe(localized_cases, use_container_width=True)
        else:
            st.info("当前还没有评测样本。")

    with tab3:
        st.header("评测面板")
        eval_mode = st.radio(
            "评测模式",
            options=["overall", "claim"],
            format_func=lambda value: "整体判定" if value == "overall" else "Claim 级核验",
            horizontal=True
        )

        if st.button("运行评测"):
            dataset = TestSetManager().get_dataset()
            samples = dataset.get("samples", [])

            if not api_key:
                st.error("请先在左侧边栏填写 API Key。")
            elif not samples:
                st.warning("当前没有可评测的样本。")
            else:
                try:
                    rag_engine = ensure_rag_engine(
                        base_url,
                        chat_model_name,
                        embed_model_name,
                        api_key,
                        vector_store_directory,
                        retrieval_top_k
                    )
                    evaluator = HallucinationEvaluator(
                        model_name=chat_model_name or None,
                        base_url=base_url or None,
                        timeout=120,
                        api_key=api_key,
                        retrieval_top_k=retrieval_top_k,
                        overall_prompt=st.session_state["overall_prompt"],
                        claim_extraction_prompt=st.session_state["claim_extraction_prompt"],
                        claim_verification_prompt=st.session_state["claim_verification_prompt"]
                    )

                    with st.spinner(f"正在评测 {len(samples)} 条样本..."):
                        results = evaluator.run_batch_eval(dataset, rag_engine, mode=eval_mode)
                        metrics = evaluator.calculate_classification_metrics(results)

                    st.session_state["eval_results"] = results
                    st.session_state["eval_metrics"] = metrics
                    st.session_state["last_eval_mode"] = eval_mode
                    st.session_state["last_eval_dataset_name"] = dataset.get("dataset_name", "")
                except Exception as exc:
                    st.error(f"评测失败：{exc}")

        render_eval_metrics(st.session_state["eval_metrics"])
        if st.session_state["eval_results"]:
            export_col1, export_col2 = st.columns(2)
            json_payload = export_results_as_json(
                st.session_state["eval_results"],
                st.session_state["eval_metrics"],
                st.session_state["last_eval_mode"],
                st.session_state["last_eval_dataset_name"],
                runtime_config
            )
            csv_payload = export_results_as_csv(st.session_state["eval_results"])
            export_col1.download_button(
                "下载评测结果 JSON",
                data=json_payload,
                file_name="evaluation_results.json",
                mime="application/json",
                use_container_width=True
            )
            export_col2.download_button(
                "下载评测结果 CSV",
                data=csv_payload,
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        render_eval_results(st.session_state["eval_results"], st.session_state["last_eval_mode"])
        render_error_analysis(st.session_state["eval_results"], st.session_state["last_eval_mode"])


if __name__ == "__main__":
    main()
