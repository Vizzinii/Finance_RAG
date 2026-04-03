import os
import sys
import json
import html

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

APP_THEME_CSS = """
<style>
:root {
    --bg-main: #0f1218;
    --bg-surface: #171b24;
    --bg-elevated: #1d2330;
    --border-strong: rgba(148, 163, 184, 0.18);
    --border-soft: rgba(148, 163, 184, 0.10);
    --text-main: #f8fafc;
    --text-body: #e2e8f0;
    --text-muted: #94a3b8;
    --text-faint: #64748b;
    --brand: #2563eb;
    --brand-soft: rgba(37, 99, 235, 0.14);
    --success: #059669;
    --success-soft: rgba(5, 150, 105, 0.16);
    --danger: #dc2626;
    --danger-soft: rgba(220, 38, 38, 0.18);
    --warning: #d97706;
    --warning-soft: rgba(217, 119, 6, 0.18);
    --shadow-soft: 0 16px 40px rgba(15, 18, 24, 0.28);
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 30%),
        radial-gradient(circle at top left, rgba(5, 150, 105, 0.08), transparent 28%),
        var(--bg-main);
    color: var(--text-body);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131925 0%, #10151d 100%);
    border-right: 1px solid var(--border-soft);
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-body);
}

h1, h2, h3, h4 {
    color: var(--text-main);
    letter-spacing: -0.02em;
}

.app-hero {
    padding: 1.6rem 1.8rem;
    border-radius: 22px;
    border: 1px solid var(--border-strong);
    background:
        linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(15, 18, 24, 0.90) 40%),
        linear-gradient(180deg, #171b24 0%, #131822 100%);
    box-shadow: var(--shadow-soft);
    margin-bottom: 1rem;
}

.app-hero .eyebrow {
    margin: 0 0 0.4rem 0;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #93c5fd;
}

.app-hero h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
}

.app-hero p {
    margin: 0.6rem 0 0;
    color: var(--text-muted);
    font-size: 0.98rem;
    line-height: 1.65;
}

.section-intro {
    margin: 0.2rem 0 0.9rem;
}

.section-intro h3 {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
}

.section-intro p {
    margin: 0.35rem 0 0;
    color: var(--text-muted);
    font-size: 0.92rem;
    line-height: 1.6;
}

.micro-card {
    padding: 1rem 1.05rem;
    border-radius: 18px;
    border: 1px solid var(--border-soft);
    background: linear-gradient(180deg, rgba(29, 35, 48, 0.95), rgba(23, 27, 36, 0.95));
    min-height: 132px;
}

.micro-card .label {
    margin: 0;
    font-size: 0.82rem;
    color: var(--text-muted);
}

.micro-card .value {
    margin: 0.5rem 0 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-main);
}

.micro-card .hint {
    margin: 0.45rem 0 0;
    font-size: 0.82rem;
    line-height: 1.55;
    color: var(--text-faint);
}

.micro-card.primary { box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.18); }
.micro-card.success { box-shadow: inset 0 0 0 1px rgba(5, 150, 105, 0.18); }
.micro-card.warning { box-shadow: inset 0 0 0 1px rgba(217, 119, 6, 0.18); }
.micro-card.danger { box-shadow: inset 0 0 0 1px rgba(220, 38, 38, 0.18); }

.mode-card {
    padding: 1rem 1.05rem;
    border-radius: 18px;
    border: 1px solid var(--border-soft);
    background: rgba(23, 27, 36, 0.88);
    min-height: 158px;
}

.mode-card.selected {
    border-color: rgba(37, 99, 235, 0.55);
    box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.15), 0 12px 28px rgba(37, 99, 235, 0.12);
    background: linear-gradient(180deg, rgba(37, 99, 235, 0.10), rgba(23, 27, 36, 0.98));
}

.mode-card .kicker {
    font-size: 0.8rem;
    color: #93c5fd;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.mode-card h4 {
    margin: 0.35rem 0 0;
    font-size: 1.08rem;
}

.mode-card p {
    margin: 0.45rem 0 0;
    color: var(--text-muted);
    line-height: 1.6;
    font-size: 0.9rem;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    line-height: 1;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}

.badge-primary { background: var(--brand-soft); color: #bfdbfe; }
.badge-success { background: var(--success-soft); color: #a7f3d0; }
.badge-warning { background: var(--warning-soft); color: #fcd34d; }
.badge-danger { background: var(--danger-soft); color: #fecaca; }
.badge-neutral { background: rgba(148, 163, 184, 0.12); color: var(--text-body); }

.detail-card {
    padding: 0.95rem 1rem;
    border-radius: 16px;
    border: 1px solid var(--border-soft);
    background: rgba(29, 35, 48, 0.72);
    margin: 0.5rem 0 0.8rem;
}

.detail-card p {
    margin: 0.15rem 0 0;
    color: var(--text-muted);
    line-height: 1.65;
}

.detail-card strong {
    color: var(--text-main);
}

.empty-state {
    padding: 1.15rem 1.2rem;
    border-radius: 18px;
    border: 1px dashed var(--border-strong);
    background: rgba(23, 27, 36, 0.72);
}

.empty-state h4 {
    margin: 0;
    font-size: 1rem;
}

.empty-state p {
    margin: 0.4rem 0 0;
    color: var(--text-muted);
    line-height: 1.6;
}

.block-note {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.6;
}

div[data-testid="stMetric"] {
    border: 1px solid var(--border-soft);
    background: linear-gradient(180deg, rgba(23, 27, 36, 0.94), rgba(29, 35, 48, 0.90));
    border-radius: 18px;
    padding: 0.9rem 1rem;
}

div[data-testid="stFileUploader"] section {
    background: rgba(29, 35, 48, 0.78);
    border: 1px dashed rgba(37, 99, 235, 0.35);
    border-radius: 18px;
}

div[data-testid="stFileUploader"] section:hover {
    border-color: rgba(37, 99, 235, 0.70);
    background: rgba(37, 99, 235, 0.08);
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
div[data-testid="stTextArea"] textarea {
    background: var(--bg-elevated) !important;
    border-color: var(--border-soft) !important;
    color: var(--text-body) !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within,
div[data-testid="stNumberInput"] div[data-baseweb="input"] > div:focus-within,
div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(37, 99, 235, 0.75) !important;
    box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.18), 0 0 0 4px rgba(37, 99, 235, 0.08);
}

div.stButton > button,
div.stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(37, 99, 235, 0.30);
    min-height: 2.7rem;
    font-weight: 600;
    transition: all 0.18s ease;
}

div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
}

div.stButton > button[kind="secondary"],
div.stDownloadButton > button[kind="secondary"] {
    background: rgba(29, 35, 48, 0.84);
    color: var(--text-body);
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(37, 99, 235, 0.55);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1.2rem;
    border-bottom: 1px solid var(--border-soft);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding-left: 0;
    padding-right: 0;
    color: var(--text-muted);
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: var(--text-main);
    border-bottom: 2px solid var(--brand);
}

div[data-testid="stExpander"] {
    border-radius: 16px;
    border: 1px solid var(--border-soft);
    background: rgba(23, 27, 36, 0.72);
}

div[data-testid="stAlert"] {
    border-radius: 16px;
}
</style>
"""

SAMPLE_COLUMN_LABELS = {
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


def inject_theme():
    st.markdown(APP_THEME_CSS, unsafe_allow_html=True)


def safe_html(value):
    return html.escape("" if value is None else str(value))


def render_page_hero(title: str, description: str):
    st.markdown(
        f"""
        <div class="app-hero">
            <p class="eyebrow">Finance RAG Evaluation Workspace</p>
            <h1>{safe_html(title)}</h1>
            <p>{safe_html(description)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section_intro(title: str, description: str):
    st.markdown(
        f"""
        <div class="section-intro">
            <h3>{safe_html(title)}</h3>
            <p>{safe_html(description)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_micro_cards(items):
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        tone = item.get("tone", "primary")
        column.markdown(
            f"""
            <div class="micro-card {tone}">
                <p class="label">{safe_html(item['label'])}</p>
                <p class="value">{safe_html(item['value'])}</p>
                <p class="hint">{safe_html(item['hint'])}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_empty_state(title: str, description: str):
    st.markdown(
        f"""
        <div class="empty-state">
            <h4>{safe_html(title)}</h4>
            <p>{safe_html(description)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def semantic_badge(text: str, tone: str = "neutral"):
    return f'<span class="status-badge badge-{tone}">{safe_html(text)}</span>'


def label_tone(label):
    return {
        "positive": "danger",
        "negative": "success"
    }.get(label, "neutral")


def verdict_tone(verdict):
    return {
        "supported": "success",
        "hallucinated": "danger",
        "uncertain": "warning",
        "contradicted": "danger",
        "insufficient_evidence": "warning"
    }.get(verdict, "neutral")


def semantic_cell_style(value):
    if isinstance(value, list):
        return "color: #e2e8f0;"
    if isinstance(value, dict):
        return "color: #e2e8f0;"
    if value in {format_label("positive"), format_verdict("hallucinated"), format_verdict("contradicted"), "否"}:
        return "background-color: rgba(220, 38, 38, 0.18); color: #fecaca; font-weight: 600;"
    if value in {format_label("negative"), format_verdict("supported"), "是"}:
        return "background-color: rgba(5, 150, 105, 0.16); color: #a7f3d0; font-weight: 600;"
    if value in {format_verdict("uncertain"), format_verdict("insufficient_evidence")}:
        return "background-color: rgba(217, 119, 6, 0.16); color: #fde68a; font-weight: 600;"
    return "color: #e2e8f0;"


def build_semantic_styler(df: pd.DataFrame):
    return (
        df.style
        .set_properties(**{
            "border-color": "rgba(148, 163, 184, 0.10)",
            "color": "#e2e8f0",
            "white-space": "pre-wrap"
        })
        .applymap(semantic_cell_style)
    )


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
if "data_status_message" not in st.session_state:
    st.session_state["data_status_message"] = ""
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


def localize_samples_dataframe(df_samples):
    localized = df_samples.copy()
    if "label" in localized.columns:
        localized["label"] = localized["label"].map(
            lambda value: format_label(value) if pd.notna(value) else value
        )
    if "reference_docs" in localized.columns:
        localized["reference_docs"] = localized["reference_docs"].map(
            lambda value: " | ".join(value) if isinstance(value, list) else value
        )
    return localized.rename(columns=SAMPLE_COLUMN_LABELS)


def render_result_badges(result):
    badges = [
        semantic_badge(f"真实标签：{format_label(result['expected_label'])}", label_tone(result["expected_label"])),
        semantic_badge(f"预测标签：{format_label(result['predicted_label'])}", label_tone(result["predicted_label"])),
        semantic_badge(f"判定结果：{format_verdict(result['verdict'])}", verdict_tone(result["verdict"]))
    ]
    if result.get("is_correct") is not None:
        badges.append(
            semantic_badge(
                f"是否判对：{BOOL_DISPLAY_MAP.get(result['is_correct'], result['is_correct'])}",
                "success" if result["is_correct"] else "danger"
            )
        )
    st.markdown("".join(badges), unsafe_allow_html=True)


def render_detail_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="detail-card">
            <strong>{safe_html(title)}</strong>
            <p>{safe_html(body)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


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
    render_section_intro(
        "评测结果总览",
        "先看核心指标，再下钻到样本级证据和错误归因，形成完整的评测闭环。"
    )
    render_micro_cards([
        {"label": "准确率", "value": f"{metrics['accuracy']:.2f}", "hint": "样本级整体预测正确率。", "tone": "primary"},
        {"label": "精确率", "value": f"{metrics['precision']:.2f}", "hint": "被判为有幻觉的回答中，有多少确实为阳性。", "tone": "danger"},
        {"label": "召回率", "value": f"{metrics['recall']:.2f}", "hint": "系统检出阳性样本的能力。", "tone": "success"},
        {"label": "F1", "value": f"{metrics['f1']:.2f}", "hint": "精确率与召回率的综合平衡指标。", "tone": "primary"},
        {"label": "不确定占比", "value": f"{metrics['uncertain_rate']:.2f}", "hint": "证据不足或判定保持谨慎的比例。", "tone": "warning"},
    ])


def render_eval_results(results, mode: str):
    if not results:
        return
    render_section_intro(
        "样本级评测结果",
        "这里展示真实标签、预测标签、判定结果与理由，便于快速核查系统是否把错误样本抓出来。"
    )
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
    st.dataframe(build_semantic_styler(localized_results), width="stretch", hide_index=True)

    if mode == "claim":
        render_section_intro(
            "Claim 级核验明细",
            "逐条查看回答中的事实断言被判为支持、矛盾还是证据不足，从而定位具体错误位点。"
        )
        for result in results:
            with st.expander(f"样本 {result['id']} - {format_verdict(result['verdict'])}", expanded=False):
                render_result_badges(result)
                render_detail_card("问题", result["question"])
                render_detail_card("候选回答", result["candidate_answer"])
                render_detail_card("判定原因", result["reason"])
                claim_counts = result.get("claim_counts", {})
                render_micro_cards([
                    {"label": "支持 Claim", "value": str(claim_counts.get("supported", 0)), "hint": "被知识库证据明确支持。", "tone": "success"},
                    {"label": "矛盾 Claim", "value": str(claim_counts.get("contradicted", 0)), "hint": "与检索证据存在直接冲突。", "tone": "danger"},
                    {"label": "证据不足 Claim", "value": str(claim_counts.get("insufficient_evidence", 0)), "hint": "证据无法完成支持或反驳。", "tone": "warning"},
                ])
                for claim_result in result.get("claim_results", []):
                    st.markdown(
                        "".join([
                            semantic_badge(format_verdict(claim_result["verdict"]), verdict_tone(claim_result["verdict"])),
                            semantic_badge(f"置信度：{claim_result['confidence']:.2f}", "neutral")
                        ]),
                        unsafe_allow_html=True
                    )
                    render_detail_card("Claim", claim_result["claim"])
                    if claim_result.get("reason"):
                        render_detail_card("核验说明", claim_result["reason"])
                    if claim_result.get("evidence"):
                        render_detail_card("对应证据", " | ".join(claim_result["evidence"]))
    else:
        render_section_intro(
            "证据摘要与判定理由",
            "聚焦整体判定模式下每条回答的结论、证据片段以及缺乏支持的部分。"
        )
        for result in results:
            with st.expander(f"样本 {result['id']} - {format_verdict(result['verdict'])}", expanded=False):
                render_result_badges(result)
                render_detail_card("问题", result["question"])
                render_detail_card("候选回答", result["candidate_answer"])
                render_detail_card("判定原因", result["reason"])
                if result.get("evidence"):
                    render_detail_card("命中证据", " | ".join(result["evidence"]))
                if result.get("unsupported_parts"):
                    render_detail_card("缺乏支持的部分", " | ".join(result["unsupported_parts"]))


def render_error_analysis(results, mode: str):
    if not results:
        return

    buckets = summarize_error_buckets(results)
    render_section_intro(
        "错误分析与可解释诊断",
        "把错误样本、不确定样本和误分类样本拆开看，帮助我们快速发现提示词、知识库或模型能力的薄弱环节。"
    )

    render_micro_cards([
        {"label": "错误样本数", "value": str(len(buckets["incorrect"])), "hint": "预测结果与真实标签不一致的样本。", "tone": "danger"},
        {"label": "不确定样本数", "value": str(len(buckets["uncertain"])), "hint": "证据不足或需要继续人工复核。", "tone": "warning"},
        {"label": "假阳性", "value": str(len(buckets["false_positive"])), "hint": "把正常回答误判为有幻觉。", "tone": "danger"},
        {"label": "假阴性", "value": str(len(buckets["false_negative"])), "hint": "把有幻觉回答漏判为正常。", "tone": "warning"},
    ])

    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
        ["错误样本", "不确定样本", "误分类样本"]
    )

    with analysis_tab1:
        if not buckets["incorrect"]:
            render_empty_state("当前没有错误样本", "这批评测结果里，系统没有出现真实标签与预测标签冲突的情况。")
        else:
            for result in buckets["incorrect"]:
                with st.expander(
                    f"样本 {result['id']} | 真实标签 {format_label(result['expected_label'])} | "
                    f"预测标签 {format_label(result['predicted_label'])}"
                ):
                    render_result_badges(result)
                    render_detail_card("问题", result["question"])
                    render_detail_card("候选回答", result["candidate_answer"])
                    render_detail_card("判定原因", result["reason"])
                    if mode == "claim":
                        for claim_result in result.get("claim_results", []):
                            st.markdown(
                                "".join([
                                    semantic_badge(format_verdict(claim_result["verdict"]), verdict_tone(claim_result["verdict"])),
                                    semantic_badge(f"置信度：{claim_result['confidence']:.2f}", "neutral")
                                ]),
                                unsafe_allow_html=True
                            )
                            render_detail_card("Claim", claim_result["claim"])
                            if claim_result.get("evidence"):
                                render_detail_card("对应证据", " | ".join(claim_result["evidence"]))
                    else:
                        if result.get("unsupported_parts"):
                            render_detail_card("缺乏支持的部分", " | ".join(result["unsupported_parts"]))
                        if result.get("evidence"):
                            render_detail_card("命中证据", " | ".join(result["evidence"]))

    with analysis_tab2:
        if not buckets["uncertain"]:
            render_empty_state("当前没有不确定样本", "这批结果里没有进入“证据不足/无法稳定判定”的样本。")
        else:
            for result in buckets["uncertain"]:
                with st.expander(f"样本 {result['id']} | 证据不足"):
                    render_result_badges(result)
                    render_detail_card("问题", result["question"])
                    render_detail_card("候选回答", result["candidate_answer"])
                    render_detail_card("判定原因", result["reason"])
                    if mode == "claim":
                        claim_counts = result.get("claim_counts", {})
                        render_micro_cards([
                            {"label": "支持 Claim", "value": str(claim_counts.get("supported", 0)), "hint": "被证据支持。", "tone": "success"},
                            {"label": "矛盾 Claim", "value": str(claim_counts.get("contradicted", 0)), "hint": "与证据冲突。", "tone": "danger"},
                            {"label": "证据不足 Claim", "value": str(claim_counts.get("insufficient_evidence", 0)), "hint": "仍需补充证据。", "tone": "warning"},
                        ])
                        for claim_result in result.get("claim_results", []):
                            st.markdown(
                                "".join([
                                    semantic_badge(format_verdict(claim_result["verdict"]), verdict_tone(claim_result["verdict"])),
                                    semantic_badge(f"置信度：{claim_result['confidence']:.2f}", "neutral")
                                ]),
                                unsafe_allow_html=True
                            )
                            render_detail_card("Claim", claim_result["claim"])
                            if claim_result.get("evidence"):
                                render_detail_card("对应证据", " | ".join(claim_result["evidence"]))
                    else:
                        if result.get("evidence"):
                            render_detail_card("命中证据", " | ".join(result["evidence"]))

    with analysis_tab3:
        misclassified = buckets["false_positive"] + buckets["false_negative"]
        if not misclassified:
            render_empty_state("当前没有误分类样本", "系统暂时没有出现假阳性或假阴性，可继续扩大样本规模做稳定性验证。")
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
            st.dataframe(build_semantic_styler(localized_misclassified), width="stretch", hide_index=True)


def main():
    inject_theme()
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

    render_page_hero(
        "金融场景 RAG 幻觉评测系统",
        "围绕金融知识库、结构化评测样本和可解释的证据核验结果，构建一套面向金融场景的 RAG 问答与幻觉评测评测工作台。"
    )

    with st.sidebar:
        st.header("系统配置")
        st.caption(f"配置文件：{CONFIG_FILE_PATH}")

        with st.container(border=True):
            render_section_intro("模型服务配置", "统一管理对话模型、向量模型与 API Key，便于本地演示时快速切换。")
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

        with st.container(border=True):
            render_section_intro("向量与检索配置", "控制知识库存储路径和检索深度，影响问答与评测时的证据召回表现。")
            st.text_input("向量库目录", key="vector_store_directory")
            st.number_input("检索 Top K", min_value=1, max_value=20, step=1, key="retrieval_top_k")

            config_col1, config_col2 = st.columns(2)
            if config_col1.button("保存配置", type="primary", width="stretch"):
                saved_config = APP_CONFIG_MANAGER.save_runtime_config(get_runtime_config())
                st.session_state["vector_store"] = None
                st.session_state["rag_engine"] = None
                st.session_state["config_status_message"] = "运行配置已保存。"
                st.session_state["_pending_runtime_config"] = APP_CONFIG_MANAGER.get_runtime_config(saved_config)
                st.rerun()
            if config_col2.button("重新加载", width="stretch"):
                reloaded_config = APP_CONFIG_MANAGER.load_config()
                st.session_state["vector_store"] = None
                st.session_state["rag_engine"] = None
                st.session_state["config_status_message"] = "已从配置文件重新加载运行配置。"
                st.session_state["_pending_runtime_config"] = APP_CONFIG_MANAGER.get_runtime_config(reloaded_config)
                st.rerun()

            if st.session_state["config_status_message"]:
                st.success(st.session_state["config_status_message"])
                st.session_state["config_status_message"] = ""

        with st.container(border=True):
            render_section_intro("评测 Prompt 配置", "支持整体判定、Claim 拆解与 Claim 核验三类提示词模板，方便针对不同实验口径做对比。")
            template_names = prompt_manager.list_templates()
            selected_template = st.selectbox(
                "Prompt 模板",
                options=template_names,
                index=template_names.index(st.session_state["active_prompt_template"])
                if st.session_state["active_prompt_template"] in template_names
                else 0
            )

            col1, col2 = st.columns(2)
            if col1.button("加载模板", width="stretch"):
                prompts = prompt_manager.load_template(selected_template)
                apply_prompt_template(prompts)
                st.session_state["active_prompt_template"] = selected_template
                st.rerun()
            if col2.button("恢复默认", width="stretch"):
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
                height=220
            )
            st.session_state["claim_extraction_prompt"] = st.text_area(
                "Claim 拆解 Prompt",
                value=st.session_state["claim_extraction_prompt"],
                height=180
            )
            st.session_state["claim_verification_prompt"] = st.text_area(
                "Claim 核验 Prompt",
                value=st.session_state["claim_verification_prompt"],
                height=220
            )

            if st.button("保存 Prompt 模板", width="stretch"):
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

        with st.container(border=True):
            render_section_intro("运行状态", "帮助我们快速确认当前配置是否生效，以及知识库是否已经初始化。")
            if st.session_state["vector_store"]:
                st.markdown(
                    semantic_badge("向量库已就绪", "success") +
                    semantic_badge("可直接问答与评测", "primary"),
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    semantic_badge("向量库尚未初始化", "warning") +
                    semantic_badge("需要先上传知识文档", "neutral"),
                    unsafe_allow_html=True
                )

            if st.button("重置运行状态", width="stretch"):
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

    runtime_config = get_runtime_config()
    base_url = runtime_config["base_url"]
    chat_model_name = runtime_config["chat_model_name"]
    embed_model_name = runtime_config["embedding_model_name"]
    vector_store_directory = runtime_config["vector_store_directory"]
    retrieval_top_k = int(runtime_config["retrieval_top_k"])
    api_key = runtime_config["api_key"].strip()

    tab1, tab2, tab3 = st.tabs(["问答", "数据", "评测"])

    with tab1:
        render_section_intro(
            "RAG 问答工作台",
            "输入金融问题后，系统会先从知识库检索证据，再生成回答，适合用于演示知识检索与问答联动。"
        )
        if not api_key:
            render_empty_state("尚未配置 API Key", "请先在左侧边栏完成模型配置，随后即可开始问答与评测。")
        else:
            rag_engine = ensure_rag_engine(
                base_url,
                chat_model_name,
                embed_model_name,
                api_key,
                vector_store_directory,
                retrieval_top_k
            )

            render_micro_cards([
                {"label": "模型服务商", "value": format_provider(st.session_state['provider']), "hint": "当前用于生成回答与评测的模型服务入口。", "tone": "primary"},
                {"label": "对话模型", "value": chat_model_name or "未配置", "hint": "问答与评测主模型名称。", "tone": "primary"},
                {"label": "检索 Top K", "value": str(retrieval_top_k), "hint": "每次检索召回的候选证据数。", "tone": "warning"},
            ])

            if not st.session_state["messages"]:
                render_empty_state(
                    "可以开始提问了",
                    "建议先从财报指标、政策结论、业务增长原因等短问题开始，便于观察检索证据和回答生成效果。"
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
        render_section_intro(
            "数据准备与评测集管理",
            "先上传知识库文档，再导入或手工补充评测样本，形成完整的评测输入闭环。"
        )
        manager = test_set_manager
        dataset = manager.get_dataset()
        samples = manager.get_all_cases()
        positive_count = len([sample for sample in samples if sample.get("label") == "positive"])
        negative_count = len([sample for sample in samples if sample.get("label") == "negative"])

        if st.session_state["data_status_message"]:
            st.success(st.session_state["data_status_message"])
            st.session_state["data_status_message"] = ""

        render_micro_cards([
            {
                "label": "当前数据集",
                "value": dataset.get("dataset_name", "未命名数据集") or "未命名数据集",
                "hint": "导入后将在当前工作区持续使用。",
                "tone": "primary"
            },
            {
                "label": "知识库版本",
                "value": dataset.get("kb_version", "") or "未填写",
                "hint": "建议与知识文档版本保持一致，便于答辩说明。",
                "tone": "warning"
            },
            {
                "label": "样本总数",
                "value": str(len(samples)),
                "hint": "当前已导入或手工新增的评测样本总量。",
                "tone": "success"
            },
            {
                "label": "阳性 / 阴性",
                "value": f"{positive_count} / {negative_count}",
                "hint": "阳性表示有幻觉，阴性表示事实一致。",
                "tone": "danger"
            }
        ])

        col1, col2 = st.columns([1.05, 0.95], gap="large")

        with col1:
            with st.container(border=True):
                render_section_intro(
                    "知识库上传",
                    "上传 PDF、TXT、DOC 或 DOCX 文档后，系统会切分文本并写入向量库，供问答与评测共同使用。"
                )
                kb_file = st.file_uploader(
                    "上传知识库文件",
                    type=["pdf", "txt", "doc", "docx"],
                    key="kb_file"
                )
                st.caption("建议优先使用整理过的核心事实文本，能显著降低演示时的检索噪声。")
                if kb_file and st.button("处理并加入知识库", type="primary", width="stretch"):
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

            with st.container(border=True):
                render_section_intro(
                    "导入评测数据集",
                    "导入结构化 JSON 评测集后，就可以直接切到评测页执行整体判定或 Claim 级核验。"
                )
                dataset_file = st.file_uploader(
                    "上传评测数据集 JSON",
                    type=["json"],
                    key="dataset_file"
                )
                if dataset_file and st.button("导入数据集 JSON", type="primary", width="stretch"):
                    try:
                        manager.import_json_text(dataset_file.getvalue().decode("utf-8"))
                        st.session_state["data_status_message"] = "评测数据集导入成功。"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"评测数据集导入失败：{exc}")

                sample_col1, sample_col2 = st.columns(2)
                if sample_col1.button("加载示例数据集", width="stretch"):
                    try:
                        manager.import_json_file(SAMPLE_DATASET_PATH)
                        st.session_state["data_status_message"] = "示例数据集已加载到当前工作区。"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"示例数据集加载失败：{exc}")
                if os.path.exists(SAMPLE_DATASET_PATH):
                    with open(SAMPLE_DATASET_PATH, "rb") as sample_file:
                        sample_col2.download_button(
                            "下载示例 JSON",
                            data=sample_file.read(),
                            file_name="sample_eval_dataset.json",
                            mime="application/json",
                            width="stretch"
                        )

        with col2:
            with st.container(border=True):
                render_section_intro(
                    "快速新增评测样本",
                    "支持手工补充单条样本，适合演示前快速添加阳性/阴性案例或修正数据集缺失字段。"
                )
                with st.form("add_case_form"):
                    question = st.text_area("问题 *", height=90, placeholder="例如：2024年末我国M2余额和同比增速分别是多少？")
                    candidate_answer = st.text_area("候选回答 *", height=120)

                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    label = meta_col1.selectbox(
                        "标签 *",
                        ["negative", "positive"],
                        format_func=format_label
                    )
                    source_model = meta_col2.text_input("来源模型", value="manual")
                    source_type = meta_col3.text_input("来源类型", value="manual")

                    ground_truth = st.text_area("标准答案（可选）", height=90)
                    reference_docs = st.text_area(
                        "参考证据（每行一条，可选）",
                        height=120
                    )
                    notes = st.text_area("备注（可选）", height=100)
                    submitted = st.form_submit_button("添加样本", type="primary", width="stretch")
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
                            st.session_state["data_status_message"] = "样本已添加。"
                            st.rerun()
                        except Exception as exc:
                            st.error(f"添加样本失败：{exc}")

        with st.container(border=True):
            render_section_intro(
                "当前评测样本",
                "表格聚合展示当前工作区样本，适合在导入后检查字段是否齐全、标签是否平衡。"
            )
            if samples:
                df_cases = pd.DataFrame(samples)
                visible_case_columns = [
                    "id",
                    "question",
                    "candidate_answer",
                    "label",
                    "source_model",
                    "source_type",
                    "ground_truth",
                    "reference_docs",
                    "notes"
                ]
                visible_case_columns = [column for column in visible_case_columns if column in df_cases.columns]
                localized_cases = localize_samples_dataframe(df_cases[visible_case_columns])
                st.dataframe(build_semantic_styler(localized_cases), width="stretch", hide_index=True)
            else:
                render_empty_state("当前还没有评测样本", "请先导入 JSON 数据集，或在右侧快速新增样本后再继续评测。")

    with tab3:
        render_section_intro(
            "评测面板",
            "基于知识库检索结果，对候选回答进行真实性评测。支持整体判定和 Claim 级核验两种模式。"
        )
        dataset = test_set_manager.get_dataset()
        samples = dataset.get("samples", [])
        eval_mode = st.radio(
            "评测模式",
            options=["overall", "claim"],
            format_func=lambda value: "整体判定" if value == "overall" else "Claim 级核验",
            horizontal=True,
            label_visibility="collapsed"
        )

        mode_col1, mode_col2 = st.columns(2, gap="large")
        mode_col1.markdown(
            f"""
            <div class="mode-card {'selected' if eval_mode == 'overall' else ''}">
                <div class="kicker">Overall Verdict</div>
                <h4>整体判定</h4>
                <p>对整条候选回答做真实性二分类，快速输出是否存在幻觉，适合批量跑通演示链路。</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        mode_col2.markdown(
            f"""
            <div class="mode-card {'selected' if eval_mode == 'claim' else ''}">
                <div class="kicker">Claim Verification</div>
                <h4>Claim 级核验</h4>
                <p>先拆解回答中的事实断言，再逐条核验证据支持情况，适合定位错误位点与解释原因。</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.container(border=True):
            run_col1, run_col2 = st.columns([1.35, 0.65], gap="large")
            with run_col1:
                render_section_intro(
                    "运行说明",
                    "当前评测将基于工作区数据集执行证据检索与真实性判断。运行完成后，页面会自动展示指标、样本结果与错误分析。"
                )
                st.markdown(
                    "".join([
                        semantic_badge(f"当前模式：{'整体判定' if eval_mode == 'overall' else 'Claim 级核验'}", "primary"),
                        semantic_badge(f"待评测样本：{len(samples)} 条", "neutral"),
                        semantic_badge(f"数据集：{dataset.get('dataset_name', '未命名数据集') or '未命名数据集'}", "neutral"),
                    ]),
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<p class='block-note'>整体判定更适合快速跑通流程，Claim 级核验更适合展示系统的可解释性与错误定位能力。</p>",
                    unsafe_allow_html=True
                )
            with run_col2:
                run_eval = st.button("运行评测", type="primary", width="stretch")

        if run_eval:
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

        if st.session_state["eval_results"]:
            with st.container(border=True):
                render_eval_metrics(st.session_state["eval_metrics"])

            with st.container(border=True):
                render_section_intro(
                    "结果导出",
                    "支持将当前批次评测结果导出为 JSON 或 CSV，便于论文留档、二次分析或展示汇报。"
                )
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
                    width="stretch"
                )
                export_col2.download_button(
                    "下载评测结果 CSV",
                    data=csv_payload,
                    file_name="evaluation_results.csv",
                    mime="text/csv",
                    width="stretch"
                )

            with st.container(border=True):
                render_eval_results(st.session_state["eval_results"], st.session_state["last_eval_mode"])
            with st.container(border=True):
                render_error_analysis(st.session_state["eval_results"], st.session_state["last_eval_mode"])
        else:
            render_empty_state(
                "还没有评测结果",
                "请先在“数据”页准备样本并完成模型配置，然后回到这里运行评测，结果会按指标、样本与错误分析三层展示。"
            )


if __name__ == "__main__":
    main()
