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


st.set_page_config(page_title="Financial Hallucination Evaluation", layout="wide")
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
    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    col4.metric("F1", f"{metrics['f1']:.2f}")
    col5.metric("Uncertain Rate", f"{metrics['uncertain_rate']:.2f}")


def render_eval_results(results, mode: str):
    if not results:
        return

    st.subheader("Sample Results")
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
    st.dataframe(df_results[visible_columns], use_container_width=True)

    if mode == "claim":
        st.subheader("Claim-Level Details")
        for result in results:
            with st.expander(f"Sample {result['id']} - {result['verdict']}"):
                st.write("Question:", result["question"])
                st.write("Candidate answer:", result["candidate_answer"])
                st.write("Reason:", result["reason"])
                st.write("Claim counts:")
                st.json(result.get("claim_counts", {}))
                for claim_result in result.get("claim_results", []):
                    st.text(
                        f"{claim_result['verdict']} | {claim_result['claim']} | "
                        f"confidence: {claim_result['confidence']:.2f}"
                    )
                    if claim_result.get("evidence"):
                        st.text("Evidence: " + " | ".join(claim_result["evidence"]))
    else:
        st.subheader("Evidence Highlights")
        for result in results:
            with st.expander(f"Sample {result['id']} - {result['verdict']}"):
                st.write("Question:", result["question"])
                st.write("Candidate answer:", result["candidate_answer"])
                st.write("Reason:", result["reason"])
                if result.get("evidence"):
                    st.text("Evidence: " + " | ".join(result["evidence"]))
                if result.get("unsupported_parts"):
                    st.text("Unsupported parts: " + " | ".join(result["unsupported_parts"]))


def render_error_analysis(results, mode: str):
    if not results:
        return

    buckets = summarize_error_buckets(results)
    st.subheader("Error Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Incorrect Cases", len(buckets["incorrect"]))
    col2.metric("Uncertain Cases", len(buckets["uncertain"]))
    col3.metric("False Positives", len(buckets["false_positive"]))
    col4.metric("False Negatives", len(buckets["false_negative"]))

    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
        ["Incorrect", "Uncertain", "Misclassified Labels"]
    )

    with analysis_tab1:
        if not buckets["incorrect"]:
            st.info("No incorrect cases found in the current run.")
        else:
            for result in buckets["incorrect"]:
                with st.expander(f"Case {result['id']} | expected {result['expected_label']} | predicted {result['predicted_label']}"):
                    st.write("Question:", result["question"])
                    st.write("Candidate answer:", result["candidate_answer"])
                    st.write("Reason:", result["reason"])
                    if mode == "claim":
                        for claim_result in result.get("claim_results", []):
                            st.text(
                                f"{claim_result['verdict']} | {claim_result['claim']} | "
                                f"confidence: {claim_result['confidence']:.2f}"
                            )
                    else:
                        if result.get("unsupported_parts"):
                            st.text("Unsupported parts: " + " | ".join(result["unsupported_parts"]))
                        if result.get("evidence"):
                            st.text("Evidence: " + " | ".join(result["evidence"]))

    with analysis_tab2:
        if not buckets["uncertain"]:
            st.info("No uncertain cases found in the current run.")
        else:
            for result in buckets["uncertain"]:
                with st.expander(f"Case {result['id']} | uncertain"):
                    st.write("Question:", result["question"])
                    st.write("Candidate answer:", result["candidate_answer"])
                    st.write("Reason:", result["reason"])
                    if mode == "claim":
                        st.json(result.get("claim_counts", {}))
                        for claim_result in result.get("claim_results", []):
                            st.text(
                                f"{claim_result['verdict']} | {claim_result['claim']} | "
                                f"confidence: {claim_result['confidence']:.2f}"
                            )
                    else:
                        if result.get("evidence"):
                            st.text("Evidence: " + " | ".join(result["evidence"]))

    with analysis_tab3:
        misclassified = buckets["false_positive"] + buckets["false_negative"]
        if not misclassified:
            st.info("No false positives or false negatives found in the current run.")
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main():
    st.title("Financial Hallucination Evaluation System")
    prompt_manager = PromptTemplateManager()
    test_set_manager = TestSetManager()
    app_config = APP_CONFIG_MANAGER.load_config()
    provider_presets = APP_CONFIG_MANAGER.get_provider_presets(app_config)

    if not provider_presets:
        st.error("No provider presets were found in the config file.")
        st.stop()

    if st.session_state["provider"] not in provider_presets:
        st.session_state["provider"] = next(iter(provider_presets))
        apply_provider_preset()

    with st.sidebar:
        st.header("Configuration")

        st.caption(f"Config file: {CONFIG_FILE_PATH}")

        st.selectbox(
            "Provider",
            list(provider_presets.keys()),
            key="provider",
            on_change=apply_provider_preset
        )
        st.text_input("Base URL", key="base_url")
        st.text_input("Chat Model Name", key="chat_model_name")
        st.text_input("Embedding Model Name", key="embedding_model_name")
        st.text_input(
            "API Key",
            key="api_key",
            type="password",
            help="Stored directly in the config file when you click Save Config."
        )
        st.text_input("Vector Store Directory", key="vector_store_directory")
        st.number_input("Retrieval Top K", min_value=1, max_value=20, step=1, key="retrieval_top_k")

        config_col1, config_col2 = st.columns(2)
        if config_col1.button("Save Config", use_container_width=True):
            saved_config = APP_CONFIG_MANAGER.save_runtime_config(get_runtime_config())
            st.session_state["vector_store"] = None
            st.session_state["rag_engine"] = None
            st.session_state["config_status_message"] = "Saved runtime config."
            st.session_state["_pending_runtime_config"] = APP_CONFIG_MANAGER.get_runtime_config(saved_config)
            st.rerun()
        if config_col2.button("Reload Config", use_container_width=True):
            reloaded_config = APP_CONFIG_MANAGER.load_config()
            st.session_state["vector_store"] = None
            st.session_state["rag_engine"] = None
            st.session_state["config_status_message"] = "Reloaded runtime config from file."
            st.session_state["_pending_runtime_config"] = APP_CONFIG_MANAGER.get_runtime_config(reloaded_config)
            st.rerun()

        if st.session_state["config_status_message"]:
            st.success(st.session_state["config_status_message"])
            st.session_state["config_status_message"] = ""

        with st.expander("Evaluation Prompts", expanded=False):
            template_names = prompt_manager.list_templates()
            selected_template = st.selectbox(
                "Prompt Template",
                options=template_names,
                index=template_names.index(st.session_state["active_prompt_template"])
                if st.session_state["active_prompt_template"] in template_names
                else 0
            )

            col1, col2 = st.columns(2)
            if col1.button("Load Template", use_container_width=True):
                prompts = prompt_manager.load_template(selected_template)
                apply_prompt_template(prompts)
                st.session_state["active_prompt_template"] = selected_template
                st.rerun()
            if col2.button("Restore Default", use_container_width=True):
                apply_prompt_template(prompt_manager.get_default_prompts())
                st.session_state["active_prompt_template"] = PromptTemplateManager.DEFAULT_TEMPLATE_NAME
                st.rerun()

            save_template_name = st.text_input(
                "Save Current Prompts As",
                value=st.session_state["active_prompt_template"]
            )

            st.session_state["overall_prompt"] = st.text_area(
                "Overall Prompt",
                value=st.session_state["overall_prompt"],
                height=260
            )
            st.session_state["claim_extraction_prompt"] = st.text_area(
                "Claim Extraction Prompt",
                value=st.session_state["claim_extraction_prompt"],
                height=220
            )
            st.session_state["claim_verification_prompt"] = st.text_area(
                "Claim Verification Prompt",
                value=st.session_state["claim_verification_prompt"],
                height=260
            )

            if st.button("Save Prompt Template", use_container_width=True):
                saved_name = prompt_manager.save_template(
                    save_template_name,
                    {
                        "overall_prompt": st.session_state["overall_prompt"],
                        "claim_extraction_prompt": st.session_state["claim_extraction_prompt"],
                        "claim_verification_prompt": st.session_state["claim_verification_prompt"]
                    }
                )
                st.session_state["active_prompt_template"] = saved_name
                st.success(f"Saved prompt template: {saved_name}")

        if st.button("Reset Runtime"):
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
        st.write("System Status")
        if st.session_state["vector_store"]:
            st.success("Vector store ready")
        else:
            st.warning("Vector store not initialized")

    runtime_config = get_runtime_config()
    base_url = runtime_config["base_url"]
    chat_model_name = runtime_config["chat_model_name"]
    embed_model_name = runtime_config["embedding_model_name"]
    vector_store_directory = runtime_config["vector_store_directory"]
    retrieval_top_k = int(runtime_config["retrieval_top_k"])
    api_key = runtime_config["api_key"].strip()

    tab1, tab2, tab3 = st.tabs(["Chat", "Data", "Eval"])

    with tab1:
        st.header("RAG Chat")
        if not api_key:
            st.error("Please provide an API key in the sidebar.")
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

            if prompt := st.chat_input("Ask a finance question..."):
                st.chat_message("user").markdown(prompt)
                st.session_state["messages"].append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Generating answer..."):
                        try:
                            response = rag_engine.generate_answer(prompt)
                            answer = response["answer"]
                            sources = response["source_documents"]

                            st.markdown(answer)
                            with st.expander("Sources"):
                                for index, doc in enumerate(sources, start=1):
                                    st.write(f"Source {index}: {doc[:300]}...")

                            st.session_state["messages"].append(
                                {"role": "assistant", "content": answer}
                            )
                        except Exception as exc:
                            st.error(f"Error: {exc}")

    with tab2:
        st.header("Data Management")
        col1, col2 = st.columns(2)
        manager = test_set_manager

        with col1:
            st.subheader("Knowledge Base Upload")
            kb_file = st.file_uploader(
                "Upload PDF, TXT, DOC, or DOCX",
                type=["pdf", "txt", "doc", "docx"],
                key="kb_file"
            )
            if kb_file and st.button("Process and Add to KB"):
                if not api_key:
                    st.error("Embedding requires an API key.")
                else:
                    with st.spinner("Processing document..."):
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
                            st.success(f"Added {len(chunks)} chunks to the knowledge base.")
                        except Exception as exc:
                            st.error(f"Failed to process file: {exc}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

            st.subheader("Import Evaluation Dataset")
            dataset_file = st.file_uploader(
                "Upload dataset JSON",
                type=["json"],
                key="dataset_file"
            )
            if dataset_file and st.button("Import Dataset JSON"):
                try:
                    manager.import_json_text(dataset_file.getvalue().decode("utf-8"))
                    st.success("Dataset imported successfully.")
                except Exception as exc:
                    st.error(f"Failed to import dataset: {exc}")

            sample_col1, sample_col2 = st.columns(2)
            if sample_col1.button("Load Sample Dataset", use_container_width=True):
                try:
                    manager.import_json_file(SAMPLE_DATASET_PATH)
                    st.success("Sample dataset loaded into the workspace.")
                except Exception as exc:
                    st.error(f"Failed to load sample dataset: {exc}")
            if os.path.exists(SAMPLE_DATASET_PATH):
                with open(SAMPLE_DATASET_PATH, "rb") as sample_file:
                    sample_col2.download_button(
                        "Download Sample JSON",
                        data=sample_file.read(),
                        file_name="sample_eval_dataset.json",
                        mime="application/json",
                        use_container_width=True
                    )

            dataset = manager.get_dataset()
            st.caption(
                f"Dataset: {dataset.get('dataset_name', '')} | "
                f"KB Version: {dataset.get('kb_version', '') or 'N/A'} | "
                f"Samples: {len(dataset.get('samples', []))}"
            )

        with col2:
            st.subheader("Add Evaluation Sample")
            with st.form("add_case_form"):
                question = st.text_input("Question")
                candidate_answer = st.text_area("Candidate Answer")
                label = st.selectbox("Label", ["negative", "positive"])
                source_model = st.text_input("Source Model", value="manual")
                source_type = st.text_input("Source Type", value="manual")
                ground_truth = st.text_area("Ground Truth (Optional)")
                reference_docs = st.text_area(
                    "Reference Docs (one per line, optional)"
                )
                notes = st.text_area("Notes (Optional)")
                submitted = st.form_submit_button("Add Sample")
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
                        st.success("Sample added.")
                    except Exception as exc:
                        st.error(f"Failed to add sample: {exc}")

        samples = manager.get_all_cases()
        st.subheader("Current Evaluation Samples")
        if samples:
            df_cases = pd.DataFrame(samples)
            st.dataframe(df_cases, use_container_width=True)
        else:
            st.info("No evaluation samples found.")

    with tab3:
        st.header("Evaluation Dashboard")
        eval_mode = st.radio(
            "Evaluation Mode",
            options=["overall", "claim"],
            format_func=lambda value: "Overall Verdict" if value == "overall" else "Claim-Level Verification",
            horizontal=True
        )

        if st.button("Run Evaluation"):
            dataset = TestSetManager().get_dataset()
            samples = dataset.get("samples", [])

            if not api_key:
                st.error("Please provide an API key in the sidebar.")
            elif not samples:
                st.warning("No evaluation samples found.")
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

                    with st.spinner(f"Evaluating {len(samples)} samples..."):
                        results = evaluator.run_batch_eval(dataset, rag_engine, mode=eval_mode)
                        metrics = evaluator.calculate_classification_metrics(results)

                    st.session_state["eval_results"] = results
                    st.session_state["eval_metrics"] = metrics
                    st.session_state["last_eval_mode"] = eval_mode
                    st.session_state["last_eval_dataset_name"] = dataset.get("dataset_name", "")
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

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
                "Download Results JSON",
                data=json_payload,
                file_name="evaluation_results.json",
                mime="application/json",
                use_container_width=True
            )
            export_col2.download_button(
                "Download Results CSV",
                data=csv_payload,
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        render_eval_results(st.session_state["eval_results"], st.session_state["last_eval_mode"])
        render_error_analysis(st.session_state["eval_results"], st.session_state["last_eval_mode"])


if __name__ == "__main__":
    main()
