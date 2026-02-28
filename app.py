from __future__ import annotations

import streamlit as st


def initialize_state() -> None:
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []


def render_sidebar() -> tuple[str, int, float, int]:
	st.sidebar.header("Settings")

	model_name = st.sidebar.selectbox(
		"LLM Model",
		options=["llama3.1:8b", "mistral:7b", "phi3:mini"],
		index=0,
	)

	top_k = st.sidebar.slider("Top K Chunks", min_value=1, max_value=10, value=4)
	temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)
	chunk_size = st.sidebar.select_slider("Chunk Size", options=[256, 512, 768, 1024], value=512)

	return model_name, top_k, temperature, chunk_size


def main() -> None:
	st.set_page_config(page_title="DeepDoc", page_icon="📄", layout="wide")
	initialize_state()

	st.title("📄 DeepDoc")
	st.caption("Ask questions from your PDF documents with local RAG + Ollama")

	model_name, top_k, temperature, chunk_size = render_sidebar()

	left_col, right_col = st.columns([2, 3])

	with left_col:
		st.subheader("1) Upload PDFs")
		uploaded_files = st.file_uploader(
			"Upload one or more PDF files",
			type=["pdf"],
			accept_multiple_files=True,
		)

		if uploaded_files:
			st.success(f"{len(uploaded_files)} file(s) ready")
			with st.expander("View selected files", expanded=False):
				for file in uploaded_files:
					st.write(f"• {file.name}")

		st.subheader("2) Build / Refresh Index")
		if st.button("Process Documents", use_container_width=True):
			if not uploaded_files:
				st.warning("Please upload at least one PDF first.")
			else:
				with st.spinner("Processing documents..."):
					st.info(
						"UI ready. Connect this button to `engine.processor` + `engine.retriever` logic."
					)

		st.markdown("---")
		st.write("**Current Config**")
		st.write(f"- Model: `{model_name}`")
		st.write(f"- Top K: `{top_k}`")
		st.write(f"- Temperature: `{temperature}`")
		st.write(f"- Chunk Size: `{chunk_size}`")

	with right_col:
		st.subheader("3) Ask Questions")
		question = st.text_area(
			"Your Question",
			placeholder="Ask something about the uploaded PDFs...",
			height=120,
		)

		ask_clicked = st.button("Get Answer", type="primary", use_container_width=True)

		st.subheader("Answer")
		if ask_clicked:
			if not question.strip():
				st.warning("Please enter a question.")
			else:
				with st.spinner("Generating answer..."):
					answer_text = (
						"UI is working. Connect this section to `engine.llm_chain` for actual RAG responses."
					)

				st.session_state.chat_history.append(
					{
						"question": question.strip(),
						"answer": answer_text,
					}
				)

				st.success(answer_text)

		st.subheader("History")
		if not st.session_state.chat_history:
			st.caption("No questions asked yet.")
		else:
			for index, item in enumerate(reversed(st.session_state.chat_history), start=1):
				with st.expander(f"Q{index}: {item['question'][:80]}", expanded=False):
					st.write(item["answer"])


if __name__ == "__main__":
	main()
