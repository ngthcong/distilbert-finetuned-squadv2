import streamlit as st
from transformers import pipeline
import logging
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        logger.info("Đang tải mô hình ...")
        try:
            self.qa_pipeline = pipeline(config.PIPELINE_NAME, model=config.MODEL_NAME)
            logger.info("✅ Mô hình được tải thành công.")
        except Exception as e:
            logger.error(f"❌ Không thể tải mô hình: {e}")
            raise e

    def query(self, context, question):
        logger.info(f"Xử lý câu hỏi: {question}")
        try:
            result = self.qa_pipeline(question=question, context=context)
            logger.info(f"Kết quả truy vấn: {result['answer']}")
            return result["answer"]
        except Exception as e:
            logger.error(f"❌ Lỗi khi truy vấn: {e}")
            raise e

@st.cache_resource
def load_model():
    try:
        model = Model()
        st.session_state.model = model
        st.sidebar.success("✅ Mô hình đã được tải!")
    except Exception as e:
        st.sidebar.error("❌ Không thể tải mô hình. Xem log.")
        logger.exception("Không thể khởi tạo mô hình.")


def process_query(model, context, query):
    try:
        result = model.query(context, query)
        return result
    except Exception as e:
        st.error("⚠️ Đã xảy ra lỗi khi xử lý truy vấn.")
        logger.exception("Lỗi trong quá trình xử lý truy vấn.")
        return None


def render_sidebar():
    st.sidebar.title("🤖 AI Model Interface")
    st.sidebar.write("Model: **DistilBERT Finetuned**")

    if st.sidebar.button("Load Model"):
        load_model()


def render_main_app():
    st.title("📝 AI Contextual Query App")

    context = st.text_area("Context:", key="context")
    query_disabled = context.strip() == ""
    query = st.text_area("Query:", disabled=query_disabled, key="query")

    if st.button("Xử lý"):
        if not context.strip():
            st.warning("Vui lòng nhập Context trước khi xử lý.")
            logger.warning("Người dùng nhấn xử lý mà chưa nhập Context.")
        elif not query.strip():
            st.warning("Vui lòng nhập Query.")
            logger.warning("Người dùng nhấn xử lý mà chưa nhập Query.")
        elif "model" not in st.session_state or st.session_state.model is None:
            st.error("Model chưa được load. Hãy nhấn 'Load Model' trong sidebar.")
            logger.warning("Model chưa được load.")
        else:
            logger.info("Bắt đầu xử lý truy vấn...")
            result = process_query(st.session_state.model, context, query)
            if result:
                st.success("🎯 Kết quả:")
                st.write(result)


def main():
    if "model" not in st.session_state:
        st.session_state.model = None

    render_sidebar()
    render_main_app()


if __name__ == "__main__":
    main()